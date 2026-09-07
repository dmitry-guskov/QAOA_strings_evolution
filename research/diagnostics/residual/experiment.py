"""Pointwise exact residual continuation. No threshold pruning; cap aborts.
Float64 truncation algebra only; no outward rounding certificate.
"""
import sys
sys.dont_write_bytecode=True
from functools import lru_cache
import hashlib,importlib.util,json,math,pathlib,statistics,time
import numpy as np
BASE=pathlib.Path(__file__).resolve().parent

@lru_cache(maxsize=131072)
def rotated(gx,gz,x,z):
    if ((gx&z).bit_count()+(gz&x).bit_count())%2==0:return None
    xx,zz=gx^x,gz^z
    power=((gx&gz).bit_count()+(x&z).bit_count()-(xx&zz).bit_count()+2*(gz&x).bit_count())%4
    return xx,zz,-1. if power==1 else 1.

def components(n,edges):
    par=list(range(n))
    def find(a):
        while par[a]!=a:par[a]=par[par[a]];a=par[a]
        return a
    for u,v,_ in edges:par[find(u)]=find(v)
    groups={}
    for v in range(n):
        r=find(v);groups[r]=groups.get(r,0)|(1<<(n-1-v))
    return tuple(groups.values())

def alive(z,comp):return all((z&c).bit_count()%2==0 for c in comp)

def setup(work, need_components=True):
    n=work['qubits'];p=work['depth'];a=work['adjacency'];theta=work['profiles']['optimized']
    edges=[(i,j,a[i][j]) for i in range(n) for j in range(i+1,n) if a[i][j]]
    norm=sum(abs(w) for _,_,w in edges)
    initial={(0,(1<<(n-1-i))|(1<<(n-1-j))):w/norm for i,j,w in edges}
    gates=[]
    for layer in reversed(range(p)):
        gates.extend((1<<(n-1-i),0,theta[p+layer],None,layer) for i in range(n))
        gates.extend((0,(1<<(n-1-i))|(1<<(n-1-j)),w*theta[layer],(i,j,w),layer) for i,j,w in edges)
    comps=[components(n,[g[3] for g in gates[k+1:] if g[3] is not None]) for k in range(len(gates))] if need_components else None
    return n,p,initial,gates,comps

def propagate_one(terms,gate,comp,cap):
    gx,gz,angle,_,_=gate;c=math.cos(2*angle);s=math.sin(2*angle);out={}
    for (x,z),value in terms.items():
        change=rotated(gx,gz,x,z)
        if change is None:out[x,z]=out.get((x,z),0.)+value
        else:
            xx,zz,sign=change
            out[x,z]=out.get((x,z),0.)+value*c
            out[xx,zz]=out.get((xx,zz),0.)+value*sign*s
    peak=len(out)
    if peak>cap:raise RuntimeError(f'Explicit cap {cap} exceeded by {peak} pre-filter terms')
    return {key:value for key,value in out.items() if alive(key[1],comp)},peak

def scalar(terms,psi,n):
    index=np.arange(1<<n)
    total=0j
    for (x,z),value in terms.items():
        phase=(1j)**((x&z).bit_count())*np.array([(-1)**((int(i)&z).bit_count()) for i in index])
        total+=value*np.vdot(psi,(phase*psi)[index^x])
    if abs(total.imag)>1e-10:raise AssertionError(total)
    return float(total.real)

def tail_states(gates,n):
    index=np.arange(1<<n);psi=np.full(1<<n,2**(-n/2),complex);tails={len(gates):psi.copy()}
    for k in reversed(range(len(gates))):
        x,z,angle,_,_=gates[k]
        if x:psi=math.cos(angle)*psi-1j*math.sin(angle)*psi[index^x]
        else:
            signs=np.array([(-1)**((int(i)&z).bit_count()) for i in index])
            psi=np.exp(-1j*angle*signs)*psi
        tails[k]=psi.copy()
    return tails

def dense_exact(work):
    n,p,initial,gates,_=setup(work,False);index=np.arange(1<<n)
    psi=np.full(1<<n,2**(-n/2),complex)
    for x,z,angle,_,_ in reversed(gates):
        if x:psi=math.cos(angle)*psi-1j*math.sin(angle)*psi[index^x]
        else:
            signs=np.array([(-1)**((int(i)&z).bit_count()) for i in index])
            psi=np.exp(-1j*angle*signs)*psi
    return scalar(initial,psi,n)

def run(work,cut,cap,window=None,oracle=False):
    setup_start=time.perf_counter();n,p,initial,gates,comps=setup(work);setup_secs=time.perf_counter()-setup_start
    tail_seconds=0.;tail_memory=0
    if oracle and window != 'all':
        st=time.perf_counter();tails=tail_states(gates,n);tail_seconds=time.perf_counter()-st
        tail_memory=sum(v.nbytes for v in tails.values())
    K=initial.copy();R={};peak_k=len(K);peak_r=0;peak_combined=len(K);visits_k=visits_r=0
    bound_l1=0.;bound_hybrid=0.;bound_scalar=0.;signed_scalar=0.;oracle_contract_seconds=0.;records=[]
    raw_discard_l1=0.;layer_length=len(gates)//p if p else 0
    start=time.perf_counter()
    for k,(gate,comp) in enumerate(zip(gates,comps)):
        visits_k+=len(K);full,kpeak=propagate_one(K,gate,comp,cap);peak_k=max(peak_k,kpeak)
        K={};D={}
        for key,value in full.items():
            if (key[0]|key[1]).bit_count()<=cut:K[key]=value
            else:D[key]=value
        raw_discard_l1+=sum(abs(value) for value in D.values())
        peak_combined=max(peak_combined,len(K))
        if window is not None:
            visits_r+=len(R);R,rpeak=propagate_one(R,gate,comp,cap)
            for key,value in D.items():R[key]=R.get(key,0.)+value
            peak_r=max(peak_r,rpeak,len(R));peak_combined=max(peak_combined,len(K)+len(R))
            if len(R)>cap:raise RuntimeError('Explicit residual post-add cap exceeded')
            boundary=(k+1)%layer_length==0
            completed=(k+1)//layer_length
            flush=boundary and (window=='all' and completed==p or window!='all' and (completed%window==0 or completed==p))
            if flush:
                l1=sum(abs(c) for c in R.values());bound_l1+=l1
                record={'through_gate':k,'processed_layers':completed,'remaining_layers':p-completed,'residual_terms':len(R),'l1':l1}
                if k==len(gates)-1:
                    terminal_scalar=sum(c for (x,z),c in R.items() if z==0)
                    record['terminal_scalar']=terminal_scalar;bound_hybrid+=abs(terminal_scalar)
                else:bound_hybrid+=l1
                if oracle:
                    st=time.perf_counter()
                    q=sum(c for (x,z),c in R.items() if z==0) if k==len(gates)-1 else scalar(R,tails[k+1],n)
                    elapsed=time.perf_counter()-st;oracle_contract_seconds+=elapsed
                    bound_scalar+=abs(q);signed_scalar+=q;record['signed_oracle_scalar']=q
                records.append(record);R={}
    kernel_total=time.perf_counter()-start
    answer={'energy':sum(value for (x,z),value in K.items() if z==0),'window':window,'cut':cut,'peak_retained_terms':peak_k,'peak_residual_terms':peak_r,'peak_combined_stored_terms':peak_combined,'retained_input_visits':visits_k,'residual_input_visits':visits_r,'total_input_visits':visits_k+visits_r,'raw_discard_l1':raw_discard_l1,'window_l1_bound':bound_l1 if window is not None else None,'window_l1_terminal_exact_bound':bound_hybrid if window is not None else None,'window_oracle_abs_bound':bound_scalar if oracle else None,'signed_oracle_error':signed_scalar if oracle else None,'setup_seconds':setup_secs,'propagation_seconds':kernel_total-oracle_contract_seconds,'oracle_tail_seconds':tail_seconds,'oracle_contract_seconds':oracle_contract_seconds,'tail_state_bytes':tail_memory,'total_seconds':setup_secs+tail_seconds+kernel_total,'windows':records,'status':'completed','secondary_truncation':'none; exact residual expansion with explicit cap abort'}
    return answer

def main():
    inp=json.loads((BASE/'inputs.json').read_text());results={};cap=inp['term_cap'];cut=inp['cutoff']
    spec=importlib.util.spec_from_file_location('pinned_reference',str(BASE/'audited_certified_evolution_v1.py'));ref=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ref;spec.loader.exec_module(ref)
    for name,work in inp['workloads'].items():
        print('START',name,flush=True)
        n=work['qubits'];p=work['depth'];ang=work['profiles']['optimized']
        reference=ref.propagate_with_bounds(np.asarray(work['adjacency']),ang[:p],ang[p:],cut)
        dense_runs=[]
        for _ in range(3):
            st=time.perf_counter();exact_energy=dense_exact(work);dense_runs.append(time.perf_counter()-st)
        records=[]
        for label,limit,window in [('truncated',cut,None),('exact_pp',n,None),('window1',cut,1),('window2',cut,2),('all',cut,'all')]:
            runs=[]
            try:
                for _ in range(3):runs.append(run(work,limit,cap,window,False))
                result=runs[-1];result['label']=label;result['median_scalar_kernel_seconds']=statistics.median(r['propagation_seconds'] for r in runs)
                result['scalar_kernel_seconds_repeats']=[r['propagation_seconds'] for r in runs]
                result['median_total_nonoracle_seconds']=statistics.median(r['total_seconds'] for r in runs)
                if window is not None:
                    oracle=run(work,limit,cap,window,True)
                    for key in ('window_oracle_abs_bound','signed_oracle_error','oracle_tail_seconds','oracle_contract_seconds','tail_state_bytes','windows'):result[key]=oracle[key]
                    result['total_seconds_including_oracle_run']=oracle['total_seconds']
                    assert abs(result['signed_oracle_error']-(exact_energy-result['energy']))<1e-10
                    assert abs(exact_energy-result['energy'])<=result['window_l1_bound']+1e-10
                    assert abs(exact_energy-result['energy'])<=result['window_l1_terminal_exact_bound']+1e-10
                    assert abs(exact_energy-result['energy'])<=result['window_oracle_abs_bound']+1e-10
                assert abs(result['energy']-(exact_energy if limit==n else reference.energy))<1e-10
                records.append(result)
            except RuntimeError as e:records.append({'label':label,'status':'cap_aborted','reason':str(e)})
            print(name,label,records[-1].get('peak_residual_terms'),records[-1].get('window_l1_bound'),records[-1].get('window_oracle_abs_bound'),flush=True)
        results[name]={'n':n,'p':p,'reference_original_surrogate_energy':reference.energy,'exact_dense_energy':exact_energy,'actual_error':abs(exact_energy-reference.energy),'reference_suffix_bound':reference.energy_bound_suffix,'dense_seconds_repeats':dense_runs,'dense_median_seconds':statistics.median(dense_runs),'runs':records}
        output={'source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'inputs_sha256':hashlib.sha256((BASE/'inputs.json').read_bytes()).hexdigest(),'numpy':np.__version__,'term_cap':cap,'scope':'pointwise optimized angles, exact residual; no threshold pruning or interval extension; rounding excluded','workloads':results}
        (BASE/'results.json').write_text(json.dumps(output,indent=2)+'\n')
    print('DONE',flush=True)
if __name__=='__main__':main()
