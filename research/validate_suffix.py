#!/usr/bin/env python3
"""Standard-library validation; numerical evidence, not outward-rounded certification."""
import cmath, itertools, json, math, pathlib, random, time

SEED = 20260906

def prod(xs):
    return math.prod(xs)

def spins(n):
    return [tuple(1-2*((s>>v)&1) for v in range(n)) for s in range(1<<n)]

def theta(edge, point):
    u,v,b,a = edge
    return b+sum(c*t for c,t in zip(a,point))

def active(n, edges, x):
    es=[e for e in edges if ((x>>e[0])^(x>>e[1]))&1]
    adj=[[] for _ in range(n)]
    for e in es:
        u,v,_,_=e
        adj[u].append((v,e)); adj[v].append((u,e))
    return es,adj

def reachable(adj,z):
    seen=set()
    for root in range(len(adj)):
        if root in seen: continue
        stack=[root]; seen.add(root); parity=0
        while stack:
            u=stack.pop(); parity^=(z>>u)&1
            for v,_ in adj[u]:
                if v not in seen: seen.add(v); stack.append(v)
        if parity: return False
    return True

def dense_state(n,edges,x,z,point):
    """Prepare complete diagonal state and apply Hermitian Pauli, no cut formula."""
    states=spins(n); norm=(1<<n)**-0.5; dim=len(point)
    phase=[sum(theta(e,point)*s[e[0]]*s[e[1]] for e in edges) for s in states]
    psi=[norm*cmath.exp(-1j*p) for p in phase]
    dpsi=[[(-1j*sum(e[3][j]*s[e[0]]*s[e[1]] for e in edges))*p for s,p in zip(states,psi)] for j in range(dim)]
    py=(1j)**((x&z).bit_count())
    q=0j; dq=[0j]*dim
    for b,s in enumerate(states):
        p=py*prod(s[v] for v in range(n) if (z>>v)&1); t=b^x
        q+=psi[t].conjugate()*p*psi[b]
        for j in range(dim):
            dq[j]+=dpsi[j][t].conjugate()*p*psi[b]+psi[t].conjugate()*p*dpsi[j][b]
    return q,dq

def exact_side(n,edges,x,z,point,side):
    _,adj=active(n,edges,x)
    chosen=[v for v in range(n) if ((x>>v)&1)==side]
    other=[v for v in range(n) if v not in chosen]
    phase=(1j)**((x&z).bit_count())*(-1j)**sum((z>>v)&1 for v in chosen)
    q=0.; dq=[0.]*len(point)
    for ss in itertools.product((-1,1),repeat=len(other)):
        s=dict(zip(other,ss)); pref=prod(s[v] for v in other if (z>>v)&1)
        fs=[]; dfs=[]
        for v in chosen:
            h=sum(theta(e,point)*s[u] for u,e in adj[v]); sine=(z>>v)&1
            fs.append(math.sin(2*h) if sine else math.cos(2*h))
            dfs.append([2*sum(e[3][j]*s[u] for u,e in adj[v])*(math.cos(2*h) if sine else -math.sin(2*h)) for j in range(len(point))])
        q+=pref*prod(fs)
        for j in range(len(point)):
            dq[j]+=pref*sum(dfs[i][j]*prod(fs[:i]+fs[i+1:]) for i in range(len(fs)))
    scale=phase/(1<<len(other))
    return q*scale,[v*scale for v in dq]

def trig_sup(lo,hi,sine):
    """Analytic |sin|/|cos| max evaluated with floats; no rounding guarantee."""
    if hi<lo: lo,hi=hi,lo
    offset=math.pi/2 if sine else 0.
    if math.ceil((lo-offset)/math.pi)<=math.floor((hi-offset)/math.pi): return 1.
    f=math.sin if sine else math.cos
    return max(abs(f(lo)),abs(f(hi)))

def bounds(n,edges,x,z,box,moment=True):
    es,adj=active(n,edges,x); dim=len(box)
    if not reachable(adj,z): return 0.,[0.]*dim
    allb=[]; alld=[]
    for side in (0,1):
        chosen=[v for v in range(n) if ((x>>v)&1)==side]
        other=[v for v in range(n) if v not in chosen]
        k=max([1]+[len(adj[v]) for v in other])
        us=[]; vs=[]
        for v in chosen:
            coeff_samples=[]; der_samples=[[] for _ in range(dim)]
            for ss in itertools.product((-1,1),repeat=len(adj[v])):
                b=sum(e[2]*s for s,(_,e) in zip(ss,adj[v]))
                a=[sum(e[3][j]*s for s,(_,e) in zip(ss,adj[v])) for j in range(dim)]
                lo=b+sum(min(c*l,c*h) for c,(l,h) in zip(a,box))
                hi=b+sum(max(c*l,c*h) for c,(l,h) in zip(a,box))
                u=trig_sup(2*lo,2*hi,(z>>v)&1)
                coeff_samples.append(u)
                opposite=trig_sup(2*lo,2*hi,not ((z>>v)&1))
                for j in range(dim): der_samples[j].append(2*abs(a[j])*opposite)
            norm=lambda vals:(sum(t**k for t in vals)/len(vals))**(1/k) if moment else max(vals)
            us.append(norm(coeff_samples)); vs.append([norm(vals) for vals in der_samples])
        allb.append(prod(us))
        alld.append([sum(vs[i][j]*prod(us[:i]+us[i+1:]) for i in range(len(us))) for j in range(dim)])
    return min([1.]+allb),[min(2*sum(abs(e[3][j]) for e in es),*(ds[j] for ds in alld)) for j in range(dim)]

def main():
    rng=random.Random(SEED); start=time.time()
    graphs={"cycle6":[(i,(i+1)%6) for i in range(6)],"K33":[(i,j) for i in range(3) for j in range(3,6)],"prism6":[(0,1),(1,2),(0,2),(3,4),(4,5),(3,5),(0,3),(1,4),(2,5)]}
    stats={}; total=0; worst={"amplitude_bound":0.,"gradient_bound":0.,"exact_side_amplitude":0.,"exact_side_gradient":0.,"imaginary":0.}
    def check(n,edges,x,z,point,box,exact=True):
        nonlocal total
        q,dq=dense_state(n,edges,x,z,point); b,db=bounds(n,edges,x,z,box)
        lm,_=bounds(n,edges,x,z,box,False)
        worst["amplitude_bound"]=max(worst["amplitude_bound"],abs(q)-b)
        worst["gradient_bound"]=max(worst["gradient_bound"],*(abs(a)-c for a,c in zip(dq,db)))
        worst["imaginary"]=max(worst["imaginary"],abs(q.imag),*(abs(a.imag) for a in dq))
        assert b<=lm+1e-12
        if exact:
            for side in (0,1):
                qs,ds=exact_side(n,edges,x,z,point,side)
                worst["exact_side_amplitude"]=max(worst["exact_side_amplitude"],abs(q-qs))
                worst["exact_side_gradient"]=max(worst["exact_side_gradient"],*(abs(a-c) for a,c in zip(dq,ds)))
        total+=1
        return abs(q),b,lm
    for name,pairs in graphs.items():
        edges=[(u,v,0.,(1.,)) for u,v in pairs]
        reachable_rows=[]
        for x in range(64):
            zs=rng.sample(range(64),16)
            for z in zs:
                for gamma in (.1,.2,.35,math.pi/4,1.):
                    row=check(6,edges,x,z,(gamma,),((gamma,gamma),))
                    if reachable(active(6,edges,x)[1],z): reachable_rows.append(row)
        stats[name]={"point_cases":5120,"reachable_cases":len(reachable_rows),"mean_exact_amplitude":sum(r[0] for r in reachable_rows)/len(reachable_rows),"mean_finner_bound":sum(r[1] for r in reachable_rows)/len(reachable_rows),"mean_localmax_bound":sum(r[2] for r in reachable_rows)/len(reachable_rows)}
    # Signed unequal weights, affine offsets, two tied parameters, interval boxes.
    weighted_cases=0
    for name,pairs in graphs.items():
        edges=[(u,v,rng.uniform(-.2,.2),(rng.choice((-.7,.4,1.3)),rng.choice((-.5,.0,.8)))) for u,v in pairs]
        for _ in range(120):
            x=rng.randrange(64); z=rng.randrange(64)
            center=[rng.uniform(-1,1),rng.uniform(-1,1)]
            radius=rng.choice((0.,.001,.03,.2,1.))
            box=tuple((c-radius,c+radius) for c in center)
            pts=list(itertools.product(*box))+[tuple(rng.uniform(l,h) for l,h in box) for _ in range(3)]
            for pt in pts: check(6,edges,x,z,pt,box); weighted_cases+=1
    # Clifford graph-state condition z = Laplacian(G) x mod 2 for theta=pi/4.
    clifford_cases=0
    for pairs in graphs.values():
        edges=[(u,v,0.,(1.,)) for u,v in pairs]
        for x in range(64):
            target=0
            for u,v in pairs:
                if ((x>>u)^(x>>v))&1: target^=(1<<u)|(1<<v)
            for z in range(64):
                q,_=dense_state(6,edges,x,z,(math.pi/4,))
                assert abs(abs(q)-(1. if z==target else 0.))<1e-12
                clifford_cases+=1
    assert max(worst.values())<1e-10,worst
    output={"seed":SEED,"validation":"floating-point numerical checks; theorem has separate analytic proof","point_and_box_cases":total,"weighted_affine_box_cases":weighted_cases,"clifford_cases":clifford_cases,"max_residuals":worst,"unweighted_point_statistics":stats,"elapsed_seconds":time.time()-start}
    out=pathlib.Path(__file__).with_name("suffix_validation_results.json")
    out.write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps(output,indent=2))

if __name__=="__main__": main()
