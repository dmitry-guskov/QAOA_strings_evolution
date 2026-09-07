import pathlib
import sys
sys.dont_write_bytecode=True
exec(open(str(pathlib.Path(__file__).resolve().parent / 'group_diagnostic.py')).read().split("data=json.load")[0])
data=json.load(open(str(pathlib.Path(__file__).resolve().parent / 'pilot_inputs.json')));out=[]
for name,cut in [('cycle6:p2',3),('k33:p2',4),('prism6:p2',4),('k33:p4',4),('prism6:p4',4)]:
 work=data['workloads'][name];a=np.asarray(work['adjacency']);angles=work['profiles']['optimized'];p=work['depth'];n=len(a);N=1<<n;idx=np.arange(N)
 edges=[(i,j,a[i,j]) for i in range(n) for j in range(i+1,n) if a[i,j]]
 gates=[]
 for layer in reversed(range(p)):
  gates.extend((1<<(n-1-i),0,angles[p+layer]) for i in range(n))
  gates.extend((0,(1<<(n-1-i))|(1<<(n-1-j)),w*angles[layer]) for i,j,w in edges)
 psi=np.full(N,1/np.sqrt(N),complex);tails={len(gates):psi.copy()}
 for k in reversed(range(len(gates))):
  x,z,theta=gates[k]
  if x: psi=np.cos(theta)*psi-1j*np.sin(theta)*psi[idx^x]
  else:
   signs=np.array([(-1)**((int(i)&z).bit_count()) for i in idx]);psi=np.exp(-1j*theta*signs)*psi
  tails[k]=psi.copy()
 m._audit_batches=[];got=m.propagate_with_bounds(a,angles[:p],angles[p:],cut)
 signed=0.;abssum=0.;earlyabs=0.;vals=[]
 for k,(pure,batch) in enumerate(m._audit_batches):
  psi=tails[k+1];value=0.
  for x,z,c in batch:
   phase=(1j)**((x&z).bit_count())*np.array([(-1)**((int(i)&z).bit_count()) for i in idx])
   value+=c*np.vdot(psi,(phase*psi)[idx^x]).real
  signed+=value;abssum+=abs(value);earlyabs+=0 if pure else abs(value)
  vals.append(value)
 exact=m.propagate_with_bounds(a,angles[:p],angles[p:],n).energy
 assert abs(signed-(exact-got.energy))<1e-11
 out.append({'workload':name,'cut':cut,'actual_error':abs(exact-got.energy),'exact_scalar_batch_abs_sum':abssum,'exact_scalar_batch_early_abs_sum':earlyabs,'certificate_suffix':got.energy_bound_suffix,'signed_telescoping_residual':signed-(exact-got.energy),'batch_signed_contributions':vals})
pathlib.Path(str(pathlib.Path(__file__).resolve().parent / 'scalar_diagnostic.json')).write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps([{k:v for k,v in r.items() if k!='batch_signed_contributions'} for r in out],indent=2))
