import sys
sys.dont_write_bytecode=True
import json,pathlib,types,numpy as np,time
root=pathlib.Path(__file__).resolve().parents[3]
source=pathlib.Path(str(pathlib.Path(__file__).resolve().parent / 'audited_certified_evolution_v1.py')).read_text()
source=source.replace('stage_l1=np.zeros(count+1); stage_suffix=np.zeros(count+1)','stage_l1=np.zeros(count+1); stage_suffix=np.zeros(count+1)\n        _audit_batches.append((pure_zz,[]))')
source=source.replace('dropped+=1; stage_dropped+=1','dropped+=1; stage_dropped+=1\n            _audit_batches[-1][1].append((x,z,float(jet[0])))')
m=types.ModuleType('instrumented_certificate');sys.modules[m.__name__]=m;m._audit_batches=[]
exec(compile(source,str(root/'certified_evolution.py'),'exec'),m.__dict__)
data=json.load(open(str(pathlib.Path(__file__).resolve().parent / 'pilot_inputs.json')));result=[]
for name,cut in [('cycle6:p2',3),('k33:p2',4),('prism6:p2',4),('k33:p4',4),('prism6:p4',4)]:
 work=data['workloads'][name];a=np.asarray(work['adjacency']);angles=work['profiles']['optimized'];p=work['depth'];n=len(a);N=1<<n;index=np.arange(N)
 m._audit_batches=[]
 got=m.propagate_with_bounds(a,angles[:p],angles[p:],cut)
 sums={'early_l1':0.,'early_exact_batch_norm':0.,'early_greedy_anticommuting':0.,'all_l1':0.,'all_exact_batch_norm':0.,'all_greedy_anticommuting':0.}
 maxdrop=0
 for pure,batch in m._audit_batches:
  if not batch:continue
  maxdrop=max(maxdrop,len(batch));D=np.zeros((N,N),complex)
  for x,z,c in batch:
   phase=(1j)**((x&z).bit_count())*np.array([(-1)**((int(i)&z).bit_count()) for i in index])
   D[index^x,index]+=c*phase
  spectral=max(abs(np.linalg.eigvalsh(D)))
  groups=[]
  for x,z,c in sorted(batch,key=lambda t:-abs(t[2])):
   for group in groups:
    if all(((x&zz).bit_count()+(z&xx).bit_count())%2 for xx,zz,_ in group):group.append((x,z,c));break
   else:groups.append([(x,z,c)])
  grouped=sum(np.sqrt(sum(c*c for _,_,c in group)) for group in groups)
  l1=sum(abs(c) for _,_,c in batch)
  for pre in ['all']+([] if pure else ['early']):
   sums[pre+'_l1']+=l1;sums[pre+'_exact_batch_norm']+=spectral;sums[pre+'_greedy_anticommuting']+=grouped
 result.append({'workload':name,'cut':cut,'max_discarded_batch_terms':maxdrop,**sums})
print(json.dumps(result,indent=2));pathlib.Path(str(pathlib.Path(__file__).resolve().parent / 'group_diagnostic.json')).write_text(json.dumps(result,indent=2)+'\n')
