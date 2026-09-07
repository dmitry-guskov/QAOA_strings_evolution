"""Reproducible bounded pilot; compare truncation bounds with exact statevectors.

Run from any directory. Results checkpoint after each case; rerunning resumes
matching cases. Exact states use the separately validated qaoa kernel, not PP.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import networkx as nx
import numpy as np
import scipy
from scipy.optimize import minimize

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def graph_bank():
    return {'cycle6':nx.cycle_graph(6), 'k33':nx.complete_bipartite_graph(3,3),
            'prism6':nx.circular_ladder_graph(3),'cube8':nx.cubical_graph(),
            'petersen10':nx.petersen_graph(),
            'cubic12':nx.random_regular_graph(3,12,seed=20260906)}


def hamiltonian(a):
    n=len(a); index=np.arange(2**n)
    signs=[1-2*((index >> (n-1-i)) & 1) for i in range(n)]
    return sum((a[i,j]*signs[i]*signs[j] for i in range(n) for j in range(i+1,n)),np.zeros(2**n))


def exact_value_gradient(q, angles, norm):
    energy=q.expectation(angles)/norm
    _,gradient=q.qaoa_qfi_matrix(angles,return_grad=True)
    return energy,gradient/norm


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--term-limit',type=int,default=500000)
    parser.add_argument('--kernel-version', choices=['current','v1','v2'], default='current')
    parser.add_argument('--graphs',nargs='+',default=['cycle6','k33','prism6','cube8','petersen10'])
    parser.add_argument('--depths',nargs='+',type=int,default=[2,3])
    parser.add_argument('--cuts',nargs='+',type=int,default=[2,3,4])
    parser.add_argument('--radii',nargs='+',type=float,default=[0.,.01])
    parser.add_argument('--profiles',nargs='+',default=['optimized','small','random'])
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name('certificate_pilot.json'))
    args=parser.parse_args()
    kernel_root=ROOT if args.kernel_version=='current' else ROOT/'research/reproduction'/args.kernel_version
    sys.path.insert(0,str(kernel_root))
    from certified_evolution import propagate_with_bounds
    from qaoa import QAOA
    hashes={name:hashlib.sha256((kernel_root/name).read_bytes()).hexdigest() for name in ['certified_evolution.py','qaoa.py']}
    if args.output.exists():
        data=json.loads(args.output.read_text())
        if data['source_sha256']!=hashes:
            raise RuntimeError('Source changed: choose a new output path instead of mixing implementations.')
    else:
        data={'source_sha256':hashes,'environment':{'python':platform.python_version(),'numpy':np.__version__,
              'scipy':scipy.__version__,'networkx':nx.__version__},'seed':20260906,
              'convention':'H=sum ZZ; observable H/m; angles chronological [gammas,betas]',
              'scope':'Truncation bounds exclude floating-point error. Point actual errors are not box-wide tests.',
              'workloads':{},'cases':[]}
    done={row['case_id'] for row in data['cases']}
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        temporary=args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(data,indent=2)+'\n'); temporary.replace(args.output)
    for graph_name in args.graphs:
        graph=graph_bank()[graph_name]
        a=nx.to_numpy_array(graph,nodelist=list(graph))
        n=len(a); h=hamiltonian(a); norm=float(a.sum()/2)
        for p in args.depths:
            key=f'{graph_name}:p{p}'
            q=QAOA(p,h)
            if key not in data['workloads']:
                seed=20260906+101*p+sum(map(ord,graph_name))
                rng=np.random.default_rng(seed)
                start=time.perf_counter(); restarts=[]
                for _ in range(3):
                    x=rng.uniform(-.4,.4,2*p)
                    result=minimize(lambda x:exact_value_gradient(q,x,norm),x,jac=True,
                                    method='L-BFGS-B',bounds=[(-np.pi/2,np.pi/2)]*p+[(-np.pi/4,np.pi/4)]*p,
                                    options={'maxiter':400,'ftol':1e-12,'gtol':1e-8})
                    restarts.append(result)
                best=min(restarts,key=lambda r:r.fun)
                profiles={'optimized':best.x.tolist(),'small':rng.uniform(-.05,.05,2*p).tolist(),
                          'random':rng.uniform(-.6,.6,2*p).tolist()}
                data['workloads'][key]={'adjacency':a.tolist(),'profiles':profiles,
                    'optimization_seconds':time.perf_counter()-start,'optimization_nfev':sum(r.nfev for r in restarts),
                    'selected_success':bool(best.success),'selected_message':str(best.message),
                    'selected_energy':float(best.fun),'seed':seed,'qubits':n,'edges':norm,'depth':p}
                save()
            for profile in args.profiles:
                angles=np.array(data['workloads'][key]['profiles'][profile])
                # Median over nine complete exact energy+gradient calls.
                timings=[]
                for _ in range(9):
                    start=time.perf_counter(); energy,gradient=exact_value_gradient(q,angles,norm)
                    timings.append(time.perf_counter()-start)
                for cut in args.cuts:
                    if cut>n: continue
                    for radius in args.radii:
                        case_id=f'{key}:{profile}:w{cut}:r{radius:g}'
                        if case_id in done: continue
                        start=time.perf_counter()
                        try:
                            result=propagate_with_bounds(a,angles[:p],angles[p:],cut,box_radius=radius,record_stages=True,term_limit=args.term_limit)
                            row=asdict(result)
                            for name,value in list(row.items()):
                                if isinstance(value,np.ndarray): row[name]=value.tolist()
                            row.update({'case_id':case_id,'workload':key,'profile':profile,'cut':cut,'radius':radius,
                                'exact_energy':float(energy),'exact_gradient':gradient.tolist(),
                                'actual_energy_error':abs(float(energy)-result.energy),
                                'actual_gradient_error':abs(gradient-result.gradient).tolist(),
                                'exact_seconds_median':float(np.median(timings)),'status':'complete'})
                            assert row['actual_energy_error']<=result.energy_bound_suffix+1e-10
                            assert np.all(abs(gradient-result.gradient)<=result.gradient_bound_suffix+1e-9)
                        except RuntimeError as exc:
                            row={'case_id':case_id,'workload':key,'profile':profile,'cut':cut,'radius':radius,
                                 'status':'term_limit','error':str(exc),'seconds':time.perf_counter()-start}
                        data['cases'].append(row); done.add(case_id); save()
                        if row['status']=='complete':
                            print(f"{case_id}: err={row['actual_energy_error']:.3g} bound={row['energy_bound_suffix']:.3g} l1={row['energy_bound_l1']:.3g} time={row['seconds']:.2f}s",flush=True)
                        else: print(case_id+': '+row['status'],flush=True)
    return data


if __name__=='__main__':
    main()
