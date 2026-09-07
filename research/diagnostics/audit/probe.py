import sys
sys.dont_write_bytecode=True
import hashlib, importlib.util, itertools, json, math, pathlib, time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('audit_certificate',pathlib.Path(str(pathlib.Path(__file__).resolve().parent / 'audited_certified_evolution_v1.py')))
m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m; spec.loader.exec_module(m)

def exact(a,theta,observable):
    n=len(a); p=len(theta)//2; N=1<<n; idx=np.arange(N)
    spin=np.array([1-2*((idx>>(n-1-i))&1) for i in range(n)])
    h=sum((a[i,j]*spin[i]*spin[j] for i in range(n) for j in range(i+1,n)),np.zeros(N))
    psi=np.full(N,1/np.sqrt(N),complex); dpsi=np.zeros((2*p,N),complex)
    for layer in range(p):
        diag=np.exp(-1j*theta[layer]*h)
        psi*=diag; dpsi*=diag
        dpsi[layer]+=-1j*h*psi
        c=np.cos(theta[p+layer]); s=np.sin(theta[p+layer])
        for v in range(n):
            flip=idx^(1<<(n-1-v)); old=psi.copy()
            psi=c*old-1j*s*old[flip]
            dpsi=c*dpsi-1j*s*dpsi[:,flip]
            dpsi[p+layer]+=-s*old-1j*c*old[flip]
    energy=0.; gradient=np.zeros(2*p)
    for label,coeff in observable.items():
        x=sum(1<<(n-1-i) for i,c in enumerate(label) if c in 'XY')
        z=sum(1<<(n-1-i) for i,c in enumerate(label) if c in 'YZ')
        phase=(1j)**((x&z).bit_count())*np.array([(-1)**((int(i)&z).bit_count()) for i in idx])
        ppsi=(phase*psi)[idx^x]
        energy+=coeff*np.vdot(psi,ppsi).real
        gradient+=coeff*2*np.real(dpsi.conj()@ppsi)
    return energy,gradient

def run():
    rng=np.random.default_rng(789321); t0=time.time()
    violations={'full_energy':0.,'full_gradient':0.,'box_energy':0.,'box_gradient':0.,'surrogate_gradient_finite_difference':0.}
    point_cases=0; box_cases=0
    for case in range(54):
        n=2+case%4; p=case%4
        a=np.triu(rng.choice([0.,0.,-.9,.3,1.4],size=(n,n)),1); a=a+a.T
        # Deliberately force disconnected and edgeless cases.
        if case%9==0: a[:]=0
        theta=rng.uniform(-1.5,1.5,2*p)
        if case%7==0: theta[:]=math.pi/4
        obs={''.join(rng.choice(list('IXYZ'),n)):float(rng.normal()) for _ in range(7)}
        obs['I'*n]=.123
        got=m.propagate_with_bounds(a,theta[:p],theta[p:],n,observable=obs)
        energy,gradient=exact(a,theta,obs)
        violations['full_energy']=max(violations['full_energy'],abs(energy-got.energy))
        violations['full_gradient']=max(violations['full_gradient'],np.max(abs(gradient-got.gradient),initial=0))
        assert got.energy_bound_l1==got.energy_bound_suffix==0
        point_cases+=1
        cut=case%(n+1); radii=rng.uniform(0,.08,2*p)
        if case%6==0: radii[:]=0
        cert=m.propagate_with_bounds(a,theta[:p],theta[p:],cut,observable=obs,box_radius=radii)
        for v in range(5):
            trial=theta+radii*(rng.uniform(-1,1,2*p) if v>1 else (-1 if v else 1))
            approx=m.propagate_with_bounds(a,trial[:p],trial[p:],cut,observable=obs)
            e,grad=exact(a,trial,obs)
            violations['box_energy']=max(violations['box_energy'],abs(e-approx.energy)-cert.energy_bound_suffix)
            violations['box_gradient']=max(violations['box_gradient'],np.max(abs(grad-approx.gradient)-cert.gradient_bound_suffix,initial=0))
            assert cert.energy_bound_suffix<=cert.energy_bound_l1+1e-12
            assert np.all(cert.gradient_bound_suffix<=cert.gradient_bound_l1+1e-12)
            box_cases+=1
        eps=1e-6
        for j in range(2*p):
            lo=theta.copy(); hi=theta.copy(); lo[j]-=eps; hi[j]+=eps
            elo=m.propagate_with_bounds(a,lo[:p],lo[p:],cut,observable=obs).energy
            ehi=m.propagate_with_bounds(a,hi[:p],hi[p:],cut,observable=obs).energy
            violations['surrogate_gradient_finite_difference']=max(violations['surrogate_gradient_finite_difference'],abs((ehi-elo)/(2*eps)-cert.gradient[j]))
    qb,db=m.suffix_finner_bound(2,2,3,((0,1,1.),(0,1,1.)),.2)
    duplicate={'n':2,'x':2,'z':3,'edges':[[0,1,1.],[0,1,1.]],'gamma':.2,'returned_amplitude_bound':qb,'exact_amplitude':math.sin(.8),'returned_derivative_bound':db,'exact_derivative':4*math.cos(.8)}
    output={'source_sha256':hashlib.sha256((pathlib.Path(str(pathlib.Path(__file__).resolve().parent / 'audited_certified_evolution_v1.py'))).read_bytes()).hexdigest(),'seed':789321,'full_dense_cases':point_cases,'box_dense_cases':box_cases,'max_residuals':violations,'duplicate_edge_counterexample_public_helper':duplicate,'elapsed_seconds':time.time()-t0}
    pathlib.Path(str(pathlib.Path(__file__).resolve().parent / 'results.json')).write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))
    assert max(v for k,v in violations.items() if 'finite_difference' not in k)<1e-10
    assert violations['surrogate_gradient_finite_difference']<1e-7

run()
