import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from certified_evolution import propagate_with_bounds, suffix_finner_bound
from pauli_evolution import propagate_qaoa, plus_expectation
import qaoa


def fixture_graph(n):
    rng=np.random.default_rng(17+n)
    a=np.triu(rng.choice([0.,.7,-.3,1.],size=(n,n)),1)
    return a+a.T


def diagonal(a):
    n=len(a); index=np.arange(2**n)
    signs=np.array([1-2*((index >> (n-1-i)) & 1) for i in range(n)])
    return sum((a[i,j]*signs[i]*signs[j] for i in range(n) for j in range(i+1,n)),np.zeros(2**n))


@pytest.mark.parametrize('n,p',[(2,1),(3,2),(4,3),(5,2)])
def test_untruncated_energy_gradient_matches_independent_statevector(n,p):
    a=fixture_graph(n); rng=np.random.default_rng(n+p)
    g,b=rng.normal(size=(2,p))
    got=propagate_with_bounds(a,g,b,n)
    h=diagonal(a); norm=sum(abs(a[i,j]) for i in range(n) for j in range(i+1,n))
    q=qaoa.QAOA(p,h); angles=np.r_[g,b]
    expected=q.expectation(angles)/norm
    _,gradient=q.qaoa_qfi_matrix(angles,return_grad=True)
    assert abs(got.energy-expected)<3e-13
    np.testing.assert_allclose(got.gradient,gradient/norm,atol=3e-12)
    assert got.energy_bound_l1==got.energy_bound_suffix==0
    np.testing.assert_array_equal(got.gradient_bound_l1,np.zeros(2*p))


@pytest.mark.parametrize('n,p,cut',[(3,2,1),(4,2,2),(5,2,3)])
def test_uniform_box_energy_and_tied_gradient_bounds(n,p,cut):
    a=fixture_graph(n); rng=np.random.default_rng(72+n)
    center=rng.uniform(-.4,.4,size=2*p); radius=.013
    bound=propagate_with_bounds(a,center[:p],center[p:],cut,box_radius=radius)
    assert bound.energy_bound_suffix <= bound.energy_bound_l1+1e-14
    np.testing.assert_array_less(bound.gradient_bound_suffix,bound.gradient_bound_l1+1e-12)
    for trial in [center,center-radius,center+radius]+[center+rng.uniform(-radius,radius,2*p) for _ in range(5)]:
        exact=propagate_with_bounds(a,trial[:p],trial[p:],n)
        approximate=propagate_with_bounds(a,trial[:p],trial[p:],cut)
        assert abs(exact.energy-approximate.energy)<=bound.energy_bound_suffix+2e-12
        assert np.all(abs(exact.gradient-approximate.gradient)<=bound.gradient_bound_suffix+2e-11)
    eps=1e-6
    finite_difference=[]
    for axis in np.eye(2*p):
        plus,minus=center+eps*axis,center-eps*axis
        eplus=propagate_with_bounds(a,plus[:p],plus[p:],cut).energy
        eminus=propagate_with_bounds(a,minus[:p],minus[p:],cut).energy
        finite_difference.append((eplus-eminus)/(2*eps))
    np.testing.assert_allclose(bound.gradient,finite_difference,atol=2e-9)


def test_final_suffix_bound_against_all_small_paulis():
    n=3; a=fixture_graph(n); h=diagonal(a)
    edges=tuple((i,j,a[i,j]) for i in range(n) for j in range(i+1,n) if a[i,j])
    index=np.arange(2**n); gamma=.271; radius=.09
    for x in range(2**n):
        for z in range(2**n):
            qb,db=suffix_finner_bound(n,x,z,edges,gamma,radius)
            for angle in (gamma-radius,gamma,gamma+radius):
                psi=np.exp(-1j*angle*h)/np.sqrt(2**n)
                parity=np.array([(-1)**((int(i)&z).bit_count()) for i in index])
                # P=i^|x&z| X^x Z^z acts on the input index before the X permutation.
                ppsi=(1j**((x&z).bit_count())*parity*psi)[index^x]
                value=np.vdot(psi,ppsi).real
                derivative=2*np.vdot(-1j*h*psi,ppsi).real
                assert abs(value)<=qb+2e-14
                assert abs(derivative)<=db+2e-13


def test_exact_terminal_reachability_and_constant_observable():
    a=np.array([[0.,1.],[1.,0.]])
    # A single Z has odd global Z parity, impossible at a terminal I/X term.
    got=propagate_with_bounds(a,[.4,.2],[.1,.3],0,observable={'ZI':1.})
    assert got.energy==got.energy_bound_suffix==got.energy_bound_l1==0
    got=propagate_with_bounds(a,[.4,.2],[.1,.3],0,observable={'II':.7})
    assert got.energy==.7 and got.energy_bound_suffix==0
    np.testing.assert_array_equal(got.gradient,np.zeros(4))


def test_parallel_edges_are_combined_before_neighbor_expectations():
    duplicated=((0,1,1.),(1,0,1.))
    q,dq=suffix_finner_bound(2,2,3,duplicated,.2)
    assert q>=abs(np.sin(.8))-1e-14
    assert dq>=4*abs(np.cos(.8))-1e-14
    assert (q,dq)==suffix_finner_bound(2,2,3,((0,1,2.),),.2)
    assert suffix_finner_bound(2,2,3,((0,1,1.),(1,0,-1.)),.2)==(0.,0.)
    with pytest.raises(ValueError): suffix_finner_bound(2,2,3,((0,0,1.),),.2)
