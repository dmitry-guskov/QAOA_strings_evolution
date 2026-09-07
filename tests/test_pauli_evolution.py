import importlib.util
from pathlib import Path
from functools import reduce
import numpy as np
import pytest
from scipy.linalg import expm

spec=importlib.util.spec_from_file_location('pauli_evolution',Path(__file__).resolve().parents[1]/'pauli_evolution.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
P={'I':np.eye(2),'X':np.array([[0,1],[1,0]]),'Y':np.array([[0,-1j],[1j,0]]),'Z':np.diag([1,-1])}
def matrix(label):
    return reduce(np.kron,[P[c] for c in label])

@pytest.mark.parametrize('depth',[0,1,2,3])
def test_full_operator_against_independent_dense_evolution(depth):
    a=np.array([[0,.7,0],[.7,0,-.4],[0,-.4,0]])
    h=.7*matrix('ZZI')-.4*matrix('IZZ')
    b=matrix('XII')+matrix('IXI')+matrix('IIX')
    rng=np.random.default_rng(23)
    gammas=rng.normal(size=depth)
    betas=rng.normal(size=depth)
    u=np.eye(8,dtype=complex)
    for g,beta in zip(gammas,betas):
        u=expm(-1j*beta*b)@expm(-1j*g*h)@u
    observable={'ZYI':.3,'XIZ':-.2,'III':.1}
    terms=m.propagate_qaoa(observable,a,gammas,betas)
    actual=sum(value*matrix(label) for label,value in terms.items())
    expected=u.conj().T@sum(value*matrix(label) for label,value in observable.items())@u
    np.testing.assert_allclose(actual,expected,atol=2e-14)
    plus=np.ones(8)/np.sqrt(8)
    assert abs(m.plus_expectation(terms)-plus@expected@plus)<2e-14
    assert max(abs(complex(c).imag) for c in terms.values())<1e-14

def test_zero_mixer_has_no_spurious_branches():
    assert m.propagate_qaoa({'ZZ':1},[[0,1],[1,0]],[.37],[0]) == {'ZZ':1}

def test_rotation_inverse_and_clifford():
    terms=m.conjugate_rotation({'Z':1},'X',np.pi/4)
    assert abs(terms['Y']-1)<1e-14
    restored=m.conjugate_rotation(terms,'X',-np.pi/4)
    assert abs(restored['Z']-1)<1e-14
    assert abs(restored.get('Y',0))<1e-14
