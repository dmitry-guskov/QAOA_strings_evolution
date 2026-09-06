import importlib.util
from pathlib import Path
import numpy as np
from scipy.linalg import expm

spec = importlib.util.spec_from_file_location('locality', Path(__file__).resolve().parents[1]/'QAOA_locality.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_empty_small_and_complex_coefficients():
    assert m.count_solutions_XI(np.zeros((4,4))) == ([], 0, 0., 0)
    x = m.get_circuit_operators([1,0],[0,0])
    ans, maximum, average, count = m.count_solutions_XI(2e-8*x, tol=1e-10)
    assert count == maximum == average == 1
    assert abs(ans[0][1]-2e-8) < 1e-20
    assert m.count_solutions_XI(1j*x)[3] == 1

def test_dense_heisenberg_layer_order():
    z0 = m.get_circuit_operators([0,0],[1,0])
    zz = m.get_circuit_operators([0,0],[1,1])
    h = .7*zz + .2*z0
    b = sum(m.get_circuit_operators([int(i==q) for i in range(2)],[0,0]) for q in range(2))
    # Legacy locality API: beta is COST, gamma is MIXER.
    beta, gamma = [.17,-.31], [.42,.09]
    u = np.eye(4, dtype=complex)
    for cost,mixer in zip(beta,gamma):
        u = expm(-1j*mixer*b)@expm(-1j*cost*h)@u
    actual=m.construct_QAOA_operator_term(z0,h,2,beta,gamma)
    np.testing.assert_allclose(actual,u.conj().T@z0@u,atol=1e-13)

def test_random_helper_all_terms_matches_dense():
    h = sum(m.get_circuit_operators([0]*3,z) for z in m.get_binary_strings(3,2))
    beta,gamma=[.21,-.4],[.16,.37]
    actual=m.generate_random_QAOA_operator(3,2,100,2,beta,gamma)
    expected=m.construct_QAOA_operator_term(h,h,2,beta,gamma)
    np.testing.assert_allclose(actual,expected,atol=1e-13)


def test_from_h_and_seeded_zero_term_case():
    h=np.diag([.1,-.2,.7,-.3])
    expected=m.construct_QAOA_operator_term(h,h,2,[.2,.5],[.3,.1])
    np.testing.assert_allclose(m.construct_QAOA_operator_from_H(np.diag(h),2,[.2,.5],[.3,.1]),expected,atol=1e-13)
    a=m.generate_random_QAOA_operator(3,2,2,seed=12)
    np.testing.assert_array_equal(a,m.generate_random_QAOA_operator(3,2,2,seed=12))
    np.testing.assert_array_equal(m.generate_random_QAOA_operator(2,2,0,seed=12),np.zeros((4,4)))
