"""Small exact Pauli reference for tied-angle QAOA (no support-only approximation).

Labels are tensor factors left to right: qubit 0 is most significant.
H = sum_{i<j} adjacency[i,j] Zi Zj; B = sum_i Xi;
U = U[p-1] ... U[0], U[l] = exp(-i beta[l] B) exp(-i gamma[l] H).
Propagation returns U† O U, merging coefficients after every rotation.
No truncation is performed. Cost can grow exponentially; this is a reference.
"""
import numpy as np


_PRODUCT = {
    ('X', 'Y'): (1j, 'Z'), ('Y', 'Z'): (1j, 'X'), ('Z', 'X'): (1j, 'Y'),
    ('Y', 'X'): (-1j, 'Z'), ('Z', 'Y'): (-1j, 'X'), ('X', 'Z'): (-1j, 'Y'),
}


def pauli_product(left, right):
    """Return phase and Hermitian Pauli label for left @ right."""
    if len(left) != len(right) or any(c not in 'IXYZ' for c in left+right):
        raise ValueError('Expected equal-length I/X/Y/Z labels.')
    phase, label = 1, []
    for a, b in zip(left, right):
        if a == 'I':
            label.append(b)
        elif b == 'I':
            label.append(a)
        elif a == b:
            label.append('I')
        else:
            factor, c = _PRODUCT[a, b]
            phase *= factor
            label.append(c)
    return phase, ''.join(label)


def conjugate_rotation(terms, generator, angle):
    """exp(+i angle P) O exp(-i angle P), with real angle and Pauli P."""
    if not np.isfinite(angle) or not np.isreal(angle):
        raise ValueError('Expected a finite real angle.')
    result = {}

    def add(label, value):
        result[label] = result.get(label, 0) + value

    for label, coefficient in terms.items():
        phase, product = pauli_product(generator, label)
        if phase.imag == 0:  # commuting Hermitian Paulis
            add(label, coefficient)
        else:
            add(label, coefficient*np.cos(2*angle))
            add(product, coefficient*1j*phase*np.sin(2*angle))
    return {label: value for label, value in result.items() if value != 0}


def propagate_qaoa(observable, adjacency, gammas, betas):
    """Return a coefficient dictionary for U† observable U in float64."""
    a = np.asarray(adjacency)
    if (a.ndim != 2 or a.shape[0] != a.shape[1] or len(a) == 0
            or not np.isrealobj(a) or not np.isfinite(a).all()
            or not np.array_equal(a, a.T) or np.any(np.diag(a))):
        raise ValueError('Expected a finite real symmetric zero-diagonal adjacency matrix.')
    g, b = np.asarray(gammas, dtype=float), np.asarray(betas, dtype=float)
    if g.ndim != 1 or b.shape != g.shape or not np.isfinite(g).all() or not np.isfinite(b).all():
        raise ValueError('Expected equal-length finite angle vectors.')
    n = len(a)
    terms = dict(observable)
    for label, value in terms.items():
        if (not isinstance(label, str) or len(label) != n
                or any(c not in 'IXYZ' for c in label) or not np.isfinite(value)):
            raise ValueError('Invalid Pauli label or coefficient.')
    for layer in reversed(range(len(g))):
        for i in range(n):
            label = 'I'*i + 'X' + 'I'*(n-i-1)
            terms = conjugate_rotation(terms, label, b[layer])
        for i in range(n):
            for j in range(i+1, n):
                if a[i, j]:
                    label = ['I']*n
                    label[i] = label[j] = 'Z'
                    terms = conjugate_rotation(terms, ''.join(label), g[layer]*a[i, j])
    return terms


def plus_expectation(terms):
    """Terminal |+> expectation: only all-I/X labels survive."""
    return sum(value for label, value in terms.items() if set(label) <= {'I', 'X'})
