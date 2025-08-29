import numpy as np
import itertools
import qiskit.quantum_info as qi
import time
import matplotlib.pyplot as plt
import networkx as nx
from itertools import product
from functools import reduce
from scipy.linalg import expm

# Produce all binary strings of length n with k 1s. If k is None, produce all possible binary strings of length n.
def get_binary_strings(n, k=None):
    """
    Produce all binary strings of length n with k 1s.
    Returns a list of binary lists.
    """
    final = []

    def kbits(r):
        result = []
        for bits in itertools.combinations(range(n), r):
            s = [0] * n
            for bit in bits:
                s[bit] = 1
            result.append(s)
        return result

    if k is not None:
        return kbits(k)

    for i in range(n + 1):
        final += kbits(i)

    return final

def get_circuit_operators(lst_x=None, lst_z=None):
    """
    Note the order! (x, z)
    """
    return qi.Pauli((lst_z, lst_x)).to_matrix()


def construct_QAOA_operator_term(H1, H2, number_of_layers=1, beta_angle=None, gamma_angle=None):
    """
    Construct the QAOA operator term, i.e. takes H1 as a term of the Hamiltonian H2 and sandwich it with mixers and exponents of H2. 
    E.g. H2 = Z1Z2 + Z2Z3 + Z1Z3 - triangle , H1 = Z1Z2 - term, QAOA = exp_H2 @ exp_X @ H1 @ exp_X.T.conjugate() @ exp_H2.T.conjugate()

    Args:
        H1: Hamiltonian for the first part.
        H2: Hamiltonian for the second part.
        number_of_layers (int, optional): Number of layers. Defaults to 1.
        beta_angle (list, optional): List of beta angles. Defaults to None.
        gamma_angle (list, optional): List of gamma angles. Defaults to None.

    Returns:
        np.ndarray: QAOA operator term.
    """
    # Define angles for mixers
    if beta_angle is None:
        beta_angle = np.random.rand(number_of_layers) * np.pi / 3

    if gamma_angle is None:
        gamma_angle = np.random.rand(number_of_layers) * np.pi / 6

    ans = H1
    nqubits = int(np.log2(len(H1)))

    mixer = 0
    x_string = np.zeros(nqubits)

    for i in range(0, nqubits):
        x_string[i] = 1
        mixer += get_circuit_operators(x_string, np.zeros(nqubits))
        x_string[i] = 0

    for p in range(number_of_layers):
        # Making unitary for Hamiltonian exp
        exp_H = expm(1j * H2 * beta_angle[p])
        # Making mixer X
        exp_X = expm(1j * mixer * gamma_angle[p])

        ans = exp_H @ exp_X @ ans @ exp_X.T.conjugate() @ exp_H.T.conjugate()

    return ans

def generate_random_QAOA_operator(nqubits, locality=2, number_of_terms=10, number_of_layers=1, beta_angle=None, gamma_angle=None):
    """
    Generate a random QAOA operator, i.e. makes a random Hamiltonian with the set number of terms and their locality.

    Args:
        nqubits (int): Number of qubits.
        locality (int): Locality constraint.
        number_of_terms (int): Number of terms.
        number_of_layers (int, optional): Number of layers. Defaults to 1.
        beta_angle (list, optional): List of beta angles. Defaults to None.
        gamma_angle (list, optional): List of gamma angles. Defaults to None.

    Returns:
        np.ndarray: Random QAOA operator.
    """
    # Define angles for mixers
    if beta_angle is None:
        beta_angle = np.random.rand(number_of_layers) * np.pi / 3
    cos_angle_beta = np.ones(number_of_layers) * np.cos(beta_angle)
    sin_angle_beta = np.ones(number_of_layers) * np.sin(beta_angle)

    if gamma_angle is None:
        gamma_angle = np.random.rand(number_of_layers) * np.pi / 6
    cos_angle_gamma = np.ones(number_of_layers) * np.cos(gamma_angle)
    sin_angle_gamma = np.ones(number_of_layers) * np.sin(gamma_angle)

    # Making Hamiltonian
    locality = min(locality, nqubits)
    number_of_terms = min(number_of_terms, nqubits)

    # Hamiltonian ZjZk
    rng = np.random.default_rng()
    Zs = get_binary_strings(nqubits, locality)
    pickedZ = rng.choice(Zs, number_of_terms, replace=False)

    # Making Hamiltonian
    H = 0
    for i in range(len(pickedZ)):
        H += get_circuit_operators(np.zeros(nqubits), pickedZ[i])

    ans = H

    for p in range(number_of_layers):
        # Making unitary for Hamiltonian exp
        identity = get_circuit_operators(np.zeros(nqubits), np.zeros(nqubits))

        unitary_z = identity

        for i in range(len(pickedZ)):
            unitary_z = unitary_z @ (cos_angle_beta[p] * identity
                                     - 1j * sin_angle_beta[p] * get_circuit_operators(np.zeros(nqubits), pickedZ[i]))

        # Making mixer X
        x_string = np.zeros(nqubits)

        unitary_x = identity

        for i in range(nqubits):
            x_string[i] = 1
            unitary_x = unitary_x @ (cos_angle_gamma[p] * identity
                                     - 1j * sin_angle_gamma[p] * get_circuit_operators(x_string, np.zeros(nqubits)))
            x_string[i] = 0

        ans = unitary_z @ unitary_x @ ans @ unitary_x.conjugate() @ unitary_z.conjugate()

    return ans

def count_solutions_XI(Operator, tol=1e-10):
    """
    Count X/I Pauli string in the operator decomposition.

    Args:
        Operator (np.array): Operator matrix.
        tol (float, optional): Tolerance. Defaults to 1e-10.

    Returns:
        tuple: X/I solutions, max locality, average locality, and the number of solutions.
    """
    nqubits = int(np.log2(len(Operator)))
    x_strings = get_binary_strings(nqubits)
    z_zeros = np.zeros(nqubits)

    ans = []
    max_locality = 0
    avg_locality = 0.0

    for i in x_strings:
        x_mat = get_circuit_operators(i, z_zeros)
        coef = np.around(1 / (1 << nqubits) * np.real(np.trace(x_mat @ Operator)), 6)
        if np.abs(coef) > tol:
            count_x = np.sum(i)
            ans.append((str(i), coef))
            if count_x > max_locality:
                max_locality = count_x
            avg_locality += count_x

    len_ans = len(ans)
    if len_ans == 0:
        len_ans = 1

    return ans, max_locality, avg_locality / len_ans, len_ans

def make_grid(nqubits, max_loc, max_terms, number_of_layers=1, beta_angle=None, gamma_angle=None):
    """
    Generate a grid of locality vs. number of terms for QAOA operators.

    Args:
        nqubits (int): Number of qubits.
        max_loc (int): Maximum locality.
        max_terms (int): Maximum number of terms.
        number_of_layers (int, optional): Number of layers. Defaults to 1.
        beta_angle (list, optional): List of beta angles. Defaults to None.
        gamma_angle (list, optional): List of gamma angles. Defaults to None.

    Returns:
        np.ndarray: Locality vs. number of terms grid.
    """
    grid = np.zeros((max_loc, max_terms, 3))
    for i in range(1, max_loc + 1):
        for j in range(1, max_terms + 1):
            if i == nqubits and j > 1:
                continue
            ans = generate_random_QAOA_operator(nqubits=nqubits, locality=i, number_of_terms=j,
                                                number_of_layers=number_of_layers, beta_angle=beta_angle,
                                                gamma_angle=gamma_angle)
            ans, max_loc, avg_loc, len_ans = count_solutions_XI(ans, 1e-10)
            grid[i - 1][j - 1][0] = max_loc
            grid[i - 1][j - 1][1] = avg_loc
            grid[i - 1][j - 1][2] = len_ans
    return grid

def plot_grid(grid, layer=-1, mode=2):
    """
    Plot the locality vs. number of terms grid.

    Args:
        grid (np.ndarray): Locality vs. number of terms grid.
        layer (int, optional): Number of layers. Defaults to -1.
        mode (int, optional): Visualization mode. Defaults to 2.
    """
    localities = [i + 1 for i in range(grid.shape[0])]
    num_of_terms = [i + 1 for i in range(grid.shape[1])]

    param_to_show_name = ''
    if mode == 2:
        param_to_show_name = 'total terms'
    elif mode == 1:
        param_to_show_name = 'average locality'
    else:
        param_to_show_name = 'max locality'

    fig, ax = plt.subplots()

    im = ax.imshow(grid[:, :, mode], interpolation='nearest', cmap='plasma')

    ax.set_xticks(np.arange(len(localities)), labels=localities)
    ax.set_yticks(np.arange(len(num_of_terms)), labels=num_of_terms)

    for locality in localities:
        for number_of_terms in num_of_terms:
            ax.text(number_of_terms - 1, locality - 1, '{:.2f}'.format(grid[locality - 1][number_of_terms - 1][mode]),
                    ha='center', va='center', color='black', fontsize=15, backgroundcolor='w')

    ax.set_xlabel('No. of Terms', fontsize=15)
    ax.set_ylabel('Locality', fontsize=15)
    if layer == -1:
        ax.set_title('{} qubits, {}'.format(len(localities), param_to_show_name), fontsize=20)
    else:
        ax.set_title('{} qubits {} layers, {}'.format(len(localities), layer, param_to_show_name), fontsize=20)
    fig.tight_layout()
    plt.show()

def make_H_maxCUT(G, full_matrix=False):
    """
    Create the Hamiltonian for the max-cut problem.

    Args:
        G (nx.Graph or np.ndarray): Input graph or adjacency matrix.
        full_matrix (bool, optional): Whether to return a full matrix. Defaults to False.

    Returns:
        np.ndarray: Hamiltonian matrix.
    """
    if type(G) == nx.classes.graph.Graph:
        n = len(G.nodes())
        adj_mat = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                temp = G.get_edge_data(i, j, default=0)
                if temp != 0:
                    adj_mat[i, j] = 1
    else:
        adj_mat = G
    n = (len(adj_mat))

    H = np.zeros((2 ** n, 2 ** n), dtype=complex)
    zeros = np.zeros(n)

    for i in range(n):
        for j in range(i + 1, n):
            if adj_mat[i][j] != 0:
                buf = np.zeros(n)
                buf[i] = buf[j] = 1
                H += adj_mat[i][j] * get_circuit_operators(zeros, buf)
    return H

PAULIS = {'I': np.eye(2),
          'X': np.array([[0, 1], [1, 0]]),
          'Y': np.array([[0, -1j], [1j, 0]]),
          'Z': np.diag([1, -1])}

def get_pauli_decomp(in_matrix, display_every=5000, PAULIS=PAULIS):
    """
    Get the Pauli decomposition of a matrix.

    Args:
        in_matrix: Input matrix.
        display_every (int): Display progress every 'display_every' iterations.
        PAULIS (dict): Dictionary of Pauli matrices.

    Returns:
        dict: Pauli decomposition.
    """
    m = int(np.log2(in_matrix.shape[0]))

    pauli_weights = {}

    k = 1
    K = len(list(product(PAULIS.keys(), repeat=m)))

    start_time = time.time()
    for u in product(PAULIS.keys(), repeat=m):
        pauli_str_name = ''.join(u)
        pauli_str_matrix = reduce(np.kron, [PAULIS[s] for s in u])
        inner_product = np.trace(in_matrix @ pauli_str_matrix) / 2 ** m
        if not np.isclose(inner_product, 0):
            pauli_weights[pauli_str_name] = inner_product
        if display_every and k % display_every == 0:
            print('\t {} qubits || {}/{} || {:.3f} s passed'.format(m, k, K, time.time() - start_time))
        k += 1

    return pauli_weights

def construct_QAOA_operator_from_H(H, number_of_layers=1, beta_angle=None, gamma_angle=None):
    """
    Construct the QAOA operator from a Hamiltonian.

    Args:
        H (np.ndarray): Hamiltonian matrix.
        number_of_layers (int, optional): Number of layers. Defaults to 1.
        beta_angle (list, optional): List of beta angles. Defaults to None.
        gamma_angle (list, optional): List of gamma angles. Defaults to None.

    Returns:
        np.ndarray: QAOA operator.
    """
    if len(H.shape) == 1:
        H = np.diag(H)
    if beta_angle is None:
        beta_angle = np.random.rand(number_of_layers) * np.pi / 3

    if gamma_angle is None:
        gamma_angle = np.random.rand(number_of_layers) * np.pi / 6

    ans = np.copy(H)
    nqubits = int(np.log2(len(H)))

    mixer = 0
    x_string = np.zeros(nqubits)

    for i in range(0, nqubits):
        x_string[i] = 1
        mixer += get_circuit_operators(x_string, np.zeros(nqubits))
        x_string[i] = 0

    for p in range(number_of_layers):
        # Making unitary for Hamiltonian exp
        exp_H = expm(1j * H * beta_angle[p])
        # Making mixer X
        exp_X = expm(1j * mixer * gamma_angle[p])

        ans = exp_H @ exp_X @ ans @ exp_X.T.conjugate() @ exp_H.T.conjugate()

    return ans

def tensor(k):
    t = k[0]
    i = 1

    while i < len(k):
        t = np.kron(t, k[i])
        i += 1
    return t

def Graph_to_Hamiltonian(G, n):
    H = np.zeros((2 ** n), dtype='float64')
    Z = np.array([1, -1], dtype='float64')

    for i in range(n):
        j = i + 1
        while j < n:
            k = [[1, 1]] * n
            k = np.array(k, dtype='float64')

            if G[i][j] != 0:
                k[i] = Z
                k[j] = Z
                H += tensor(k) * G[i][j]

            j += 1

    return H

def get_Hamiltonian(G):
    graph_array = nx.to_numpy_array(G)
    H = Graph_to_Hamiltonian(graph_array, G.number_of_nodes())
    return H
