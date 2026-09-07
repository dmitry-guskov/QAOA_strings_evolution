# Final commuting-ZZ suffix certificate

Facts: This is an independently derived specialization of a known weighted-IQP expectation formula, combined with the known Finner generalized Hölder inequality. It is not a claim that either ingredient or the elementary telescoping error bound is new. The potentially useful research contribution would require an actual error/cost improvement on the retained and discarded terms arising in QAOA.

## Conventions and scope

Let P(x,z)=i^r X^x Z^z, where x,z are binary n-vectors and r=|x intersect z| is an integer, not merely its parity. Thus a site with x=z=1 is the ordinary Hermitian Y. Let U(theta)=exp(-i sum_e theta_e Z_u Z_v), on a loop-free graph with real edge angles. Parallel edges must be combined first. Let q(x,z;theta)=<+|U† P(x,z) U|+>. The formulas apply only when this diagonal commuting suffix is immediately adjacent to the product input in the remaining Heisenberg propagation. Any intervening noncommuting mixer invalidates direct use of the suffix formula.

For conventional MaxCut C=sum_e w_e(1-Z_u Z_v)/2 and U_C=exp(-i gamma C), discard the identity phase and set theta_e=-gamma w_e/2. Different source conventions can reverse signs or absorb the factor two. Bounds in absolute value survive sign reversal, but exact amplitudes and gradients must use a consistent convention.

The parameterization may be affine, theta_e(t)=b_e+sum_j a_ej t_j. A box B is the Cartesian product [l_j,h_j] in the independent tied parameters t. We do not pretend each occurrence of a tied parameter is independent when forming local affine sums.

## Exact cut formula, including phases

Let A={v:x_v=1}, B=V minus A. Only cut edges H=delta(A) survive: conjugating a computational bit string by X^x flips exactly their Ising terms. With independent uniform Rademacher spins s_v,

q=i^r E_s [s^z exp(-2i sum_{e in H} theta_e s_u s_v)].

For either side S of H, write T=V minus S and h_v(s_T;t)=sum_{u neighbor v in H} theta_uv(t) s_u. Summing spins in S gives

q=i^r (-i)^{|z_S|} E_{s_T} [s_T^{z_T} product_{v in S} g_v(2h_v)],

where g_v(y)=cos(y) for z_v=0 and sin(y) for z_v=1. Choosing S=A cancels the prefactor exactly because |z_A|=r. Consequently

q=E_{s_B} [s_B^{z_B} product_{v in A} g_v(2h_v)]

is a manifestly real expression for every Hermitian Pauli. Choosing S=B is also valid, with the explicit constant unit-modulus phase above. All subsequent absolute bounds work for either side. An empty product is one. Isolated vertices with z_v=1 force zero.

Derivation: P|s>=i^r s^z |s xor x>, so the diagonal state contributes exp(i Phi(s xor x)-i Phi(s)) with Phi=sum theta_e s_u s_v. This is the displayed cut exponential. Averaging each chosen spin gives cos(2h) or -i sin(2h). No commutation phase has been suppressed.

## Component parity and graph-state relation

The high-temperature expansion of the cut exponential contains an edge-subset F only if its binary vertex boundary equals z. Equivalently z is in the GF(2) column span of the incidence matrix of H. This holds exactly when every connected component of H, including isolated vertices, contains an even number of z vertices. Failure proves q identically zero throughout parameter space, and proves every parameter derivative zero. This is a necessary structural condition, not an assurance that the scalar is nonzero at a particular angle. Under independent generic edge angles it is sufficient for a nonzero polynomial. Uniform positive equal weights also have a nonzero minimal-degree term when reachable. Signed tied weights can produce additional cancellations, so do not claim generic sufficiency for arbitrary tied weights.

At theta_e=pi/4 on every edge, the cut exponential reduces to a single edge-subset term, F=H. Then |q|=1 if z=boundary(H), and q=0 otherwise. Since boundary(delta(A))=L_G x over GF(2), this is the graph-state stabilizer condition modified by the local Z rotations relating exp(-i pi ZZ/4)|+> to the usual CZ graph state. It is stronger than component parity at this special Clifford angle. A naive adjacency-only condition would omit the degree-diagonal contribution: L_G=diag(degree)+adjacency over GF(2).

If H is a forest and component parity holds, there is one edge subset F with boundary z. The exact amplitude is i^r (-i)^|F| product_{e in F} sin(2theta_e) product_{e not in F} cos(2theta_e). For general H, boundary solutions form an affine cycle space of dimension |E(H)|-|V(H)|+components(H); exact coset enumeration has 2^dimension terms. Equivalently the cut-spin expression has 2^min(|A|,|B|) terms before symmetries. These are useful exact small-instance fallbacks, not general polynomial-time algorithms.

## Cheap local supremum and moment bounds

Fix a side S and T=V minus S. For each chosen vertex v, take independent uniform spins only on its neighborhood. Set

u_v=(E_neighbors [sup_{t in B}|g_v(2h_v(t))|^k])^(1/k),

v_vj=(E_neighbors [sup_{t in B}|2 (partial_j h_v) g'_v(2h_v(t))|^k])^(1/k),

where k=max(1,max_{u in T} degree_H(u)). The same k must be used for every factor on this side. Each T spin occurs in at most k neighborhood functions. Finner's product-measure generalized Hölder inequality therefore gives

sup_B |q| <= product_{v in S} u_v,

sup_B |partial_j q| <= sum_{v in S} v_vj product_{w in S,w!=v} u_w.

The derivative follows by differentiating the finite exact product, applying the triangle inequality to its product-rule terms, and applying the same Finner inequality with one differentiated factor. It does not require derivatives to be statistically independent or different gate angles to be independent. Finner is applied to independent computational basis spins, not to the circuit parameters. Padding with unit factors handles vertices of read degree below k.

Take the minimum of bounds from the two sides and the universal caps 1 for amplitude and 2 sum_{e in H}|a_ej| for derivative. A zero from component parity overrides all these bounds. The derivative cap follows directly by differentiating the cut exponential; it is also a spectral-width commutator bound.

A cheaper bound replaces each local L_k moment by its maximum over neighborhood spins. It is always at least as large as the displayed moment bound when implemented with the same pointwise interval majorants. Do not multiply local expectations E|g_v| as though overlapping neighborhoods were independent. That generally fails. The valid exponent is controlled by the maximum number of factors reading one underlying spin. A disjoint-neighborhood subset is a valid special case with k=1 after dropping other factors bounded by one.

For affine theta and fixed neighborhood signs, first combine the tied coefficients: h_v=b_v+sum_j a_vj t_j, with b_v=sum_neighbor b_e s_u and a_vj=sum_neighbor a_ej s_u. Its interval is exactly [b_v+sum_j min(a_vj l_j,a_vj h_j), b_v+sum_j max(a_vj l_j,a_vj h_j)]. Bound |sin| or |cos| on twice this interval by checking endpoints and extrema. The derivative factor is the constant 2|a_vj| times the opposite trigonometric supremum. There is no interval dependence loss within this local affine expression. Separate supremums across vertices and across product-rule terms remain conservative.

For unweighted theta_e=gamma, the local field is gamma h with h in {-d,-d+2,...,d}. The moment requires d+1 terms with binomial probabilities 2^-d binom(d,r), instead of enumerating 2^d sign strings. For signed or unequal weights, 2^d enumeration is straightforward and cheap at bounded degree. The method never expands omitted Pauli branches. Its cost depends on the discarded Pauli's active cut graph, the local degree, and the number of independent parameters.

## Insertion into a truncated-propagation certificate

For a fixed, parameter-independent sequence of Pauli masks, let D_t be the dropped operator batch at step t after merging equal Pauli strings. Exact telescoping expresses the final scalar error as the sum over t of expectation values of D_t propagated by the remaining exact circuit. If the remaining suffix at a discard consists only of the diagonal ZZ gates above, write D_t=sum_P d_tP P. Given certified coefficient bounds C_tP>=sup_B|d_tP| and J_tPj>=sup_B|partial_j d_tP|, the suffix contribution satisfies

sup_B |error_t| <= sum_P C_tP Q_tP,

sup_B |partial_j error_t| <= sum_P (J_tPj Q_tP + C_tP R_tPj),

where Q and R are the amplitude and derivative bounds above. Take the minimum with the corresponding generic operator-norm bounds. Merging coefficients before obtaining C and J preserves exact coefficient cancellation; summing positive certificate terms afterwards does not exploit cancellation between different discarded Paulis or different times.

For general remaining circuits the rigorous baseline is sum_t ||D_t||_infinity, upper-bounded by cumulative dropped coefficient l1. The gradient baseline is sum_t [||partial_j D_t||_infinity + 2 ||D_t||_infinity sum_{remaining gates g}|a_gj|] for gates exp(-i a_gj t_j P_g). Pairwise anticommuting groups bound a batch norm by a sum of Euclidean coefficient-group norms. Bare coefficient l2 is not a pure-state observable error bound; normalized Frobenius conversion costs an exponential support factor. These baselines already occur in Pauli-propagation literature and software.

If truncation depends on the angle, the map may change nonsmoothly across mask boundaries. A fixed-mask gradient proof cannot simply omit this dependence. The present implementation target is a fixed Pauli-weight mask. Float interval arithmetic without directed rounding is an implementation approximation, even when the analytic inequality is rigorous. Reproduction below is numerical validation of the formulas, not a proof of outward-rounded floating-point enclosures.

## Novelty boundaries and source provenance

Leontica and Amaro, Exploring the neighborhood of 1-layer QAOA with Instantaneous Quantum Polynomial circuits, arXiv:2210.05526v3, Section VII, especially Eqs (12)-(17), derive arbitrary weighted IQP Pauli expectations by a sum over one cut side. Equation (17) is a cosine/sine product sum with 2^(|x|-1) terms. This directly precedes the exact suffix formula used here. Source: https://arxiv.org/html/2210.05526v3 .

Finner, A Generalization of Hölder's Inequality and Some Probability Inequalities, Annals of Probability 20(4), 1893-1901 (1992), supplies the product-measure inequality. Pelekis, Ramon and Wang, Hölder-type inequalities and their applications to concentration and correlation bounds, arXiv:1511.07204, explicitly states it as Theorem 2.1, with the fractional-matching condition defined on page 2. The sufficient constraint is that the sum of reciprocal exponents of factors incident to a base random variable is at most one. Theorem source: https://arxiv.org/pdf/1511.07204 . Gavinsky, Lovett, Saks and Srinivasan, A Tail Bound for Read-k Families of Functions, arXiv:1205.1478, Section 1.1 Theorem 1.2, records the indicator special case and cites Finner for the continuous analogue: https://arxiv.org/pdf/1205.1478 .

Lerch et al., Efficient quantum-enhanced classical simulation for patches of quantum landscapes, arXiv:2411.19896v1, Appendix D.3 Supplemental Theorem 5 Eqs (290)-(293), already proves a worst-case sine-order Pauli truncation guarantee for adversarial/correlated small angles, with patch radius r<=kappa/m and bound (e m r/kappa)^kappa in its convention. Appendix D.4 Proposition 5 treats all gates tied to one random angle. It would be incorrect to claim that prior guarantees universally require independent angles. Source: https://arxiv.org/html/2411.19896v1 .

Lin, Granet, Hemery and Dreyer, Backpropagating Pauli Propagation, arXiv:2607.15184v1, Appendix B Eqs (33)-(48), derives rigorous cumulative discarded l2/Frobenius and l1 bounds, then deliberately uses a root-sum-square l2 quantity as an empirical proxy because rigorous bounds are often loose. Section II.2.2 discusses the failure of small coefficients to guarantee small gradient contributions. Source: https://arxiv.org/html/2607.15184v1 .

Onah and Michielsen, Fundamental Limitations of QAOA on Constrained Problems and a Route to Exponential Enhancement, arXiv:2511.17259, Eq. (33) on PDF page 28 of the version retrieved 2026-09-06, explicitly invokes the same maximum-read-degree Hölder/Finner bound in a QAOA setting. Its proposed application is joint feasibility probability via overlapping row light cones, not a Pauli suffix truncation-error certificate. I have verified the inequality statement, not independently audited their quantum-to-product-probability reduction. Source: https://arxiv.org/pdf/2511.17259 .

Assessment: The direct exact formula, graph-incidence parity, graph-state endpoint, elementary l1 certificate, and Finner inequality are established mathematical ingredients. A focused search did not establish a direct prior implementation of this particular final-suffix Finner interval-gradient certificate inside truncated tied-angle QAOA, but that negative search is not evidence sufficient to claim novelty. The discriminating question is whether the composed bound substantially improves actual discarded-term certificates at a lower total cost than exact suffix evaluation, exact/lightcone propagation, or an ordinary PP implementation with the same accuracy target.

## Reproduction and limitations

Run `python3 research/validate_suffix.py`. The script uses only Python's standard library, seed 20260906, six-qubit cycle, K3,3 and triangular prism graphs. It compares direct dense diagonal states and Hermitian-Pauli application against both cut-side exact formulas; checks point amplitude and shared-gradient bounds; checks unequal signed weights, affine edge offsets, two shared parameters, several interval widths and sampled box points; and exhaustively checks the Clifford Laplacian condition. It writes suffix_validation_results.json next to itself. The fixed-angle samples are arbitrary Pauli samples, not the actual distribution of dropped QAOA terms. A posteriori tightness on this sample cannot establish useful end-to-end truncation performance. Box sample tests validate code consistency but cannot replace the analytic sup proof or a directed-rounding implementation.

Validation result (2026-09-06): 17,880 direct-state point/box cases, including 2,520 signed weighted affine two-parameter box cases, and 12,288 exhaustive Clifford cases passed. Maximum amplitude-bound residual was 7.78e-16; gradient-bound residual 1.34e-15; exact-side gradient disagreement 2.67e-15. Reachable-sample mean exact/Finner/localmax amplitude was .1883/.2342/.3058 on C6, .1786/.2348/.3339 on K3,3, and .1435/.2178/.3178 on the triangular prism. These numbers supersede earlier exploratory samples with a different unrecorded random draw.

## Optional combination with anticommuting groups

Derivation: If dropped Paulis P_i in a group pairwise anticommute, then their common-suffix expectations satisfy sum_i q_i^2<=1. Indeed A_alpha=sum_i alpha_i P_i obeys A_alpha^2=||alpha||_2^2 I, so the support function of the expectation vector is at most ||alpha||_2. Given individual terminal caps Q_i and coefficient caps C_i, a rigorous group contribution is the support function max sum_i C_i t_i subject to 0<=t_i<=Q_i and ||t||_2<=1. This is at most min(sum_i C_i Q_i,||C||_2). If ||Q||_2<=1 the optimum is t=Q; otherwise solve the monotone one-dimensional equation sum_i min(Q_i,C_i/lambda)^2=1, handling zero C_i separately. This is standard capped-ball water filling, not a new uncertainty relation.

For the all-commuting ZZ suffix, the shared derivative vector has a second useful Euclidean constraint: ||partial_j q||_2<=2 sqrt(Var_+(G_j))=2 sqrt(sum_e a_ej^2), where G_j=sum_e a_ej Z_uZ_v, parallel edges have been combined, and every edge is nonidentity. Proof: for every unit vector alpha, A_alpha has norm one; differentiating its expectation and applying Cauchy-Schwarz to the state tangent yields |alpha dot partial_j q|<=2 sqrt(Var(G_j)) sqrt(Var(A_alpha))<=2 sqrt(Var(G_j)). The diagonal suffix commutes with G_j, so its variance equals the |+> variance; distinct nonidentity ZZ Paulis are orthogonal there. Consequently the same support-function routine can combine individual R_i bounds with this derivative radius. The total group derivative error is bounded by the amplitude support function with coefficient-derivative caps J_i, plus the derivative support function with coefficient caps C_i. This optional combination has an analytic proof but was not implemented in the accompanying numerical validation. It only applies to a group with a common remaining suffix; never combine expectations from different discard times under a single anticommuting constraint without further proof.
