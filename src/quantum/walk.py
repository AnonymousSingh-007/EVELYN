# src/quantum/walk.py
#
# PURPOSE: Compute U(t) = e^{-iHt}, the quantum walk evolution operator,
# for one or many time values t, using the eigendecomposition from
# hamiltonian.py. This is the mathematical core of the entire project.
#
# WHY EIGENDECOMPOSITION INSTEAD OF scipy.linalg.expm DIRECTLY:
#   scipy.linalg.expm(M) recomputes the matrix exponential FROM SCRATCH
#   every single call — expensive if you need U(t) at 10 different t
#   values per graph, across 1000 graphs (10,000 expm calls).
#
#   If we already have H = V @ diag(eigenvalues) @ V.T (the eigendecomposition),
#   then:
#       e^{-iHt} = V @ diag(e^{-i * eigenvalues * t}) @ V.T
#   which means computing U(t) for a NEW t is just: exponentiate a vector
#   of eigenvalues (cheap), then two matrix multiplies. This is the
#   textbook "diagonalize once, evaluate many times" trick, and it's
#   also MORE numerically stable than repeated calls to expm for a
#   Hermitian matrix, because we're never re-deriving anything — we're
#   reusing an exact decomposition.
#
# We cross-validate against scipy's expm as a numerical sanity check
# (see verify_against_scipy below) — if our fast eigendecomposition
# path and scipy's general-purpose path disagree, that's a bug signal.

import numpy as np
from scipy.linalg import expm

from src.quantum.hamiltonian import build_hamiltonian


def compute_U(eigenvalues: np.ndarray, eigenvectors: np.ndarray, t: float) -> np.ndarray:
    """
    Computes U(t) = e^{-iHt} from a precomputed eigendecomposition.

    H = V @ diag(eigenvalues) @ V.T   (V = eigenvectors, real orthogonal since H is symmetric)
    e^{-iHt} = V @ diag(e^{-i * eigenvalues * t}) @ V.T

    Returns an n x n COMPLEX matrix (dtype=complex128).
    """
    V = eigenvectors

    # e^{-i * lambda * t} for each eigenvalue lambda — this is a vector
    # operation (cheap), not a matrix operation. np.exp on a complex
    # argument automatically returns complex output.
    phase_factors = np.exp(-1j * eigenvalues * t)

    # V @ diag(phase_factors) @ V.T
    # We use V * phase_factors (broadcasting across columns) instead of
    # explicitly constructing the diagonal matrix — mathematically
    # identical, but avoids allocating an n x n matrix of mostly zeros.
    U = (V * phase_factors) @ V.T

    return U


def quantum_walk(G_hamiltonian_result: dict, t_values: list) -> dict:
    """
    Computes U(t) for MULTIPLE t values from one Hamiltonian decomposition.

    Parameters:
        G_hamiltonian_result : the dict returned by build_hamiltonian()
        t_values             : list of floats, e.g. [0.5, 1.0, 2.0, 5.0]
                                Multiple t captures structure at different
                                scales — small t = local neighbourhood,
                                large t = global topology (per Stage 0).

    Returns:
    {
        "U_matrices":   {t: np.ndarray (n x n complex) for each t in t_values},
        "node_order":   list (same as input, passed through for convenience),
        "is_unitary":   {t: bool} — sanity check per t value,
        "t_values":     list (same as input),
    }
    """
    eigenvalues  = G_hamiltonian_result["eigenvalues"]
    eigenvectors = G_hamiltonian_result["eigenvectors"]
    node_order   = G_hamiltonian_result["node_order"]
    n            = G_hamiltonian_result["n_nodes"]

    U_matrices = {}
    is_unitary = {}

    for t in t_values:
        U = compute_U(eigenvalues, eigenvectors, t)
        U_matrices[t] = U

        # ── Numerical stability check: unitarity ────────────────────────
        # A valid quantum evolution operator MUST satisfy U @ U^dagger = I
        # (U^dagger = conjugate transpose). If this fails by more than a
        # tiny floating-point tolerance, something upstream is wrong —
        # either H wasn't actually Hermitian, or there's a numerical
        # precision issue with very large t or very large eigenvalue
        # spread (a known failure mode: e^{-i * lambda * t} oscillates
        # faster for large lambda*t, and float64 precision can degrade).
        U_dagger = U.conj().T
        identity_check = U @ U_dagger
        is_unitary[t] = np.allclose(identity_check, np.eye(n), atol=1e-8)

    return {
        "U_matrices": U_matrices,
        "node_order": node_order,
        "is_unitary": is_unitary,
        "t_values":   t_values,
    }


def verify_against_scipy(n_nodes: int = 5, t: float = 1.0, seed: int = 42) -> bool:
    """
    SELF-TEST: cross-validates our eigendecomposition-based compute_U()
    against scipy.linalg.expm() directly on a random graph.

    These are two INDEPENDENT mathematical paths to the same answer:
      Path A (ours):   diagonalize H, exponentiate eigenvalues, recombine
      Path B (scipy):  general-purpose matrix exponential (Pade approximation)

    If they disagree beyond floating-point tolerance, there's a bug
    in our implementation — this is the single most important
    correctness check in the entire quantum module.
    """
    import networkx as nx

    rng = np.random.default_rng(seed)
    G = nx.gnm_random_graph(n_nodes, n_nodes + 2, seed=seed)

    # Ensure connected (random graphs can have isolated nodes)
    if not nx.is_connected(G):
        G = nx.path_graph(n_nodes)   # fallback to guaranteed-connected graph

    ham_result = build_hamiltonian(G, weighted=False, variant="adjacency")
    H = ham_result["H"]

    # Path A: our eigendecomposition method
    U_ours = compute_U(ham_result["eigenvalues"], ham_result["eigenvectors"], t)

    # Path B: scipy's general matrix exponential, directly on H
    U_scipy = expm(-1j * H * t)

    max_diff = np.max(np.abs(U_ours - U_scipy))
    passed = max_diff < 1e-9

    print(f"\n  Verification: eigendecomposition method vs scipy.linalg.expm")
    print(f"    Graph: {n_nodes} nodes, t={t}")
    print(f"    Max element-wise difference: {max_diff:.2e}")
    print(f"    {'✓ PASS' if passed else '✗ FAIL'} — methods {'agree' if passed else 'DISAGREE, investigate!'}")

    return passed


def _print_walk_summary(result: dict) -> None:
    print(f"\n  Quantum walk summary")
    print(f"  t values tested: {result['t_values']}")
    for t in result["t_values"]:
        unitary = result["is_unitary"][t]
        flag = "✓" if unitary else "✗ NUMERICAL INSTABILITY"
        print(f"    t={t:>6}: unitary={unitary}  {flag}")
    print(f"  {'─'*52}")


# ── SELF-TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import networkx as nx

    print("\n" + "=" * 58)
    print("  EVELYN — walk()")
    print("=" * 58)

    # Test 1: cross-validation against scipy (the critical correctness check)
    verify_against_scipy(n_nodes=5, t=1.0)
    verify_against_scipy(n_nodes=8, t=3.0)

    # Test 2: full multi-t walk on the Stage 0 triangle graph
    print("\n  Multi-t quantum walk on triangle graph (K3):")
    G = nx.complete_graph(3)
    ham = build_hamiltonian(G, weighted=False, variant="adjacency")
    walk_result = quantum_walk(ham, t_values=[0.5, 1.0, 2.0, 5.0, 10.0])
    _print_walk_summary(walk_result)

    # Test 3: numerical stability stress test — large t value
    # This is exactly the kind of test that catches the "float64
    # precision degrades for large lambda*t" failure mode mentioned
    # in the unitarity check comment above.
    print("\n  Stress test: very large t value (numerical stability check):")
    walk_stress = quantum_walk(ham, t_values=[100.0, 1000.0])
    _print_walk_summary(walk_stress)