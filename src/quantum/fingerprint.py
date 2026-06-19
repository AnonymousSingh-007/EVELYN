# src/quantum/fingerprint.py
#
# PURPOSE: Extract φ(G), the final fixed-structure feature representation,
# from the quantum walk results in walk.py. This is the object that
# gets clustered (Stage 4) and compared against the GNN baseline
# (gnn_baseline.py / compare_blind_spots.py).
#
# CRITICAL DESIGN DECISION — graphs have DIFFERENT SIZES:
#   Your real data (Stage 1) showed phishing graphs with ~15-20 nodes
#   and benign graphs from 2 to 50 nodes (after truncation). The raw
#   |U(t)|^2 matrix is n x n — DIFFERENT SIZES for different graphs,
#   which means you CANNOT directly compare or cluster raw matrices
#   across graphs of different sizes.
#
#   The fix, used in the quantum-walk-graph-comparison literature
#   (Wang & Douglas; the CTQW attributed-graph-matching paper from
#   our literature search): extract SPECTRAL / DISTRIBUTIONAL summary
#   statistics from |U(t)|^2 that are fixed-length REGARDLESS of n.
#   This file implements three such summaries, of increasing
#   sophistication, and you can choose which to use later via ablation.

import numpy as np

from src.quantum.hamiltonian import build_hamiltonian
from src.quantum.walk import quantum_walk


# Default time values — chosen to span "local" (small t) to "global"
# (large t) structure, per the Stage 0 explanation. These become a
# hyperparameter you formally justify/ablate in Stage 4.
DEFAULT_T_VALUES = [0.5, 1.0, 2.0, 5.0, 10.0]


def extract_fingerprint(G, t_values: list = None, method: str = "eigenvalue_histogram") -> dict:
    """
    Computes φ(G) — a FIXED-LENGTH feature vector — for any graph,
    regardless of its node count.

    Parameters:
        G        : a NetworkX graph
        t_values : list of time values (defaults to DEFAULT_T_VALUES)
        method   : "eigenvalue_histogram" — bins the Hamiltonian's
                       eigenvalue SPECTRUM into a fixed number of bins.
                       This is THE classic spectral graph-comparison
                       approach: the spectrum is itself permutation-
                       invariant (doesn't depend on node order), and
                       binning makes it size-invariant too.
                   "return_probability" — diagonal of |U(t)|^2 (prob.
                       of returning to the SAME node after time t),
                       summarized by mean/std/percentiles across nodes.
                       This measures how "tightly knit" vs "sprawling"
                       the graph is at each time scale.
                   "combined" — concatenates both (recommended for
                       your actual experiments — more information,
                       still fixed-length).

    Returns:
    {
        "phi":          np.ndarray, FIXED LENGTH regardless of n_nodes,
        "method":       str,
        "t_values":     list,
        "n_nodes":      int (original graph size, kept for reference/audit),
        "is_unitary":   dict (passed through from quantum_walk, for QA),
    }
    """
    if t_values is None:
        t_values = DEFAULT_T_VALUES

    ham_result = build_hamiltonian(G, weighted=True, variant="adjacency")
    walk_result = quantum_walk(ham_result, t_values)

    if method == "eigenvalue_histogram":
        phi = _eigenvalue_histogram_fingerprint(ham_result["eigenvalues"])
    elif method == "return_probability":
        phi = _return_probability_fingerprint(walk_result, ham_result["n_nodes"])
    elif method == "combined":
        phi_a = _eigenvalue_histogram_fingerprint(ham_result["eigenvalues"])
        phi_b = _return_probability_fingerprint(walk_result, ham_result["n_nodes"])
        phi = np.concatenate([phi_a, phi_b])
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "phi":        phi,
        "method":     method,
        "t_values":   t_values,
        "n_nodes":    ham_result["n_nodes"],
        "is_unitary": walk_result["is_unitary"],
    }


def _eigenvalue_histogram_fingerprint(eigenvalues: np.ndarray, n_bins: int = 20,
                                        range_min: float = -10.0, range_max: float = 10.0) -> np.ndarray:
    """
    Bins the eigenvalue spectrum into a FIXED number of histogram bins.

    Why this is permutation- AND size-invariant:
      - Eigenvalues of H don't depend on node ORDERING (a core linear
        algebra fact — relabeling nodes is a similarity transform,
        which preserves eigenvalues exactly).
      - Histogram binning converts a VARIABLE-length list of eigenvalues
        (one per node, so length = n_nodes) into a FIXED-length vector
        of n_bins counts, regardless of how many eigenvalues there were.

    range_min/range_max: for H = -A with edge weights in [0,1] (per
    build_graph.py's weight normalization), eigenvalues of -A are
    bounded by graph degree, which for our node-budget-capped graphs
    (max 50 nodes) won't exceed roughly ±15. We use ±10 as a sensible
    default and NORMALIZE counts so different total node-counts don't
    bias the histogram toward graphs with more nodes.
    """
    counts, _ = np.histogram(eigenvalues, bins=n_bins, range=(range_min, range_max))
    # Normalize to a probability distribution (sums to 1) — this is
    # what makes the fingerprint comparable across graphs with very
    # different total node counts (15 nodes vs 50 nodes).
    total = counts.sum()
    if total > 0:
        counts = counts.astype(np.float64) / total
    return counts


def _return_probability_fingerprint(walk_result: dict, n_nodes: int) -> np.ndarray:
    """
    For each t value, computes the RETURN PROBABILITY distribution —
    diag(|U(t)|^2), i.e. the probability the walk starting at node k
    is found back at node k after time t, for every node k.

    Summarizes this (variable-length, n_nodes-long) vector into FIXED
    statistics: mean, std, min, max, median. 5 numbers per t value,
    regardless of graph size.

    Interpretation: HIGH average return probability means the walk
    stays "trapped" near its starting point — a tightly-connected,
    small-diameter graph (like a tight phishing campaign's hub-and-
    spoke infrastructure). LOW return probability means the walk
    spreads out — a sprawling, loosely connected graph.
    """
    features = []
    for t in walk_result["t_values"]:
        U = walk_result["U_matrices"][t]
        return_probs = np.abs(np.diag(U)) ** 2   # length n_nodes

        features.extend([
            np.mean(return_probs),
            np.std(return_probs),
            np.min(return_probs),
            np.max(return_probs),
            np.median(return_probs),
        ])

    return np.array(features)


def verify_permutation_invariance(G, method: str = "combined", seed: int = 7) -> bool:
    """
    SELF-TEST: the single most important property claim in your entire
    paper. Relabels the SAME graph's nodes randomly, recomputes φ(G),
    and checks that the fingerprint is IDENTICAL (within floating-point
    tolerance) regardless of the relabeling.

    This is the computational proof of the claim from Stage 0:
    "the wave doesn't know or care what your domains are named."
    """
    import networkx as nx

    rng = np.random.default_rng(seed)

    # Original fingerprint
    result_original = extract_fingerprint(G, method=method)
    phi_original = result_original["phi"]

    # Relabel nodes randomly — e.g. "domain" -> "zzz_random_label_3",
    # "ip1" -> "aaa_random_label_7", etc. This simulates an attacker
    # using completely different domain names for structurally
    # identical infrastructure.
    nodes = list(G.nodes())
    shuffled_labels = nodes.copy()
    rng.shuffle(shuffled_labels)
    mapping = dict(zip(nodes, shuffled_labels))
    G_relabeled = nx.relabel_nodes(G, mapping, copy=True)

    result_relabeled = extract_fingerprint(G_relabeled, method=method)
    phi_relabeled = result_relabeled["phi"]

    max_diff = np.max(np.abs(phi_original - phi_relabeled))
    passed = max_diff < 1e-8

    print(f"\n  Verification: permutation invariance (method={method})")
    print(f"    Max difference after random relabeling: {max_diff:.2e}")
    print(f"    {'✓ PASS' if passed else '✗ FAIL'} — fingerprint is "
          f"{'invariant' if passed else 'NOT invariant — BUG, this breaks the core thesis!'}")

    return passed


def fingerprint_distance(phi_a: np.ndarray, phi_b: np.ndarray, metric: str = "cosine") -> float:
    """
    Computes distance/similarity between two fingerprints. This is the
    function dbscan_cluster.py will call repeatedly in Stage 4.

    "cosine"     : 1 - cosine_similarity. Range [0, 2]. 0 = identical
                   direction (same topology pattern), insensitive to
                   overall magnitude — good default for comparing
                   graphs of different sizes/densities.
    "euclidean"  : straight-line distance. Sensitive to magnitude —
                   use only if you've separately normalized fingerprints.
    """
    if metric == "cosine":
        norm_a = np.linalg.norm(phi_a)
        norm_b = np.linalg.norm(phi_b)
        if norm_a == 0 or norm_b == 0:
            return 1.0   # maximally dissimilar if either vector is all-zero
        cosine_sim = np.dot(phi_a, phi_b) / (norm_a * norm_b)
        return 1.0 - cosine_sim
    elif metric == "euclidean":
        return float(np.linalg.norm(phi_a - phi_b))
    else:
        raise ValueError(f"Unknown metric: {metric}")


# ── SELF-TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import networkx as nx

    print("\n" + "=" * 58)
    print("  EVELYN — fingerprint()")
    print("=" * 58)

    # Test 1: permutation invariance — THE core correctness claim
    G_triangle = nx.complete_graph(3)
    verify_permutation_invariance(G_triangle, method="eigenvalue_histogram")
    verify_permutation_invariance(G_triangle, method="return_probability")
    verify_permutation_invariance(G_triangle, method="combined")

    # Test 2: a more realistic small "phishing-like" graph
    print("\n  Permutation invariance on a star-plus-cycle graph:")
    G_realistic = nx.Graph()
    G_realistic.add_edges_from([
        ("domain", "ip1"), ("domain", "registrar"), ("domain", "cert_peer1"),
        ("ip1", "asn1"), ("ip1", "cert_peer1"),
    ])
    verify_permutation_invariance(G_realistic, method="combined")

    # Test 3: distance between two DIFFERENT graphs (should be > 0)
    print("\n  Fingerprint distance test — two structurally different graphs:")
    G_a = nx.complete_graph(4)        # densely connected
    G_b = nx.path_graph(4)            # sparse chain
    phi_a = extract_fingerprint(G_a, method="combined")["phi"]
    phi_b = extract_fingerprint(G_b, method="combined")["phi"]
    dist = fingerprint_distance(phi_a, phi_b, metric="cosine")
    print(f"    K4 (complete) vs P4 (path) cosine distance: {dist:.4f}")
    print(f"    (should be clearly > 0 — these are structurally very different)")

    # Test 4: distance between two graphs with SAME topology, different size labels
    print("\n  Fingerprint distance test — same topology, different node labels:")
    G_c = nx.complete_graph(4)
    G_d = nx.relabel_nodes(G_c, {0: "w", 1: "x", 2: "y", 3: "z"})
    phi_c = extract_fingerprint(G_c, method="combined")["phi"]
    phi_d = extract_fingerprint(G_d, method="combined")["phi"]
    dist2 = fingerprint_distance(phi_c, phi_d, metric="cosine")
    print(f"    K4 vs relabeled K4 cosine distance: {dist2:.6f}")
    print(f"    (should be ~0.0 — same topology, different labels)")