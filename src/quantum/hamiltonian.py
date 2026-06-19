# src/quantum/hamiltonian.py
#
# PURPOSE: Convert a NetworkX graph (your G_i from build_graph.py) into
# a Hamiltonian matrix H, and provide its eigendecomposition. This is
# the bridge between "graph data structure" and "quantum mechanical
# object" — everything downstream (walk.py, fingerprint.py) operates
# on the matrices this file produces.
#
# FULL RIGOR NOTES:
#   - We use H = -A (negative adjacency), the standard choice in CTQW
#     literature (Farhi & Guttmann 1998; Kempe 2003), but we ALSO support
#     the normalized Laplacian variant H = L = D - A, since some papers
#     argue this is better-behaved for graphs with very uneven degree
#     distributions (exactly the kind of graph YOUR pipeline produces —
#     recall google.com's hub-and-spoke structure before truncation).
#   - We compute the full eigendecomposition (not just iterating expm)
#     because (a) it lets us verify Hermitian-ness analytically, and
#     (b) once you have eigenvectors/eigenvalues, computing U(t) for
#     MANY different t values becomes a cheap matrix multiply instead
#     of re-running expm from scratch every time — this matters a lot
#     once you're doing this for hundreds of graphs at multiple t's.

import numpy as np
import networkx as nx


def build_hamiltonian(G: nx.Graph, weighted: bool = True, variant: str = "adjacency") -> dict:
    """
    Builds the Hamiltonian matrix for a graph.

    Parameters:
        G        : a NetworkX graph (e.g. from build_graph.py)
        weighted : if True, uses edge "weight" attributes (TTL-based,
                   age-based, etc. from build_graph.py). If False,
                   treats the graph as a simple 0/1 adjacency structure.
        variant  : "adjacency" → H = -A      (standard CTQW Hamiltonian)
                   "laplacian" → H = D - A    (graph Laplacian, better
                                  behaved for high-degree-variance graphs)

    Returns:
    {
        "H":            np.ndarray (n x n, real symmetric),
        "node_order":   list of node labels, index i corresponds to H row/col i,
        "eigenvalues":  np.ndarray (n,) sorted ascending,
        "eigenvectors": np.ndarray (n x n), column k is eigenvector for eigenvalues[k],
        "is_hermitian": bool — sanity check, MUST be True for valid physics,
        "n_nodes":      int,
        "variant":      str,
    }
    """

    if G.number_of_nodes() == 0:
        raise ValueError("Cannot build a Hamiltonian for an empty graph (0 nodes).")

    # node_order fixes a CONSISTENT mapping from graph nodes to matrix
    # indices. This ordering is otherwise arbitrary (NetworkX doesn't
    # guarantee node iteration order is meaningful) — but it doesn't
    # matter for our purposes, because φ(G) is permutation-invariant
    # by construction. We just need ONE consistent ordering to build
    # the matrix; which specific ordering we pick is irrelevant to the
    # final fingerprint, which is exactly the property we're relying on.
    node_order = list(G.nodes())
    n = len(node_order)
    node_index = {node: i for i, node in enumerate(node_order)}

    # Build the adjacency matrix A (or weighted adjacency if requested)
    A = np.zeros((n, n), dtype=np.float64)
    for u, v, data in G.edges(data=True):
        i, j = node_index[u], node_index[v]
        w = data.get("weight", 1.0) if weighted else 1.0
        A[i, j] = w
        A[j, i] = w   # undirected graph → symmetric matrix

    if variant == "adjacency":
        H = -A
    elif variant == "laplacian":
        # D is the degree matrix: D[i,i] = sum of weights of edges at node i
        D = np.diag(A.sum(axis=1))
        H = D - A
    else:
        raise ValueError(f"Unknown variant: {variant}. Use 'adjacency' or 'laplacian'.")

    # ── Numerical stability check #1: symmetry ──────────────────────────
    # H must be exactly symmetric (Hermitian, since we're using real
    # matrices) for the physics to be valid — e^{-iHt} is only unitary
    # (probability-preserving) if H is Hermitian. Floating point error
    # from the weight assignment above can introduce TINY asymmetries
    # (e.g. H[i,j] = -0.5000000001 vs H[j,i] = -0.4999999998), so we
    # symmetrize defensively rather than trust the construction above blindly.
    H = (H + H.T) / 2.0
    is_hermitian = np.allclose(H, H.T, atol=1e-10)

    if not is_hermitian:
        raise RuntimeError(
            "Hamiltonian failed symmetry check after correction — "
            "this indicates a bug in graph construction, not normal "
            "floating-point noise. Investigate before proceeding."
        )

    # ── Eigendecomposition ───────────────────────────────────────────────
    # np.linalg.eigh (NOT np.linalg.eig) is specifically for symmetric/
    # Hermitian matrices. It's both faster AND more numerically stable
    # than the general-purpose eig(), and it GUARANTEES real eigenvalues
    # and orthonormal eigenvectors — both required for valid quantum
    # mechanics. Using the wrong eig function here is a classic mistake
    # that produces subtly wrong results without throwing an error.
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    return {
        "H":            H,
        "node_order":   node_order,
        "eigenvalues":  eigenvalues,
        "eigenvectors": eigenvectors,
        "is_hermitian": is_hermitian,
        "n_nodes":      n,
        "variant":      variant,
    }


def verify_against_known_graph() -> None:
    """
    SELF-TEST: verify our Hamiltonian construction against a graph with
    ANALYTICALLY KNOWN eigenvalues — the triangle graph (K3, complete
    graph on 3 nodes) from Stage 0's worked example.

    For K3 with H = -A:
      A = [[0,1,1],[1,0,1],[1,1,0]]
      Known eigenvalues of A are: 2, -1, -1 (this is textbook —
      the complete graph K_n has eigenvalues (n-1) once and -1 with
      multiplicity (n-1))
      So eigenvalues of H = -A are: -2, 1, 1

    If our code doesn't reproduce this exactly, something is wrong
    in the matrix construction or eigendecomposition — this is our
    ground-truth check before trusting the code on real phishing graphs.
    """
    G = nx.complete_graph(3)   # K3: triangle, unweighted

    result = build_hamiltonian(G, weighted=False, variant="adjacency")
    eigenvalues = sorted(result["eigenvalues"])

    expected = sorted([-2.0, 1.0, 1.0])

    print("\n  Verification: K3 (triangle graph) Hamiltonian eigenvalues")
    print(f"    Computed: {[round(e, 6) for e in eigenvalues]}")
    print(f"    Expected: {expected}")

    if np.allclose(eigenvalues, expected, atol=1e-6):
        print("    ✓ PASS — matches analytical solution exactly")
    else:
        print("    ✗ FAIL — does NOT match analytical solution. Investigate immediately.")

    print(f"    is_hermitian: {result['is_hermitian']}")


def _print_summary(result: dict) -> None:
    print(f"\n  Hamiltonian summary")
    print(f"  n_nodes:      {result['n_nodes']}")
    print(f"  variant:      {result['variant']}")
    print(f"  is_hermitian: {result['is_hermitian']}")
    print(f"  eigenvalues (sorted): {[round(e, 4) for e in sorted(result['eigenvalues'])]}")
    print(f"  {'─'*52}")


# ── SELF-TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 58)
    print("  EVELYN — hamiltonian()")
    print("=" * 58)

    # Test 1: analytical verification against known graph spectrum
    verify_against_known_graph()

    # Test 2: a slightly larger, asymmetric (degree-varied) toy graph,
    # similar in SHAPE to a small real phishing ego-graph: one "domain"
    # hub connected to several leaf nodes plus one extra cross-edge.
    print("\n  Test on a small star-plus-one-edge graph (mimics a phishing ego-graph):")
    G2 = nx.Graph()
    G2.add_edges_from([
        ("domain", "ip1"), ("domain", "registrar"), ("domain", "cert_peer1"),
        ("ip1", "asn1"), ("ip1", "cert_peer1"),   # the "+1 edge" creating a cycle
    ])
    result2 = build_hamiltonian(G2, weighted=False, variant="adjacency")
    _print_summary(result2)

    print("\n  Same graph, Laplacian variant:")
    result3 = build_hamiltonian(G2, weighted=False, variant="laplacian")
    _print_summary(result3)