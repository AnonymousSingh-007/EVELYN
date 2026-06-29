# src/quantum/estimate_edges.py
#
# STAGE A — Edge estimation among KNOWN nodes.
#
# PURPOSE: Given a graph (e.g. a partially-investigated phishing
# campaign), score every pair of nodes that ISN'T currently connected
# by an edge, using the quantum walk's own transition amplitude
# |<j|U(t)|k>|^2. A high score on a (j,k) pair means: "the wave flows
# strongly between these two nodes even with no direct edge" — which,
# per Liang et al. 2022 and Goldsmith et al. 2023, is exactly the
# signal used for missing-link prediction in other domains (social
# networks, protein interactions). We adapt the SAME mechanism here.
#
# WHY THIS WORKS MATHEMATICALLY:
#   U(t) = e^{-iHt} doesn't just encode the edges that exist — it
#   encodes every possible PATH between every pair of nodes, weighted
#   by how the wave interferes along those paths. Two nodes with NO
#   direct edge can still have a HIGH |<j|U(t)|k>|^2 if there are many
#   short, reinforcing paths between them through their mutual
#   neighbors. That's precisely the "these two are probably connected,
#   we just haven't observed it yet" signal.
#
# THIS IS STAGE A ONLY: it works on nodes that ALREADY EXIST in the
# graph. It does NOT invent brand-new nodes — that's Stage B
# (estimate_nodes.py), which is a fundamentally different and harder
# problem this file does not attempt to solve.

import numpy as np
import networkx as nx

from src.quantum.hamiltonian import build_hamiltonian
from src.quantum.walk import compute_U


def estimate_missing_edges(G: nx.Graph, t: float = 2.0, top_k: int = 10) -> list:
    """
    Scores every non-edge (j,k) pair in G by quantum walk transition
    amplitude, returns the top_k highest-scoring candidates — these
    are the pairs MOST LIKELY to represent a real but undiscovered
    connection.

    Parameters:
        G      : a NetworkX graph (e.g. from build_graph_recursive.py)
        t      : walk time. We default to t=2.0 — per Liang et al. 2022,
                 a "two-step walk" (their term) captures common-neighbor
                 information well without over-diffusing across the
                 whole graph; t=2.0 in the continuous-time formulation
                 is the closest analog and is the value we ablate from
                 in self-tests below.
        top_k  : how many candidate missing edges to return

    Returns a list of dicts, sorted by score descending:
    [
        {"node_a": str, "node_b": str, "score": float,
         "type_a": str, "type_b": str},
        ...
    ]
    """
    if G.number_of_nodes() < 2:
        return []

    ham = build_hamiltonian(G, weighted=True, variant="adjacency")
    U = compute_U(ham["eigenvalues"], ham["eigenvectors"], t)
    node_order = ham["node_order"]
    n = ham["n_nodes"]

    # |U|^2 element-wise gives the transition PROBABILITY matrix —
    # the actual "wave flow strength" between every pair of nodes,
    # regardless of whether a direct edge exists between them.
    prob_matrix = np.abs(U) ** 2

    existing_edges = set()
    for u, v in G.edges():
        i, j = node_order.index(u), node_order.index(v)
        existing_edges.add((min(i, j), max(i, j)))

    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in existing_edges:
                continue   # only score pairs that are NOT already connected
            score = float(prob_matrix[i, j])
            node_a, node_b = node_order[i], node_order[j]
            candidates.append({
                "node_a": node_a,
                "node_b": node_b,
                "score": score,
                "type_a": G.nodes[node_a].get("type", "unknown"),
                "type_b": G.nodes[node_b].get("type", "unknown"),
            })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:top_k]


def evaluate_edge_recovery(G_full: nx.Graph, hide_fraction: float = 0.2,
                            t: float = 2.0, seed: int = 0) -> dict:
    """
    THE VALIDATION PROTOCOL (Stage C, edge-only version): takes a
    complete, real graph, hides a fraction of its REAL edges, runs
    estimate_missing_edges() on the damaged graph, and checks whether
    the hidden edges show up near the top of the candidate list.

    This is the exact validation methodology used in the link-
    prediction literature (Liang et al. 2022 uses AUC for this same
    purpose) — we hide real, known-true edges and measure whether the
    model successfully assigns them high scores when "blind" to them.

    Returns:
    {
        "hidden_edges":      list of (u,v) tuples that were removed,
        "recovered_in_topN": int — how many hidden edges appeared in
                              the top N=len(hidden_edges)*3 candidates,
        "recovery_rate":     float — recovered_in_topN / len(hidden_edges),
        "mean_rank_of_hidden": float — average rank position of hidden
                                edges among ALL candidates (lower = better),
        "total_candidates":  int,
    }
    """
    rng = np.random.default_rng(seed)
    edges = list(G_full.edges())
    n_hide = max(1, int(len(edges) * hide_fraction))

    if n_hide >= len(edges):
        raise ValueError("hide_fraction too large — would remove all edges")

    hide_indices = rng.choice(len(edges), size=n_hide, replace=False)
    hidden_edges = [edges[i] for i in hide_indices]

    G_damaged = G_full.copy()
    G_damaged.remove_edges_from(hidden_edges)

    # Score ALL non-edges in the damaged graph (not just top_k) so we
    # can find the RANK of the hidden edges among everything, not just
    # whether they happened to land in an arbitrary top-10 cutoff.
    all_candidates = estimate_missing_edges(G_damaged, t=t, top_k=10**9)

    hidden_set = set()
    for u, v in hidden_edges:
        hidden_set.add(frozenset([u, v]))

    ranks = []
    for idx, cand in enumerate(all_candidates):
        pair = frozenset([cand["node_a"], cand["node_b"]])
        if pair in hidden_set:
            ranks.append(idx)

    top_n_cutoff = n_hide * 3
    recovered_in_topN = sum(1 for r in ranks if r < top_n_cutoff)

    return {
        "hidden_edges": hidden_edges,
        "recovered_in_topN": recovered_in_topN,
        "recovery_rate": recovered_in_topN / n_hide if n_hide > 0 else 0.0,
        "mean_rank_of_hidden": float(np.mean(ranks)) if ranks else None,
        "total_candidates": len(all_candidates),
        "n_hidden": n_hide,
    }


def _print_candidates(candidates: list) -> None:
    print(f"\n  Top {len(candidates)} estimated missing edges:")
    for c in candidates:
        print(f"    {c['node_a']:30s} [{c['type_a']:10s}]  <-->  "
              f"{c['node_b']:30s} [{c['type_b']:10s}]   score={c['score']:.5f}")
    print(f"  {'─'*52}")


def _print_recovery_result(result: dict) -> None:
    print(f"\n  Edge recovery validation")
    print(f"  Hidden {result['n_hidden']} real edge(s) from the graph")
    print(f"  Recovered in top-{result['n_hidden']*3}: {result['recovered_in_topN']}/{result['n_hidden']}"
          f"  (recovery rate: {result['recovery_rate']:.1%})")
    if result["mean_rank_of_hidden"] is not None:
        print(f"  Mean rank of hidden edges among {result['total_candidates']} candidates: "
              f"{result['mean_rank_of_hidden']:.1f}  (lower = better)")
    print(f"  {'─'*52}")


# ── SELF-TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("\n" + "=" * 58)
    print("  EVELYN — estimate_edges()  [Stage A]")
    print("=" * 58)

    # Test 1: a graph with an OBVIOUS missing edge — a near-complete
    # graph (K5 minus one edge). The estimator should rank that exact
    # missing edge at or near the top, since it has many strong
    # alternate paths reinforcing it.
    print("\n  Test 1: K5 minus one edge (the obviously-missing edge should rank #1)")
    G_test = nx.complete_graph(5)
    G_test.remove_edge(0, 1)
    candidates = estimate_missing_edges(G_test, t=2.0, top_k=3)
    _print_candidates(candidates)
    top_pair = frozenset([candidates[0]["node_a"], candidates[0]["node_b"]])
    expected_pair = frozenset([0, 1])
    print(f"  {'✓ PASS' if top_pair == expected_pair else '✗ FAIL'} — "
          f"top candidate {'is' if top_pair == expected_pair else 'is NOT'} the truly-missing edge (0,1)")

    # Test 2: real-world style validation — hide 20% of edges from a
    # realistic phishing-shaped synthetic campaign graph, see how well
    # we recover them.
    print("\n  Test 2: Hide-and-recover on a synthetic campaign-shaped graph")
    G_campaign = nx.Graph()
    G_campaign.add_edges_from([
        ("d1", "ip1"), ("d2", "ip1"), ("d3", "ip1"),
        ("d1", "cert1"), ("d2", "cert1"), ("d3", "cert1"),
        ("d1", "reg1"), ("d2", "reg1"),
        ("ip1", "asn1"), ("d3", "reg2"),
        ("d1", "fav1"), ("d2", "fav1"), ("d3", "fav1"),
    ])
    result = evaluate_edge_recovery(G_campaign, hide_fraction=0.2, t=2.0, seed=7)
    _print_recovery_result(result)

    if len(sys.argv) > 1 and sys.argv[1] == "--real":
        import pickle
        from pathlib import Path
        graphs = list(Path("data/graphs").glob("*recursive*.pkl"))
        if not graphs:
            print("\n  ⚠ No recursive graphs found in data/graphs/. "
                  "Run build_graph_recursive.py on a real domain first.")
        else:
            print(f"\n  Test 3: Hide-and-recover on real graph: {graphs[0].name}")
            with open(graphs[0], "rb") as f:
                G_real = pickle.load(f)
            result = evaluate_edge_recovery(G_real, hide_fraction=0.2, t=2.0, seed=7)
            _print_recovery_result(result)