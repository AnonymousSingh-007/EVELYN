# src/quantum/compare_blind_spots.py
#
# PURPOSE: Run BOTH the quantum walk fingerprint AND the GraphSAGE
# baseline on the same set of graphs, and flag cases where they
# DISAGREE about whether two graphs are similar or different. A
# disagreement where the quantum walk correctly separates two
# structurally-different graphs that GraphSAGE collapses together
# is your "beyond the concept" finding — a demonstrated capability
# gap, not just a marginal accuracy improvement.
#
# This file works on BOTH synthetic known-blind-spot graphs (for
# proof-of-concept, available right now) AND your real saved graphs
# from data/graphs/ (once you have enough of them from batch_pipeline.py).

import pickle
from pathlib import Path
import numpy as np
import networkx as nx

from src.quantum.fingerprint import extract_fingerprint, fingerprint_distance
from src.quantum.gnn_baseline import graphsage_embed, embedding_distance


GRAPHS_DIR = Path("data/graphs")


def compare_pair(G_a: nx.Graph, G_b: nx.Graph, label_a: str = "A", label_b: str = "B") -> dict:
    """
    Computes BOTH fingerprints for a pair of graphs and reports both
    distance scores side by side, plus a verdict on whether this pair
    is a "blind spot" case (quantum walk sees a difference, GNN doesn't).
    """
    phi_a = extract_fingerprint(G_a, method="combined")["phi"]
    phi_b = extract_fingerprint(G_b, method="combined")["phi"]
    quantum_dist = fingerprint_distance(phi_a, phi_b, metric="cosine")

    emb_a = graphsage_embed(G_a, seed=1)["graph_embedding"]
    emb_b = graphsage_embed(G_b, seed=1)["graph_embedding"]
    gnn_dist = embedding_distance(emb_a, emb_b, metric="cosine")

    # Thresholds for "structurally similar" vs "structurally different" —
    # these are deliberately simple/interpretable for this diagnostic
    # tool. In Stage 4/5 you'll calibrate these against real labeled
    # campaign pairs instead of using a fixed cutoff.
    SIMILAR_THRESHOLD = 0.05

    quantum_says_different = quantum_dist > SIMILAR_THRESHOLD
    gnn_says_different     = gnn_dist > SIMILAR_THRESHOLD

    is_blind_spot = quantum_says_different and not gnn_says_different

    return {
        "label_a":               label_a,
        "label_b":               label_b,
        "quantum_distance":      quantum_dist,
        "gnn_distance":          gnn_dist,
        "quantum_says_different": quantum_says_different,
        "gnn_says_different":     gnn_says_different,
        "is_blind_spot":          is_blind_spot,
    }


def _print_comparison(result: dict) -> None:
    print(f"\n  {result['label_a']}  vs  {result['label_b']}")
    print(f"  {'─'*52}")
    print(f"  Quantum walk distance:  {result['quantum_distance']:.6f}  "
          f"→ {'DIFFERENT' if result['quantum_says_different'] else 'similar'}")
    print(f"  GraphSAGE distance:     {result['gnn_distance']:.6f}  "
          f"→ {'DIFFERENT' if result['gnn_says_different'] else 'similar'}")
    if result["is_blind_spot"]:
        print(f"  🎯 BLIND SPOT DETECTED: quantum walk distinguishes these graphs;")
        print(f"     GraphSAGE-style aggregation does NOT.")
    else:
        print(f"  No blind spot — both methods agree on similarity/difference.")
    print(f"  {'─'*52}")


def run_synthetic_blind_spot_suite() -> list:
    """
    Runs the comparison on graph pairs with ANALYTICALLY KNOWN GNN
    expressiveness limitations — these are not phishing-specific, but
    they prove the MECHANISM exists before you go looking for it in
    noisy real-world data. This is good methodology: establish the
    phenomenon on clean synthetic cases first, then test for it in
    your messier real dataset.
    """
    results = []

    # Case 1: C6 (single 6-cycle) vs 2xC3 (two disjoint triangles).
    # Textbook GNN blind spot — every node has identical 2-hop
    # neighbourhood structure in both graphs (related to 1-WL test
    # limitations), but they are NOT isomorphic (one is connected,
    # one isn't; different number of triangles; etc).
    G_c6 = nx.cycle_graph(6)
    G_2c3 = nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(3))
    results.append(compare_pair(G_c6, G_2c3, "C6 (one ring)", "2xC3 (two triangles)"))

    # Case 2: Two different "star" topologies that GraphSAGE mean-
    # aggregation can struggle with — a star graph (one hub, many
    # leaves) vs a "double star" (two hubs connected, each with leaves).
    # This is structurally closer to YOUR actual phishing graphs: hub
    # nodes (domains/IPs) with leaf nodes (cert_peers, co_hosts).
    G_star = nx.star_graph(8)   # 1 center, 8 leaves
    G_double_star = nx.Graph()
    G_double_star.add_edges_from([("hub1", "hub2")] +
                                  [("hub1", f"leaf{i}") for i in range(4)] +
                                  [("hub2", f"leaf{i+4}") for i in range(4)])
    results.append(compare_pair(G_star, G_double_star, "Single star (1 hub, 8 leaves)",
                                  "Double star (2 hubs, 4 leaves each)"))

    # Case 3: A REALISTIC phishing-topology pair — two "campaigns" that
    # are structurally distinct (one shares an IP across 3 domains, one
    # shares a CERT across 3 domains, both otherwise similar size/shape)
    G_campaign_ip_shared = nx.Graph()
    G_campaign_ip_shared.add_edges_from([
        ("d1", "ip_shared"), ("d2", "ip_shared"), ("d3", "ip_shared"),
        ("d1", "reg1"), ("d2", "reg2"), ("d3", "reg3"),
    ])
    G_campaign_cert_shared = nx.Graph()
    G_campaign_cert_shared.add_edges_from([
        ("d1", "cert_shared"), ("d2", "cert_shared"), ("d3", "cert_shared"),
        ("d1", "ip1"), ("d2", "ip2"), ("d3", "ip3"),
    ])
    results.append(compare_pair(G_campaign_ip_shared, G_campaign_cert_shared,
                                  "Campaign: shared-IP pattern", "Campaign: shared-cert pattern"))

    return results


def run_on_real_graphs(graphs_dir: Path = GRAPHS_DIR, max_pairs: int = 20) -> list:
    """
    Loads real saved graphs from data/graphs/ (produced by build_graph.py
    / batch_pipeline.py) and runs pairwise blind-spot comparisons across
    them. Run this once you have enough real graphs (Stage 1 batch runs)
    to make this meaningful — with only a handful of graphs, pairwise
    comparison won't have much statistical value, but it's still useful
    as a smoke test that the real pipeline data works end-to-end with
    both methods.
    """
    graph_files = list(graphs_dir.glob("*.pkl"))
    if len(graph_files) < 2:
        print(f"\n  ⚠ Only {len(graph_files)} graph(s) found in {graphs_dir}. "
              f"Need at least 2 to compare. Run batch_pipeline.py first.")
        return []

    graphs = []
    for path in graph_files[:max_pairs]:
        with open(path, "rb") as f:
            G = pickle.load(f)
        graphs.append((path.stem, G))

    print(f"\n  Loaded {len(graphs)} real graphs from {graphs_dir}")
    print(f"  Running pairwise blind-spot comparison "
          f"({len(graphs)*(len(graphs)-1)//2} pairs)...")

    results = []
    for i in range(len(graphs)):
        for j in range(i + 1, len(graphs)):
            name_a, G_a = graphs[i]
            name_b, G_b = graphs[j]
            if G_a.number_of_nodes() == 0 or G_b.number_of_nodes() == 0:
                continue
            result = compare_pair(G_a, G_b, name_a, name_b)
            results.append(result)

    return results


# ── SELF-TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("\n" + "=" * 58)
    print("  EVELYN — compare_blind_spots()")
    print("=" * 58)

    print("\n  PART 1: Synthetic known-blind-spot test suite")
    print("  (establishes the mechanism exists before searching real data)")
    synthetic_results = run_synthetic_blind_spot_suite()
    for r in synthetic_results:
        _print_comparison(r)

    blind_spot_count = sum(1 for r in synthetic_results if r["is_blind_spot"])
    print(f"\n  Synthetic suite: {blind_spot_count}/{len(synthetic_results)} pairs showed a blind spot")

    if "--real" in sys.argv:
        print("\n\n  PART 2: Real graph corpus (data/graphs/)")
        real_results = run_on_real_graphs()
        for r in real_results:
            _print_comparison(r)
        if real_results:
            blind_spot_count_real = sum(1 for r in real_results if r["is_blind_spot"])
            print(f"\n  Real corpus: {blind_spot_count_real}/{len(real_results)} pairs showed a blind spot")
    else:
        print("\n  (Run with --real flag to also test against your saved real graphs in data/graphs/)")