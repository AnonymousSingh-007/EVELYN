# src/pipeline/build_graph_recursive.py
#
# PURPOSE: Expand a single-domain ego-graph (from build_graph.py) by
# recursively running build_graph() on its "leaf" nodes — cert_peer
# and co_host domains discovered one hop out. This grows your graph
# from a tight single-domain snapshot (~15-25 nodes) into a genuine
# multi-hop INFRASTRUCTURE MAP (potentially 50-100+ nodes), which is
# what real threat hunters mean by "mapping a campaign."
#
# WHY THIS IS RECURSION, NOT JUST A BIGGER build_graph():
#   Depth 0: build_graph(target_domain) — what you already have.
#   Depth 1: for each cert_peer/co_host found at depth 0, call
#            build_graph() on THAT domain too, merging results into
#            ONE combined graph.
#   Depth 2: repeat for depth-1's newly discovered leaves.
#   We cap depth (default 2) and total node budget (default 80) —
#   this can otherwise grow exponentially and never terminate, since
#   popular shared-hosting IPs can pull in hundreds of unrelated domains.
#
# THIS IS YOUR FIRST REAL "CRAWLER" — treat the budget/depth limits as
# load-bearing safety code, not optional extras.

import networkx as nx
from pathlib import Path
import pickle
import time

from src.pipeline.build_graph import build_graph
from src.pipeline.suspicion_filter import filter_suspicious_domains

GRAPHS_DIR = Path("data/graphs")

def build_graph_recursive(seed_url, label=None, max_depth=2, max_total_nodes=80,
                           max_expansions_per_level=5, verbose=True, save=False):
    combined = build_graph(seed_url, label=label, verbose=verbose, save=False)
    if combined.number_of_nodes() == 0: return combined
    expansion_log = []
    already_expanded = set()
    already_expanded.add(combined.graph.get("domain"))
    current_frontier = _get_expandable_leaves(combined, already_expanded)
    for depth in range(1, max_depth + 1):
        if not current_frontier:
            if verbose: print(f"  [depth {depth}] No more expandable leaves.")
            break
        if combined.number_of_nodes() >= max_total_nodes:
            if verbose: print(f"  [depth {depth}] Node budget ({max_total_nodes}) reached.")
            break
        to_expand = current_frontier[:max_expansions_per_level]
        if verbose: print(f"\n  [depth {depth}] Expanding {len(to_expand)} leaf domain(s)")
        new_frontier = []
        for leaf_domain in to_expand:
            if leaf_domain in already_expanded: continue
            if combined.number_of_nodes() >= max_total_nodes: break
            already_expanded.add(leaf_domain)
            leaf_url = f"https://{leaf_domain}"
            leaf_graph = build_graph(leaf_url, label=None, verbose=False, save=False)
            if leaf_graph.number_of_nodes() == 0:
                expansion_log.append({"depth": depth, "domain": leaf_domain, "status": "empty"})
                continue
            combined = nx.compose(combined, leaf_graph)
            new_leaves = _get_expandable_leaves(leaf_graph, already_expanded)
            new_frontier.extend(new_leaves)
            expansion_log.append({"depth": depth, "domain": leaf_domain, "status": "expanded",
                                  "nodes_added": leaf_graph.number_of_nodes()})
            if verbose: print(f"    ✓ {leaf_domain}: +{leaf_graph.number_of_nodes()} nodes (total {combined.number_of_nodes()})")
            time.sleep(1.0)
        seen = set()
        current_frontier = [d for d in new_frontier if not (d in seen or seen.add(d))]
    combined.graph["expansion_log"] = expansion_log
    combined.graph["final_node_count"] = combined.number_of_nodes()
    combined.graph["final_edge_count"] = combined.number_of_edges()
    if save:
        GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = combined.graph.get("domain", "unknown").replace(".", "_")
        label_tag = f"_L{label}" if label is not None else ""
        path = GRAPHS_DIR / f"{safe_name}{label_tag}_recursive.pkl"
        with open(path, "wb") as f: pickle.dump(combined, f)
        combined.graph["saved_path"] = str(path)
        if verbose: print(f"  Saved → {path}")
    return combined

def _get_expandable_leaves(G, already_expanded, apply_suspicion_filter=True):
    cert_peer_leaves = []
    co_host_candidates = []
    for node, data in G.nodes(data=True):
        node_type = data.get("type")
        if node in already_expanded: continue
        if node_type == "cert_peer": cert_peer_leaves.append(node)
        elif node_type == "co_host": co_host_candidates.append(node)
    if apply_suspicion_filter and co_host_candidates:
        co_host_leaves = filter_suspicious_domains(co_host_candidates, skip_whois=False, verbose=True)
    else:
        co_host_leaves = co_host_candidates
    return cert_peer_leaves + co_host_leaves


# ── SELF-TEST / CLI ────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "https://github.com"
    label = int(sys.argv[2]) if len(sys.argv) > 2 else None

    print("\n" + "=" * 58)
    print("  EVELYN — build_graph_recursive()")
    print("=" * 58)
    print("  NOTE: this makes MANY more network calls than build_graph() —")
    print("  expect 30-90+ seconds depending on how much infrastructure exists\n")

    G = build_graph_recursive(url, label=label, max_depth=2,
                              max_total_nodes=80, max_expansions_per_level=3,
                              verbose=True, save=True)
    print(f"\n  Final: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")