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


def build_graph_recursive(seed_url: str,
                           label: int = None,
                           max_depth: int = 2,
                           max_total_nodes: int = 80,
                           max_expansions_per_level: int = 5,
                           verbose: bool = True,
                           save: bool = False) -> nx.Graph:
    """
    Builds a multi-hop infrastructure graph by recursively expanding
    cert_peer and co_host nodes.

    Parameters:
        seed_url                  : the starting URL (depth 0)
        label                     : 1=phishing, 0=benign, None=unknown
        max_depth                 : how many hops out to expand (2 is a
                                     reasonable default — depth 3+ tends
                                     to pull in unrelated internet noise)
        max_total_nodes           : hard ceiling on the COMBINED graph,
                                     across all depths. Prevents runaway
                                     growth from popular shared-hosting IPs.
        max_expansions_per_level  : how many leaf domains to actually
                                     expand at each depth (prevents
                                     exploding API call counts — if a
                                     domain has 15 cert_peers, we only
                                     recursively expand the first 5)
        verbose                   : print expansion progress
        save                      : serialise final combined graph to disk

    Returns a single merged nx.Graph spanning all expanded domains.
    G.graph["expansion_log"] records exactly what was expanded at
    each depth — essential for your methods section's reproducibility.
    """

    # Start with the depth-0 graph — your existing, already-tested logic
    combined = build_graph(seed_url, label=label, verbose=verbose, save=False)

    if combined.number_of_nodes() == 0:
        return combined

    expansion_log = []
    already_expanded = set()   # domains we've already run build_graph() on
    already_expanded.add(combined.graph.get("domain"))

    current_frontier = _get_expandable_leaves(combined, already_expanded)

    for depth in range(1, max_depth + 1):
        if not current_frontier:
            if verbose:
                print(f"  [depth {depth}] No more expandable leaves. Stopping early.")
            break

        if combined.number_of_nodes() >= max_total_nodes:
            if verbose:
                print(f"  [depth {depth}] Node budget ({max_total_nodes}) reached. Stopping.")
            break

        # Only expand a LIMITED number of leaves per level — expanding
        # every single leaf at every depth is how you accidentally map
        # the entire internet instead of one campaign.
        to_expand = current_frontier[:max_expansions_per_level]

        if verbose:
            print(f"\n  [depth {depth}] Expanding {len(to_expand)} leaf domain(s): {to_expand}")

        new_frontier = []

        for leaf_domain in to_expand:
            if leaf_domain in already_expanded:
                continue
            if combined.number_of_nodes() >= max_total_nodes:
                break

            already_expanded.add(leaf_domain)

            # Run the FULL pipeline on this leaf — same DNS/WHOIS/cert/
            # ASN/geo/shared-hosting logic, just on a domain that was
            # discovered RATHER than given to us directly.
            leaf_url = f"https://{leaf_domain}"
            leaf_graph = build_graph(leaf_url, label=None, verbose=False, save=False)

            if leaf_graph.number_of_nodes() == 0:
                expansion_log.append({"depth": depth, "domain": leaf_domain, "status": "empty"})
                continue

            # Merge: nx.compose keeps all nodes/edges from BOTH graphs.
            # Where a node exists in both (e.g. leaf_domain might already
            # be in `combined` as a cert_peer LABEL), compose() takes
            # attributes from the SECOND graph — so leaf_graph's richer
            # attributes (it's now a fully-investigated domain, not just
            # a label) correctly overwrite the placeholder node.
            combined = nx.compose(combined, leaf_graph)

            # The leaf's own discovered cert_peers/co_hosts become the
            # NEXT frontier — this is the actual recursion step.
            new_leaves = _get_expandable_leaves(leaf_graph, already_expanded)
            new_frontier.extend(new_leaves)

            expansion_log.append({
                "depth": depth, "domain": leaf_domain, "status": "expanded",
                "nodes_added": leaf_graph.number_of_nodes(),
            })

            if verbose:
                print(f"    ✓ {leaf_domain}: +{leaf_graph.number_of_nodes()} nodes "
                      f"(total now {combined.number_of_nodes()})")

            time.sleep(1.0)   # courtesy delay — we're now making MANY more API calls

        # Deduplicate the next frontier, preserving order
        seen = set()
        current_frontier = [d for d in new_frontier if not (d in seen or seen.add(d))]

    combined.graph["expansion_log"] = expansion_log
    combined.graph["max_depth_reached"] = depth if "depth" in dir() else 0
    combined.graph["final_node_count"] = combined.number_of_nodes()
    combined.graph["final_edge_count"] = combined.number_of_edges()

    if verbose:
        print(f"\n  {'─'*52}")
        print(f"  RECURSIVE EXPANSION COMPLETE")
        print(f"  Total nodes: {combined.number_of_nodes()}  "
              f"(started at depth-0 with far fewer)")
        print(f"  Total edges: {combined.number_of_edges()}")
        print(f"  Domains expanded: {len(already_expanded)}")
        print(f"  {'─'*52}")

    if save:
        GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = combined.graph.get("domain", "unknown").replace(".", "_")
        label_tag = f"_L{label}" if label is not None else ""
        path = GRAPHS_DIR / f"{safe_name}{label_tag}_recursive.pkl"
        with open(path, "wb") as f:
            pickle.dump(combined, f)
        combined.graph["saved_path"] = str(path)
        if verbose:
            print(f"  Saved → {path}")

    return combined


def _get_expandable_leaves(G: nx.Graph, already_expanded: set) -> list:
    """
    Finds cert_peer and co_host nodes in a graph that haven't been
    expanded yet. These are domain-name STRINGS (not yet investigated),
    which is exactly what makes them "expandable" — we can run
    build_graph() on them to discover what infrastructure THEY connect to.
    """
    leaves = []
    for node, data in G.nodes(data=True):
        node_type = data.get("type")
        if node_type in ("cert_peer", "co_host") and node not in already_expanded:
            leaves.append(node)
    return leaves


def print_recursive_summary(G: nx.Graph) -> None:
    print(f"\n  {'─'*52}")
    print(f"  RECURSIVE GRAPH SUMMARY")
    print(f"  nodes: {G.number_of_nodes()}   edges: {G.number_of_edges()}")
    type_counts = {}
    for _, data in G.nodes(data=True):
        t = data.get("type", "?")
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, c in sorted(type_counts.items()):
        print(f"    {t:14s}: {c}")
    print(f"  expansion log entries: {len(G.graph.get('expansion_log', []))}")
    print(f"  {'─'*52}")


# ── SELF-TEST ──────────────────────────────────────────────────────────────
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
    print_recursive_summary(G)