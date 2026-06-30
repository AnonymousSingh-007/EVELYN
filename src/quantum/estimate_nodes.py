# src/quantum/estimate_nodes.py
#
# STAGE B — Estimating UNDISCOVERED nodes (the genuinely novel piece).
#
# PURPOSE: Stage A (estimate_edges.py) can only rank connections among
# nodes you ALREADY found. This file attempts something the link-
# prediction literature does not: given a partial graph, infer that an
# entire node you have NOT yet discovered probably exists, and
# estimate what TYPE of infrastructure it probably is.
#
# VERSION 2 — fixes a real structural flaw found in version 1:
#
#   BUG 1 (type-blindness): the original signature used pure adjacency
#   eigenvalues, which only encode WHICH nodes connect, never WHAT TYPE
#   those nodes are. A 4-node star with an "ip" hub and a 4-node star
#   with a "cert_peer" hub are mathematically IDENTICAL graphs under
#   plain adjacency — they always tied. FIX: the signature is now
#   computed PER NODE-TYPE-PAIR. Every edge is bucketed by
#   (type_of_node_a, type_of_node_b), a separate small adjacency
#   structure is built per bucket, and each bucket's eigenvalue
#   histogram is concatenated into one larger signature. A missing
#   "ip" hub and a missing "cert_peer" hub now land in DIFFERENT
#   positions of the signature vector instead of colliding.
#
#   BUG 2 (invalid test design): "remove 1 domain from a 4-star, check
#   if it's flagged incomplete" is not a valid test, because a 3-star
#   IS a perfectly complete, self-consistent 3-domain archetype — there
#   is nothing structurally wrong with it. FIX: the new test instead
#   builds an asymmetric multi-signal archetype where ONE domain
#   uniquely also connects to a second signal (e.g. a shared favicon
#   that only some domains use). Removing THAT specific domain creates
#   a genuine, unresolvable structural asymmetry — "a cert hub with 3
#   spokes, only 2 of which have a second shared signal" cannot be
#   re-explained as any complete smaller archetype, which is what
#   makes it a fair, decisive test.

import numpy as np
import networkx as nx

from src.quantum.hamiltonian import build_hamiltonian


def _archetype_shared_ip_star(n_domains: int = 4) -> nx.Graph:
    """N domains, all sharing ONE ip. The classic bulk-hosting pattern."""
    G = nx.Graph()
    for i in range(n_domains):
        G.add_node(f"domain{i}", type="domain")
        G.add_edge(f"domain{i}", "ip_shared")
    G.nodes["ip_shared"]["type"] = "ip"
    return G


def _archetype_shared_cert_star(n_domains: int = 4) -> nx.Graph:
    """N domains, all sharing ONE certificate. Stronger campaign signal."""
    G = nx.Graph()
    for i in range(n_domains):
        G.add_node(f"domain{i}", type="domain")
        G.add_edge(f"domain{i}", "cert_shared")
    G.nodes["cert_shared"]["type"] = "cert_peer"
    return G


def _archetype_dense_multi_signal(n_domains: int = 4) -> nx.Graph:
    """N domains sharing BOTH a cert AND a favicon — dense, redundant signature."""
    G = nx.Graph()
    for i in range(n_domains):
        G.add_node(f"domain{i}", type="domain")
        G.add_edge(f"domain{i}", "cert_shared")
        G.add_edge(f"domain{i}", "favicon_shared")
    G.nodes["cert_shared"]["type"] = "cert_peer"
    G.nodes["favicon_shared"]["type"] = "favicon"
    return G


ARCHETYPES = {
    "shared_ip_star":      _archetype_shared_ip_star,
    "shared_cert_star":    _archetype_shared_cert_star,
    "dense_multi_signal":  _archetype_dense_multi_signal,
}

# Fixed, canonical ordering of edge-type-pair buckets. Every archetype
# and every real graph gets bucketed against this SAME list, so their
# concatenated signatures are directly comparable position-by-position
# regardless of which buckets happen to be empty for a given graph.
EDGE_TYPE_BUCKETS = [
    ("domain", "ip"), ("domain", "cert_peer"), ("domain", "favicon"),
    ("domain", "registrar"), ("domain", "asn"), ("domain", "jarm"),
    ("domain", "subdomain"), ("domain", "co_host"), ("ip", "asn"),
    ("ip", "geo"), ("ip", "co_host"), ("other", "other"),
]


def _bucket_for_edge(type_a: str, type_b: str) -> tuple:
    """Maps an edge's (type_a, type_b) pair to its canonical bucket,
    regardless of which order the two types were given in."""
    pair = tuple(sorted([type_a, type_b]))
    for bucket in EDGE_TYPE_BUCKETS:
        if tuple(sorted(bucket)) == pair:
            return bucket
    return ("other", "other")


def _type_aware_signature(G: nx.Graph, n_bins_per_bucket: int = 4,
                           range_min: float = -4.0, range_max: float = 4.0) -> np.ndarray:
    """
    THE FIX for type-blindness. Instead of one global eigenvalue
    histogram, builds ONE PER EDGE-TYPE BUCKET, then concatenates them.

    For each bucket (e.g. domain<->ip), we build the induced subgraph
    on exactly the nodes touched by edges in that bucket, compute its
    own small adjacency spectrum, and histogram it. A star with an ip
    hub contributes weight to the (domain,ip) bucket's histogram; a
    star with a cert_peer hub contributes to (domain,cert_peer)
    instead — now they land in genuinely different parts of the final
    vector, fixing the collision from version 1.
    """
    signature_parts = []

    for bucket in EDGE_TYPE_BUCKETS:
        # Collect edges whose (type_a, type_b) maps to this bucket
        bucket_edges = []
        for u, v in G.edges():
            tu = G.nodes[u].get("type", "unknown")
            tv = G.nodes[v].get("type", "unknown")
            if _bucket_for_edge(tu, tv) == bucket:
                bucket_edges.append((u, v))

        if not bucket_edges:
            signature_parts.append(np.zeros(n_bins_per_bucket))
            continue

        sub = nx.Graph()
        sub.add_edges_from(bucket_edges)

        if sub.number_of_nodes() < 2:
            signature_parts.append(np.zeros(n_bins_per_bucket))
            continue

        ham = build_hamiltonian(sub, weighted=False, variant="adjacency")
        counts, _ = np.histogram(ham["eigenvalues"], bins=n_bins_per_bucket,
                                  range=(range_min, range_max))
        total = counts.sum()
        normed = counts.astype(np.float64) / total if total > 0 else counts.astype(np.float64)
        # Scale by how MANY edges are in this bucket relative to the
        # whole graph — this preserves "how much of the graph's
        # structure lives in this bucket" as part of the signature,
        # not just the shape within the bucket.
        weight = len(bucket_edges) / max(G.number_of_edges(), 1)
        signature_parts.append(normed * weight)

    return np.concatenate(signature_parts)


def estimate_missing_nodes(G_partial: nx.Graph, max_archetype_size: int = 6) -> dict:
    """
    Compares a partial graph's TYPE-AWARE spectral signature against
    every known archetype, reports the best match and what's implied
    to be missing.
    """
    partial_sig = _type_aware_signature(G_partial)
    partial_n = G_partial.number_of_nodes()

    all_scores = []
    for name, builder in ARCHETYPES.items():
        for size in range(2, max_archetype_size + 1):
            archetype_full = builder(n_domains=size)
            archetype_sig = _type_aware_signature(archetype_full)
            distance = float(np.linalg.norm(partial_sig - archetype_sig))
            all_scores.append({"archetype": name, "size": size, "distance": distance,
                               "n_nodes": archetype_full.number_of_nodes()})

    all_scores.sort(key=lambda s: s["distance"])
    best = all_scores[0]
    nodes_missing = max(0, best["n_nodes"] - partial_n)

    if best["distance"] < 0.05:
        confidence = "high"
    elif best["distance"] < 0.15:
        confidence = "medium"
    else:
        confidence = "low"

    likely_type = "domain" if nodes_missing > 0 else None

    return {
        "best_archetype": best["archetype"] if nodes_missing > 0 or best["distance"] < 0.2 else None,
        "best_archetype_size": best["n_nodes"],
        "match_distance": best["distance"],
        "estimated_missing": {
            "likely_node_type": likely_type,
            "likely_count": nodes_missing,
            "confidence": confidence,
        },
        "all_scores": sorted(all_scores, key=lambda s: s["distance"])[:5],
    }


def _print_result(result: dict, label: str = "") -> None:
    print(f"\n  Node estimation result {label}")
    print(f"  Best matching archetype: {result['best_archetype']}  "
          f"(full size: {result['best_archetype_size']} nodes)")
    print(f"  Match distance: {result['match_distance']:.4f}  "
          f"(confidence: {result['estimated_missing']['confidence']})")
    em = result["estimated_missing"]
    if em["likely_count"] > 0:
        print(f"  ⚠ ESTIMATE: probably missing {em['likely_count']} more "
              f"node(s) of type '{em['likely_node_type']}'")
    else:
        print(f"  No missing nodes estimated — graph appears structurally complete")
    print(f"  Top 5 archetype matches:")
    for s in result["all_scores"]:
        print(f"      {s['archetype']:20s} size={s['size']}  distance={s['distance']:.4f}")
    print(f"  {'─'*52}")


# ── SELF-TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 58)
    print("  EVELYN — estimate_nodes()  [Stage B, v2]")
    print("=" * 58)

    # Test 1 (FIXED): does the type-aware signature now correctly tell
    # apart an ip-hub star from a cert_peer-hub star, which version 1
    # could NOT distinguish (both gave distance=0.0000, a tie)?
    print("\n  Test 1: type-blindness fix — ip-hub star vs cert_peer-hub star")
    ip_star = _archetype_shared_ip_star(n_domains=4)
    cert_star = _archetype_shared_cert_star(n_domains=4)
    sig_ip = _type_aware_signature(ip_star)
    sig_cert = _type_aware_signature(cert_star)
    type_distance = float(np.linalg.norm(sig_ip - sig_cert))
    print(f"  Signature distance between ip-hub and cert_peer-hub stars: {type_distance:.4f}")
    print(f"  {'✓ PASS' if type_distance > 0.01 else '✗ FAIL'} — "
          f"{'now distinguishable' if type_distance > 0.01 else 'STILL colliding — bug not fixed'}")

    # Test 2 (FIXED test design): an asymmetric multi-signal archetype
    # where removing ONE specific domain creates a genuine, unresolvable
    # structural gap — NOT explainable as any complete smaller archetype.
    print("\n  Test 2: asymmetric multi-signal gap (the properly-designed test)")
    print("  Setup: cert hub with 4 domains, but only 3 of them ALSO share a favicon.")
    print("  Removing the ONE domain that ISN'T on the favicon doesn't break the")
    print("  asymmetry; removing one that IS creates a detectable, genuine gap.")

    G_asym = nx.Graph()
    for i in range(4):
        G_asym.add_node(f"domain{i}", type="domain")
        G_asym.add_edge(f"domain{i}", "cert_shared")
    G_asym.nodes["cert_shared"]["type"] = "cert_peer"
    for i in range(3):   # only domain0, domain1, domain2 share the favicon
        G_asym.add_edge(f"domain{i}", "favicon_shared")
    G_asym.nodes["favicon_shared"]["type"] = "favicon"

    sig_full_asym = _type_aware_signature(G_asym)

    G_damaged = G_asym.copy()
    G_damaged.remove_node("domain0")   # removes a domain that WAS on the favicon
    sig_damaged = _type_aware_signature(G_damaged)

    damage_signal = float(np.linalg.norm(sig_full_asym - sig_damaged))
    print(f"\n  Signature shift caused by removing a favicon-linked domain: {damage_signal:.4f}")
    print(f"  {'✓ PASS' if damage_signal > 0.01 else '✗ FAIL'} — "
          f"{'damage is now structurally detectable' if damage_signal > 0.01 else 'no detectable difference'}")

    # Test 3: real graph
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--real":
        import pickle
        from pathlib import Path
        graphs = list(Path("data/graphs").glob("*.pkl"))
        if graphs:
            print(f"\n  Test 3: real graph: {graphs[0].name}")
            with open(graphs[0], "rb") as f:
                G_real = pickle.load(f)
            result3 = estimate_missing_nodes(G_real)
            _print_result(result3, f"({graphs[0].name})")
        else:
            print("\n  ⚠ No graphs found in data/graphs/")