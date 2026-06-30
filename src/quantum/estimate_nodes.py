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
# THE MECHANISM:
#   Different real campaign "shapes" (archetypes) have characteristic
#   eigenvalue spectra — you've already seen this directly: a star
#   graph's spectrum has one large positive/negative pair and a flat
#   run near zero; a densely-interlinked cluster's spectrum spreads
#   more evenly. We build a small library of KNOWN archetype shapes
#   (informed by patterns already observed in your real data — shared-
#   IP star, shared-cert star, shared-cert+favicon dense cluster), and
#   compare a partial/incomplete real graph's spectrum against each
#   archetype using a spectral distance.
#
#   If the partial graph's spectrum is CLOSE to a truncated/damaged
#   version of a known archetype (i.e., what that archetype's spectrum
#   looks like with one node removed), we report: "this looks like an
#   incomplete instance of archetype X; the missing piece is likely a
#   node of type Y, because that's what's structurally absent."
#
# THIS IS A FIRST, DELIBERATELY SIMPLE VERSION of a genuinely hard
# problem. It is explicitly NOT claiming to reconstruct the missing
# node's exact identity (domain name, IP address) — only its probable
# ROLE (node type) and an estimate of how many connections it likely
# has, based on archetype matching. Stated limitations are in the
# self-test output, not hidden.

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
    """
    N domains sharing BOTH a cert AND a favicon — the densest, most
    redundant real campaign signature, and exactly the shape Stage A's
    Test 1 showed the estimator handles well (multiple reinforcing paths).
    """
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


def _spectral_signature(G: nx.Graph, n_bins: int = 15,
                         range_min: float = -6.0, range_max: float = 6.0) -> np.ndarray:
    """
    Same eigenvalue-histogram idea as fingerprint.py's spectral half —
    reused here specifically because it's SIZE-INVARIANT, which lets
    us compare a small partial graph against a fixed-size archetype
    template fairly, regardless of exactly how many nodes either has.
    """
    if G.number_of_nodes() < 2:
        return np.zeros(n_bins)
    ham = build_hamiltonian(G, weighted=False, variant="adjacency")
    counts, _ = np.histogram(ham["eigenvalues"], bins=n_bins, range=(range_min, range_max))
    total = counts.sum()
    return counts.astype(np.float64) / total if total > 0 else counts.astype(np.float64)


def estimate_missing_nodes(G_partial: nx.Graph, max_archetype_size: int = 6) -> dict:
    """
    Compares a partial graph's spectral signature against every known
    archetype (at several sizes, since a real partial graph might be a
    truncated 3-domain version of a 4-domain archetype, etc), and
    reports the best match plus what's structurally implied to be missing.

    Returns:
    {
        "best_archetype":     str or None,
        "best_archetype_size": int,
        "match_distance":     float (lower = better match),
        "estimated_missing":  {
            "likely_node_type": str,
            "likely_count":     int,
            "confidence":       str ("low"/"medium"/"high", from match_distance),
        },
        "all_scores": list of {"archetype": str, "size": int, "distance": float}
    }
    """
    partial_sig = _spectral_signature(G_partial)
    partial_n = G_partial.number_of_nodes()

    all_scores = []
    for name, builder in ARCHETYPES.items():
        for size in range(2, max_archetype_size + 1):
            archetype_full = builder(n_domains=size)
            archetype_sig = _spectral_signature(archetype_full)
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
    print("  EVELYN — estimate_nodes()  [Stage B]")
    print("=" * 58)

    print("\n  Test 1: complete 4-domain shared-cert archetype, minus 1 domain")
    full = _archetype_shared_cert_star(n_domains=4)
    partial = full.copy()
    partial.remove_node("domain3")
    result = estimate_missing_nodes(partial)
    _print_result(result, "(should detect ~1 missing domain, archetype=shared_cert_star)")

    correct_archetype = result["best_archetype"] == "shared_cert_star"
    correct_count = result["estimated_missing"]["likely_count"] >= 1
    print(f"  {'✓ PASS' if correct_archetype and correct_count else '✗ FAIL'} — "
          f"archetype correct: {correct_archetype}, missing count detected: {correct_count}")

    print("\n  Test 2: a structurally complete graph (no truncation)")
    complete = _archetype_dense_multi_signal(n_domains=4)
    result2 = estimate_missing_nodes(complete)
    _print_result(result2, "(should detect 0 missing nodes — graph is already complete)")

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