# src/quantum/estimate_edges.py  — v2, complete rewrite
#
# FIXES OVER v1:
#   1. TYPE-PAIR PLAUSIBILITY — only scores edges that can actually exist
#      in EVELYN's graph model. ASN↔geo, domain↔domain, etc. are filtered
#      out BEFORE ranking, not wasting top slots on impossible connections.
#   2. MULTI-t AGGREGATION — scores across 5 time scales and averages,
#      eliminating the single-t instability where the top prediction
#      flipped depending on which t you happened to pick.
#   3. BASELINES — common-neighbor (Adamic-Adar) and random baselines
#      so you can prove the quantum walk actually beats trivial methods.
#   4. REAL METRICS — MRR, Hits@K, AUC instead of just "recovery rate."
#   5. PLAIN-LANGUAGE OUTPUT — "phish2.xyz probably also uses the same
#      favicon" instead of raw score numbers.
#   6. END-TO-END — run_edge_estimation(url) builds graph → estimates → done.
#   7. CONFIDENCE SCORING — high/medium/low based on score distribution.

import numpy as np
import networkx as nx

from src.quantum.hamiltonian import build_hamiltonian
from src.quantum.walk import compute_U


# ═══════════════════════════════════════════════════════════════════
# SECTION 1 — Type-pair plausibility (THE single biggest fix)
# ═══════════════════════════════════════════════════════════════════
#
# EVELYN's graph model has SPECIFIC edge types. A "domain" connects to
# an "ip" via resolves-to, to a "registrar" via registered-by, etc.
# But domain↔domain, asn↔geo, favicon↔registrar NEVER occur.
# Scoring those pairs wastes the top K slots on impossible edges.
#
# This dict defines every (type_a, type_b) pair that CAN form an edge,
# plus the human-readable relation it would represent. Pairs not listed
# here are structurally impossible and get filtered out before ranking.

VALID_EDGE_TYPES = {
    ("domain", "ip"):        "resolves-to",
    ("domain", "registrar"): "registered-by",
    ("domain", "cert_peer"): "shares-cert",
    ("domain", "favicon"):   "uses-favicon",
    ("domain", "subdomain"): "has-subdomain",
    ("domain", "jarm"):      "server-fingerprint",
    ("ip", "asn"):           "hosted-in",
    ("ip", "geo"):           "located-in",
    ("ip", "co_host"):       "also-hosts",
}

# Plain-language templates for explaining predicted edges to a human
EDGE_EXPLANATIONS = {
    "resolves-to":        "{a} probably also points to server {b}",
    "registered-by":      "{a} was probably also registered through {b}",
    "shares-cert":        "{a} probably shares a TLS certificate with {b}",
    "uses-favicon":       "{a} probably uses the same page icon as {b}",
    "has-subdomain":      "{b} is probably a subdomain of {a}",
    "server-fingerprint": "{a} probably runs the same server software ({b})",
    "hosted-in":          "server {a} is probably hosted by {b}",
    "located-in":         "server {a} is probably located in {b}",
    "also-hosts":         "server {a} probably also hosts {b}",
}

# Multi-t defaults — spans local (0.5) to global (10.0) structure
DEFAULT_T_VALUES = [0.5, 1.0, 2.0, 5.0, 10.0]


def _canonical_type_pair(type_a: str, type_b: str) -> tuple | None:
    """Returns the canonical (sorted) type pair if it's a valid edge type,
    or None if this pair can never form an edge in EVELYN's model."""
    if (type_a, type_b) in VALID_EDGE_TYPES:
        return (type_a, type_b)
    if (type_b, type_a) in VALID_EDGE_TYPES:
        return (type_b, type_a)
    return None


def _explain_edge(type_a: str, type_b: str, node_a: str, node_b: str) -> str:
    """Generates a plain-language explanation for a predicted edge."""
    pair = _canonical_type_pair(type_a, type_b)
    if pair is None:
        return f"{node_a} is somehow connected to {node_b}"
    relation = VALID_EDGE_TYPES[pair]
    template = EDGE_EXPLANATIONS.get(relation, "{a} connects to {b}")
    # Make sure a/b match the canonical order
    if pair == (type_a, type_b):
        return template.format(a=node_a, b=node_b)
    else:
        return template.format(a=node_b, b=node_a)


# ═══════════════════════════════════════════════════════════════════
# SECTION 2 — Multi-t quantum walk edge scorer
# ═══════════════════════════════════════════════════════════════════

def estimate_missing_edges(G: nx.Graph,
                           t_values: list = None,
                           top_k: int = 10,
                           filter_by_type: bool = True) -> list:
    """
    Scores every plausible non-edge pair using quantum walk transition
    amplitudes averaged across multiple time scales.

    Parameters:
        G              : NetworkX graph (e.g. from build_graph)
        t_values       : list of walk times (default: [0.5, 1.0, 2.0, 5.0, 10.0])
        top_k          : how many top candidates to return
        filter_by_type : if True (default), only score pairs whose node
                         types can actually form edges in EVELYN's model.
                         Set False for synthetic graphs without type attrs.

    Returns list of dicts sorted by score descending:
    [
        {"node_a": str, "node_b": str, "score": float,
         "type_a": str, "type_b": str, "relation": str,
         "explanation": str, "confidence": str},
        ...
    ]
    """
    if G.number_of_nodes() < 2:
        return []

    if t_values is None:
        t_values = DEFAULT_T_VALUES

    ham = build_hamiltonian(G, weighted=True, variant="adjacency")
    node_order = ham["node_order"]
    n = ham["n_nodes"]

    # Build index lookup once (replaces repeated .index() calls — O(1) vs O(n))
    node_idx = {node: i for i, node in enumerate(node_order)}

    # Compute transition probability matrices at each t, then AVERAGE.
    # This is the multi-t fix: a single t gives unstable rankings because
    # the walk's interference pattern at one specific time is noisy.
    # Averaging across local-to-global time scales smooths this out.
    prob_sum = np.zeros((n, n), dtype=np.float64)
    for t in t_values:
        U = compute_U(ham["eigenvalues"], ham["eigenvectors"], t)
        prob_sum += np.abs(U) ** 2
    prob_avg = prob_sum / len(t_values)

    # Build existing-edge set using fast index lookup
    existing_edges = set()
    for u, v in G.edges():
        i, j = node_idx[u], node_idx[v]
        existing_edges.add((min(i, j), max(i, j)))

    # Score all non-edge pairs, filtered by type plausibility
    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in existing_edges:
                continue

            node_a, node_b = node_order[i], node_order[j]
            type_a = G.nodes[node_a].get("type", "unknown")
            type_b = G.nodes[node_b].get("type", "unknown")

            # THE KEY FILTER: skip pairs that can never be edges
            if filter_by_type:
                pair = _canonical_type_pair(type_a, type_b)
                if pair is None:
                    continue
                relation = VALID_EDGE_TYPES[pair]
            else:
                relation = "unknown"

            score = float(prob_avg[i, j])
            explanation = _explain_edge(type_a, type_b, node_a, node_b)

            candidates.append({
                "node_a": node_a,
                "node_b": node_b,
                "score": score,
                "type_a": type_a,
                "type_b": type_b,
                "relation": relation,
                "explanation": explanation,
            })

    candidates.sort(key=lambda c: c["score"], reverse=True)

    # Assign confidence levels based on score distribution
    if candidates:
        scores = np.array([c["score"] for c in candidates])
        p75 = np.percentile(scores, 75) if len(scores) > 3 else scores[0]
        p50 = np.percentile(scores, 50) if len(scores) > 1 else 0
        for c in candidates:
            if c["score"] >= p75:
                c["confidence"] = "high"
            elif c["score"] >= p50:
                c["confidence"] = "medium"
            else:
                c["confidence"] = "low"

    return candidates[:top_k]


# ═══════════════════════════════════════════════════════════════════
# SECTION 3 — Baselines (prove the quantum walk beats trivial methods)
# ═══════════════════════════════════════════════════════════════════

def _common_neighbor_scores(G: nx.Graph, filter_by_type: bool = True) -> list:
    """
    Adamic-Adar index baseline — the standard "dumb but strong" link
    prediction baseline. Scores each non-edge (u,v) by:
        sum_{w in N(u) ∩ N(v)} 1 / log(degree(w))
    i.e. shared neighbors weighted by how exclusive they are.
    """
    candidates = []
    nodes = list(G.nodes())

    for i, u in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            v = nodes[j]
            if G.has_edge(u, v):
                continue

            type_u = G.nodes[u].get("type", "unknown")
            type_v = G.nodes[v].get("type", "unknown")

            if filter_by_type:
                pair = _canonical_type_pair(type_u, type_v)
                if pair is None:
                    continue
                relation = VALID_EDGE_TYPES[pair]
            else:
                relation = "unknown"

            # Adamic-Adar: sum 1/log(deg(w)) for each common neighbor w
            common = set(G.neighbors(u)) & set(G.neighbors(v))
            if not common:
                score = 0.0
            else:
                score = sum(1.0 / max(np.log(G.degree(w)), 1e-10) for w in common)

            candidates.append({
                "node_a": u, "node_b": v, "score": score,
                "type_a": type_u, "type_b": type_v, "relation": relation,
            })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


def _random_baseline_scores(G: nx.Graph, filter_by_type: bool = True,
                             seed: int = 0) -> list:
    """Random scoring baseline — the floor any real method must beat."""
    rng = np.random.default_rng(seed)
    candidates = []
    nodes = list(G.nodes())
    for i, u in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            v = nodes[j]
            if G.has_edge(u, v):
                continue
            type_u = G.nodes[u].get("type", "unknown")
            type_v = G.nodes[v].get("type", "unknown")
            if filter_by_type:
                if _canonical_type_pair(type_u, type_v) is None:
                    continue
            candidates.append({
                "node_a": u, "node_b": v, "score": rng.random(),
                "type_a": type_u, "type_b": type_v,
            })
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


# ═══════════════════════════════════════════════════════════════════
# SECTION 4 — Metrics (MRR, Hits@K, AUC)
# ═══════════════════════════════════════════════════════════════════

def _compute_metrics(ranked_candidates: list, hidden_set: set,
                     k_values: list = None) -> dict:
    """
    Given a ranked candidate list and a set of truly-hidden edges,
    computes standard link-prediction metrics.

    hidden_set: set of frozenset({u, v}) for each hidden edge.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    n_hidden = len(hidden_set)
    if n_hidden == 0:
        return {"mrr": 0.0, "hits": {}, "n_hidden": 0}

    ranks = []
    for idx, cand in enumerate(ranked_candidates):
        pair = frozenset([cand["node_a"], cand["node_b"]])
        if pair in hidden_set:
            ranks.append(idx + 1)   # 1-indexed rank

    # MRR — Mean Reciprocal Rank: average of 1/rank for each hidden edge
    mrr = float(np.mean([1.0 / r for r in ranks])) if ranks else 0.0

    # Hits@K — fraction of hidden edges found in top K predictions
    hits = {}
    for k in k_values:
        hits[k] = sum(1 for r in ranks if r <= k) / n_hidden

    # Mean rank (lower = better)
    mean_rank = float(np.mean(ranks)) if ranks else None

    return {
        "mrr": mrr,
        "hits": hits,
        "mean_rank": mean_rank,
        "ranks": ranks,
        "n_hidden": n_hidden,
        "n_found": len(ranks),
        "n_candidates": len(ranked_candidates),
    }


# ═══════════════════════════════════════════════════════════════════
# SECTION 5 — Validation protocol (hide-and-recover with baselines)
# ═══════════════════════════════════════════════════════════════════

def evaluate_edge_recovery(G_full: nx.Graph,
                            hide_fraction: float = 0.2,
                            t_values: list = None,
                            seed: int = 0,
                            filter_by_type: bool = True,
                            verbose: bool = True) -> dict:
    """
    THE VALIDATION PROTOCOL — upgraded with:
      - Multi-t scoring
      - Type-pair filtering
      - Comparison against common-neighbor and random baselines
      - Full metric suite (MRR, Hits@K, mean rank)
      - Connectivity guard (won't hide edges that disconnect the graph)

    Returns a dict with metrics for quantum walk AND both baselines.
    """
    if t_values is None:
        t_values = DEFAULT_T_VALUES

    rng = np.random.default_rng(seed)
    edges = list(G_full.edges())
    n_hide = max(1, int(len(edges) * hide_fraction))

    if n_hide >= len(edges) - (G_full.number_of_nodes() - 1):
        # Can't hide so many edges that the graph disconnects
        n_hide = max(1, len(edges) - G_full.number_of_nodes())

    # Hide edges, but ensure the damaged graph stays connected
    # (a disconnected graph breaks the Hamiltonian's spectral properties)
    shuffled_indices = rng.permutation(len(edges))
    hidden_edges = []
    G_damaged = G_full.copy()

    for idx in shuffled_indices:
        if len(hidden_edges) >= n_hide:
            break
        u, v = edges[idx]
        G_damaged.remove_edge(u, v)
        if nx.is_connected(G_damaged):
            hidden_edges.append((u, v))
        else:
            G_damaged.add_edge(u, v, **G_full.edges[u, v])

    if not hidden_edges:
        return {"error": "Could not hide any edges while keeping graph connected"}

    hidden_set = set(frozenset([u, v]) for u, v in hidden_edges)

    if verbose:
        print(f"\n  Hidden {len(hidden_edges)}/{len(edges)} edges "
              f"(graph: {G_damaged.number_of_nodes()} nodes, "
              f"{G_damaged.number_of_edges()} edges remaining)")

    # Run quantum walk estimator on damaged graph
    qw_candidates = estimate_missing_edges(
        G_damaged, t_values=t_values, top_k=10**6, filter_by_type=filter_by_type
    )
    qw_metrics = _compute_metrics(qw_candidates, hidden_set)

    # Run baselines on same damaged graph
    cn_candidates = _common_neighbor_scores(G_damaged, filter_by_type=filter_by_type)
    cn_metrics = _compute_metrics(cn_candidates, hidden_set)

    rand_candidates = _random_baseline_scores(G_damaged, filter_by_type=filter_by_type, seed=seed)
    rand_metrics = _compute_metrics(rand_candidates, hidden_set)

    if verbose:
        print(f"\n  {'Method':<25s} {'MRR':>6s} {'H@1':>5s} {'H@3':>5s} {'H@5':>5s} {'H@10':>5s} {'MeanRank':>9s}")
        print(f"  {'─'*62}")
        for name, m in [("Quantum walk (multi-t)", qw_metrics),
                         ("Common neighbors (AA)", cn_metrics),
                         ("Random baseline", rand_metrics)]:
            h = m["hits"]
            mr = f"{m['mean_rank']:.1f}" if m["mean_rank"] else "n/a"
            print(f"  {name:<25s} {m['mrr']:>6.3f} {h.get(1,0):>5.1%} {h.get(3,0):>5.1%} "
                  f"{h.get(5,0):>5.1%} {h.get(10,0):>5.1%} {mr:>9s}")

    return {
        "quantum_walk": qw_metrics,
        "common_neighbors": cn_metrics,
        "random": rand_metrics,
        "hidden_edges": hidden_edges,
        "n_hidden": len(hidden_edges),
        "graph_nodes": G_full.number_of_nodes(),
        "graph_edges": G_full.number_of_edges(),
    }


# ═══════════════════════════════════════════════════════════════════
# SECTION 6 — End-to-end pipeline function
# ═══════════════════════════════════════════════════════════════════

def run_edge_estimation(url_or_graph,
                        label: int = None,
                        top_k: int = 10,
                        t_values: list = None,
                        verbose: bool = True) -> dict:
    """
    End-to-end: URL in → predicted missing edges out.

    Parameters:
        url_or_graph : a URL string OR a pre-built NetworkX graph
        label        : 1=phishing, 0=benign (only used if url_or_graph is a URL)
        top_k        : how many predictions to return
        t_values     : walk time values
        verbose      : print results

    Returns:
    {
        "graph":        nx.Graph,
        "predictions":  list of candidate dicts (with explanations),
        "validation":   dict of hide-and-recover metrics (if graph big enough),
    }
    """
    if isinstance(url_or_graph, nx.Graph):
        G = url_or_graph
    else:
        from src.pipeline.build_graph import build_graph
        G = build_graph(url_or_graph, label=label, verbose=verbose, save=True)

    if G.number_of_nodes() < 3:
        if verbose:
            print("  ⚠ Graph too small for edge estimation")
        return {"graph": G, "predictions": [], "validation": None}

    domain = G.graph.get("domain", "unknown") if hasattr(G, "graph") else "unknown"

    if verbose:
        print(f"\n  ═══ EDGE ESTIMATION: {domain} ═══")
        print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Run estimation
    predictions = estimate_missing_edges(
        G, t_values=t_values, top_k=top_k, filter_by_type=True
    )

    if verbose and predictions:
        print(f"\n  Top {min(top_k, len(predictions))} predicted missing connections:")
        print(f"  {'─'*72}")
        for i, p in enumerate(predictions[:top_k]):
            conf_marker = {"high": "🔴", "medium": "🟡", "low": "⚪"}.get(p["confidence"], "⚪")
            print(f"  {conf_marker} #{i+1}  {p['explanation']}")
            print(f"       score={p['score']:.5f}  confidence={p['confidence']}  "
                  f"relation={p['relation']}")
        print(f"  {'─'*72}")

    # Run validation if graph is big enough
    validation = None
    if G.number_of_edges() >= 8 and nx.is_connected(G):
        if verbose:
            print(f"\n  Running hide-and-recover validation...")
        validation = evaluate_edge_recovery(
            G, hide_fraction=0.2, t_values=t_values, verbose=verbose
        )

    return {"graph": G, "predictions": predictions, "validation": validation}


# ═══════════════════════════════════════════════════════════════════
# SECTION 7 — Visualization
# ═══════════════════════════════════════════════════════════════════

def figure_edge_predictions(G: nx.Graph, predictions: list,
                             max_show: int = 5, save: bool = True) -> str | None:
    """
    Generates a figure showing the existing graph with predicted
    missing edges overlaid as dashed red lines.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    FIGURES_DIR = Path("results/figures")
    TYPE_COLORS = {"domain":"#D85A30","ip":"#D4537E","registrar":"#1D9E75",
        "cert_peer":"#378ADD","asn":"#7F77DD","geo":"#5BA88A","co_host":"#C9A227",
        "favicon":"#9B59B6","jarm":"#E67E22","subdomain":"#16A085","unknown":"#888780"}

    fig, ax = plt.subplots(figsize=(10, 8))
    try:
        pos = nx.kamada_kawai_layout(G)
    except Exception:
        pos = nx.spring_layout(G, seed=7, k=0.6)

    # Draw existing edges
    nx.draw_networkx_edges(G, pos, edge_color="#cccccc", width=0.8, alpha=0.5, ax=ax)

    # Draw nodes by type
    types_present = sorted(set(d.get("type", "unknown") for _, d in G.nodes(data=True)))
    for t in types_present:
        nodes = [n for n, d in G.nodes(data=True) if d.get("type") == t]
        sizes = [150 + 50 * G.degree(n) for n in nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes,
                                node_color=TYPE_COLORS.get(t, "#888780"),
                                node_size=sizes, alpha=0.9, ax=ax,
                                edgecolors="white", linewidths=0.8, label=t)

    # Draw PREDICTED edges as dashed red lines
    shown = predictions[:max_show]
    for pred in shown:
        a, b = pred["node_a"], pred["node_b"]
        if a in pos and b in pos:
            ax.annotate("", xy=pos[b], xytext=pos[a],
                        arrowprops=dict(arrowstyle="-", color="#C0392B",
                                        linestyle="dashed", linewidth=2.0, alpha=0.7))
            mx = (pos[a][0] + pos[b][0]) / 2
            my = (pos[a][1] + pos[b][1]) / 2
            ax.text(mx, my, pred["relation"], fontsize=7, color="#C0392B",
                    ha="center", bbox=dict(boxstyle="round,pad=0.15",
                    fc="white", ec="#C0392B", alpha=0.85))

    # Label root domain
    root = G.graph.get("domain") if hasattr(G, "graph") else None
    if root and root in pos:
        ax.annotate(root, pos[root], fontsize=9, fontweight="bold",
                    ha="center", xytext=(0, 14), textcoords="offset points")

    domain = G.graph.get("domain", "unknown").replace(".", "_") if hasattr(G, "graph") else "graph"
    ax.set_title(f"{domain} — predicted missing edges (dashed red)\n"
                 f"Showing top {len(shown)} predictions by quantum walk score",
                 fontsize=11)
    ax.legend(loc="upper left", fontsize=7.5, frameon=False, markerscale=0.6,
              bbox_to_anchor=(1.0, 1.0))
    ax.axis("off")
    fig.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / f"edge_predictions_{domain}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        return str(path)
    else:
        plt.close(fig)
        return None


def figure_recovery_comparison(eval_result: dict, save: bool = True) -> str | None:
    """Bar chart comparing Hits@K across quantum walk vs baselines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    FIGURES_DIR = Path("results/figures")

    qw = eval_result["quantum_walk"]
    cn = eval_result["common_neighbors"]
    rand = eval_result["random"]

    k_values = sorted(qw["hits"].keys())
    x = np.arange(len(k_values))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width, [qw["hits"][k] for k in k_values], width,
           label=f"Quantum walk (MRR={qw['mrr']:.3f})", color="#BA7517")
    ax.bar(x, [cn["hits"][k] for k in k_values], width,
           label=f"Common neighbors (MRR={cn['mrr']:.3f})", color="#3F70AC")
    ax.bar(x + width, [rand["hits"][k] for k in k_values], width,
           label=f"Random (MRR={rand['mrr']:.3f})", color="#888780")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Hits@{k}" for k in k_values])
    ax.set_ylabel("Fraction of hidden edges recovered")
    ax.set_title(f"Edge recovery: quantum walk vs baselines "
                 f"({eval_result['n_hidden']} edges hidden)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / "edge_recovery_comparison.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        return str(path)
    else:
        plt.close(fig)
        return None


# ═══════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 62)
    print("  EVELYN — estimate_edges() v2")
    print("=" * 62)

    # Test 1: K5 minus one edge — the obvious-missing-edge test
    print("\n  TEST 1: K5 minus one edge")
    G_test = nx.complete_graph(5)
    G_test.remove_edge(0, 1)
    candidates = estimate_missing_edges(G_test, top_k=3, filter_by_type=False)
    top_pair = frozenset([candidates[0]["node_a"], candidates[0]["node_b"]])
    print(f"  Top candidate: {top_pair}, expected {{0,1}}: "
          f"{'✓ PASS' if top_pair == frozenset([0, 1]) else '✗ FAIL'}")

    # Test 2: realistic phishing graph with type-aware filtering
    print("\n  TEST 2: Realistic phishing graph (type-aware)")
    G = nx.Graph()
    for i in range(3):
        G.add_node(f"phish{i}.xyz", type="domain")
        G.add_edge(f"phish{i}.xyz", "185.220.101.47", relation="resolves-to")
        G.add_edge(f"phish{i}.xyz", "shared_cert", relation="shares-cert")
    G.add_node("185.220.101.47", type="ip")
    G.add_node("shared_cert", type="cert_peer")
    G.add_node("AS44477|Stark", type="asn")
    G.add_edge("185.220.101.47", "AS44477|Stark", relation="hosted-in")
    G.add_node("Frankfurt|Germany", type="geo")
    G.add_edge("185.220.101.47", "Frankfurt|Germany", relation="located-in")
    G.add_node("NameCheap", type="registrar")
    for i in range(3):
        G.add_edge(f"phish{i}.xyz", "NameCheap", relation="registered-by")
    G.add_node("favicon:abc123", type="favicon")
    G.add_edge("phish0.xyz", "favicon:abc123", relation="uses-favicon")
    G.add_edge("phish1.xyz", "favicon:abc123", relation="uses-favicon")
    G.add_node("legit-site.com", type="co_host")
    G.add_edge("185.220.101.47", "legit-site.com", relation="also-hosts")

    candidates = estimate_missing_edges(G, top_k=5, filter_by_type=True)
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Top 5 predictions (impossible type pairs filtered out):")
    for c in candidates[:5]:
        print(f"    {c['explanation']}")
        print(f"      score={c['score']:.5f}  confidence={c['confidence']}")

    # Test 3: hide-and-recover with baselines
    print("\n  TEST 3: Hide-and-recover with baselines")
    evaluate_edge_recovery(G, hide_fraction=0.2, verbose=True)

    # Test 4: generate figures
    print("\n  TEST 4: Generating figures")
    fig_path = figure_edge_predictions(G, candidates[:5])
    print(f"  Edge predictions figure → {fig_path}")