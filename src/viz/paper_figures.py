# src/viz/paper_figures.py
#
# Publication-quality figures for the EVELYN paper.
# Each function = one figure in the paper. Call individually or use
# --demo to generate all with synthetic data for style-checking.
#
# FIGURE INDEX (matches paper order):
#   fig01 — Dataset composition (class balance + graph sizes)
#   fig02 — Example ego-graph (real saved graph)
#   fig03 — Numerical validation (eigendecomp vs scipy + unitarity)
#   fig04 — Blind spot comparison (quantum walk vs GNN)
#   fig05 — Eigenvalue spectrum (phishing vs benign)
#   fig06 — Recursive campaign hypergraph
#   fig07 — Edge estimation predictions (graph + predicted edges overlay)
#   fig08 — Edge recovery comparison (Hits@K bar chart: QW vs baselines)
#   fig09 — Multi-seed recovery stability (box plot across seeds)
#   fig10 — Score distribution (histogram: hidden vs non-hidden edges)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import networkx as nx
from pathlib import Path
import pickle

FIGURES_DIR = Path("results/figures")

COLOR_PHISHING = "#C0392B"
COLOR_BENIGN   = "#2E6E9E"
COLOR_QUANTUM  = "#BA7517"
COLOR_GNN      = "#3F70AC"
COLOR_RANDOM   = "#888780"
COLOR_NEUTRAL  = "#888780"

TYPE_COLORS = {
    "domain": "#D85A30", "ip": "#D4537E", "registrar": "#1D9E75",
    "cert_peer": "#378ADD", "asn": "#7F77DD", "geo": "#5BA88A",
    "co_host": "#C9A227", "favicon": "#9B59B6", "jarm": "#E67E22",
    "subdomain": "#16A085", "unknown": "#888780",
}

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.edgecolor":    "#444444",
    "figure.facecolor":  "white",
    "savefig.facecolor": "white",
})


def _save(fig, name: str) -> str:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png = FIGURES_DIR / f"{name}.png"
    pdf = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {png}")
    print(f"  ✓ {pdf}")
    return str(png)


# ═══════════════════════════════════════════════════════════════════
# FIG 01 — Dataset composition
# ═══════════════════════════════════════════════════════════════════

def figure_dataset_composition(results_csv="data/processed/batch_results.csv"):
    import pandas as pd
    df = pd.read_csv(results_csv)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    counts = df["label"].value_counts().sort_index()
    bars = axes[0].bar(["Benign", "Phishing"],
                       [counts.get(0, 0), counts.get(1, 0)],
                       color=[COLOR_BENIGN, COLOR_PHISHING], width=0.55)
    for bar, val in zip(bars, [counts.get(0, 0), counts.get(1, 0)]):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     str(val), ha="center", fontsize=9)
    axes[0].set_ylabel("Domains")
    axes[0].set_title("(a) Class balance", fontsize=10, loc="left")

    bins = np.linspace(0, max(df["node_count"].max(), 10), 15)
    axes[1].hist(df[df["label"] == 0]["node_count"], bins=bins, alpha=0.6,
                 label="Benign", color=COLOR_BENIGN)
    axes[1].hist(df[df["label"] == 1]["node_count"], bins=bins, alpha=0.6,
                 label="Phishing", color=COLOR_PHISHING)
    axes[1].set_xlabel("Nodes per ego-graph")
    axes[1].set_ylabel("Count")
    axes[1].set_title("(b) Graph size distribution", fontsize=10, loc="left")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _save(fig, "fig01_dataset_composition")


# ═══════════════════════════════════════════════════════════════════
# FIG 03 — Numerical validation
# ═══════════════════════════════════════════════════════════════════

def figure_numerical_validation(eigenvalue_diffs, unitarity_checks):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))

    ts = [t for t, _ in eigenvalue_diffs]
    diffs = [d for _, d in eigenvalue_diffs]
    axes[0].semilogy(ts, diffs, "o-", color=COLOR_QUANTUM, markersize=4)
    axes[0].axhline(1e-9, color=COLOR_NEUTRAL, linestyle="--", linewidth=0.8,
                    label="Tolerance threshold")
    axes[0].set_xlabel("Time parameter t")
    axes[0].set_ylabel("Max |U_ours − U_scipy|")
    axes[0].set_title("(a) Cross-validation vs scipy.expm", fontsize=10, loc="left")
    axes[0].legend(frameon=False, fontsize=8)

    ts2 = sorted(unitarity_checks.keys())
    vals = [1 if unitarity_checks[t] else 0 for t in ts2]
    axes[1].bar([str(t) for t in ts2], vals, color=COLOR_QUANTUM, width=0.5)
    axes[1].set_ylim(0, 1.3)
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["Fails", "Passes"])
    axes[1].set_xlabel("Time parameter t")
    axes[1].set_title("(b) Unitarity check (U·U† = I)", fontsize=10, loc="left")
    fig.tight_layout()
    return _save(fig, "fig03_numerical_validation")


# ═══════════════════════════════════════════════════════════════════
# FIG 04 — Blind spot comparison
# ═══════════════════════════════════════════════════════════════════

def figure_blind_spot_comparison(comparison_results):
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [f"{r['label_a']}\nvs\n{r['label_b']}" for r in comparison_results]
    quantum_dists = [r["quantum_distance"] for r in comparison_results]
    gnn_dists     = [r["gnn_distance"] for r in comparison_results]
    blind_spots   = [r["is_blind_spot"] for r in comparison_results]

    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, quantum_dists, width, label="Quantum walk φ(G)", color=COLOR_QUANTUM)
    ax.bar(x + width/2, gnn_dists, width, label="GraphSAGE baseline", color=COLOR_GNN)

    for i, is_blind in enumerate(blind_spots):
        if is_blind:
            ax.annotate("blind\nspot", xy=(x[i], max(quantum_dists[i], 0.02) + 0.015),
                        fontsize=7, ha="center", color=COLOR_PHISHING, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Pairwise distance (cosine)")
    ax.set_title("Structural distinguishability: quantum walk vs. GNN-style aggregation")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _save(fig, "fig04_blind_spot_comparison")


# ═══════════════════════════════════════════════════════════════════
# FIG 05 — Eigenvalue spectrum comparison
# ═══════════════════════════════════════════════════════════════════

def figure_spectrum_comparison(phishing_eigenvalues_list, benign_eigenvalues_list):
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, eigs in enumerate(phishing_eigenvalues_list):
        ax.scatter(eigs, [i] * len(eigs), color=COLOR_PHISHING, alpha=0.6, s=18,
                   label="Phishing" if i == 0 else None)
    offset = len(phishing_eigenvalues_list) + 1
    for i, eigs in enumerate(benign_eigenvalues_list):
        ax.scatter(eigs, [i + offset] * len(eigs), color=COLOR_BENIGN, alpha=0.6, s=18,
                   label="Benign" if i == 0 else None)
    ax.set_xlabel("Eigenvalue of H = −A")
    ax.set_yticks([])
    ax.set_ylabel("Individual graphs (stacked)")
    ax.set_title("Hamiltonian eigenvalue spectra: phishing vs. benign infrastructure")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _save(fig, "fig05_spectrum_comparison")


# ═══════════════════════════════════════════════════════════════════
# FIG 07 — Edge estimation predictions (graph + dashed overlay)
# ═══════════════════════════════════════════════════════════════════

def figure_edge_predictions(G: nx.Graph, predictions: list,
                             max_show: int = 5, title: str = None) -> str:
    """
    Graph visualization with predicted missing edges overlaid as
    dashed red lines. Each predicted edge is labeled with its
    relation type. Nodes colored by type, sized by degree.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    try:
        pos = nx.kamada_kawai_layout(G)
    except Exception:
        pos = nx.spring_layout(G, seed=7, k=0.6)

    # Existing edges — light grey, recede visually
    nx.draw_networkx_edges(G, pos, edge_color="#cccccc", width=0.8, alpha=0.5, ax=ax)

    # Nodes by type
    types_present = sorted(set(d.get("type", "unknown") for _, d in G.nodes(data=True)))
    for t in types_present:
        nodes = [n for n, d in G.nodes(data=True) if d.get("type") == t]
        sizes = [150 + 50 * G.degree(n) for n in nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes,
                                node_color=TYPE_COLORS.get(t, "#888780"),
                                node_size=sizes, alpha=0.9, ax=ax,
                                edgecolors="white", linewidths=0.8, label=t)

    # Predicted edges — dashed red with relation labels
    shown = predictions[:max_show]
    for pred in shown:
        a, b = pred["node_a"], pred["node_b"]
        if a in pos and b in pos:
            ax.annotate("", xy=pos[b], xytext=pos[a],
                        arrowprops=dict(arrowstyle="-", color=COLOR_PHISHING,
                                        linestyle="dashed", linewidth=2.0, alpha=0.7))
            mx = (pos[a][0] + pos[b][0]) / 2
            my = (pos[a][1] + pos[b][1]) / 2
            ax.text(mx, my, pred.get("relation", "?"), fontsize=7, color=COLOR_PHISHING,
                    ha="center", bbox=dict(boxstyle="round,pad=0.15",
                    fc="white", ec=COLOR_PHISHING, alpha=0.85))

    # Label root domain
    root = G.graph.get("domain") if hasattr(G, "graph") else None
    if root and root in pos:
        ax.annotate(root, pos[root], fontsize=9, fontweight="bold",
                    ha="center", xytext=(0, 14), textcoords="offset points")

    domain = G.graph.get("domain", "unknown").replace(".", "_") if hasattr(G, "graph") else "graph"
    ax.set_title(title or f"{domain} — predicted missing infrastructure links (dashed red)\n"
                 f"Top {len(shown)} predictions by quantum walk score",
                 fontsize=11)
    ax.legend(loc="upper left", fontsize=7.5, frameon=False, markerscale=0.6,
              bbox_to_anchor=(1.0, 1.0))
    ax.axis("off")
    fig.tight_layout()
    return _save(fig, f"fig07_edge_predictions_{domain}")


# ═══════════════════════════════════════════════════════════════════
# FIG 08 — Edge recovery: Hits@K bar chart (QW vs baselines)
# ═══════════════════════════════════════════════════════════════════

def figure_recovery_comparison(eval_result: dict, title: str = None) -> str:
    """
    Grouped bar chart: Hits@K for quantum walk, common neighbors, random.
    MRR shown in legend. This is THE results figure for the pivot section.
    """
    qw   = eval_result["quantum_walk"]
    cn   = eval_result["common_neighbors"]
    rand = eval_result["random"]

    k_values = sorted(qw["hits"].keys())
    x = np.arange(len(k_values))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width, [qw["hits"][k] for k in k_values], width,
           label=f"Quantum walk (MRR={qw['mrr']:.3f})", color=COLOR_QUANTUM)
    ax.bar(x, [cn["hits"][k] for k in k_values], width,
           label=f"Common neighbors (MRR={cn['mrr']:.3f})", color=COLOR_GNN)
    ax.bar(x + width, [rand["hits"][k] for k in k_values], width,
           label=f"Random (MRR={rand['mrr']:.3f})", color=COLOR_RANDOM)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Hits@{k}" for k in k_values], fontsize=10)
    ax.set_ylabel("Fraction of hidden edges recovered", fontsize=10)
    ax.set_title(title or f"Edge recovery: quantum walk vs baselines "
                 f"({eval_result['n_hidden']} edges hidden from "
                 f"{eval_result['graph_edges']} total)", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_ylim(0, 1.1)

    fig.tight_layout()
    return _save(fig, "fig08_edge_recovery_comparison")


# ═══════════════════════════════════════════════════════════════════
# FIG 09 — Multi-seed recovery stability (box plot)
# ═══════════════════════════════════════════════════════════════════

def figure_recovery_stability(G: nx.Graph, n_seeds: int = 10,
                                hide_fraction: float = 0.2) -> str:
    """
    Runs hide-and-recover across N different random seeds and plots
    the MRR distribution as a box plot. Shows the method is stable,
    not just lucky on one particular random split.
    """
    from src.quantum.estimate_edges import evaluate_edge_recovery

    qw_mrrs, cn_mrrs, rand_mrrs = [], [], []
    for seed in range(n_seeds):
        result = evaluate_edge_recovery(G, hide_fraction=hide_fraction,
                                         seed=seed, verbose=False)
        if "error" in result:
            continue
        qw_mrrs.append(result["quantum_walk"]["mrr"])
        cn_mrrs.append(result["common_neighbors"]["mrr"])
        rand_mrrs.append(result["random"]["mrr"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = [qw_mrrs, cn_mrrs, rand_mrrs]
    labels = ["Quantum walk", "Common\nneighbors", "Random"]
    colors = [COLOR_QUANTUM, COLOR_GNN, COLOR_RANDOM]

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    # Overlay individual points
    for i, (d, c) in enumerate(zip(data, colors)):
        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(d))
        ax.scatter(np.full(len(d), i + 1) + jitter, d, color=c,
                   alpha=0.5, s=30, zorder=5, edgecolors="white", linewidths=0.5)

    ax.set_ylabel("MRR (Mean Reciprocal Rank)", fontsize=10)
    ax.set_title(f"Recovery stability across {n_seeds} random edge-hiding splits\n"
                 f"({hide_fraction:.0%} of edges hidden each time)", fontsize=11)
    fig.tight_layout()
    return _save(fig, "fig09_recovery_stability")


# ═══════════════════════════════════════════════════════════════════
# FIG 10 — Score distribution: hidden vs non-hidden
# ═══════════════════════════════════════════════════════════════════

def figure_score_distribution(G: nx.Graph, hide_fraction: float = 0.2,
                                seed: int = 0) -> str:
    """
    Histogram showing how the quantum walk scores distribute for
    truly-hidden edges vs. all other non-edges. Good separation =
    the method is assigning HIGH scores to real edges and LOW scores
    to non-edges. Overlapping distributions = the method can't tell
    them apart reliably.
    """
    from src.quantum.estimate_edges import estimate_missing_edges

    rng = np.random.default_rng(seed)
    edges = list(G.edges())
    n_hide = max(1, int(len(edges) * hide_fraction))

    # Hide edges while keeping connectivity
    G_damaged = G.copy()
    hidden_edges = []
    shuffled = rng.permutation(len(edges))
    for idx in shuffled:
        if len(hidden_edges) >= n_hide:
            break
        u, v = edges[idx]
        G_damaged.remove_edge(u, v)
        if nx.is_connected(G_damaged):
            hidden_edges.append((u, v))
        else:
            G_damaged.add_edge(u, v, **G.edges[u, v])

    hidden_set = set(frozenset([u, v]) for u, v in hidden_edges)

    # Score all non-edges
    all_candidates = estimate_missing_edges(G_damaged, top_k=10**6, filter_by_type=True)

    hidden_scores = []
    other_scores = []
    for c in all_candidates:
        pair = frozenset([c["node_a"], c["node_b"]])
        if pair in hidden_set:
            hidden_scores.append(c["score"])
        else:
            other_scores.append(c["score"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, max(max(hidden_scores, default=0),
                              max(other_scores, default=0)) * 1.1, 30)

    ax.hist(other_scores, bins=bins, alpha=0.6, color=COLOR_GNN,
            label=f"Non-edges ({len(other_scores)})", density=True)
    ax.hist(hidden_scores, bins=bins, alpha=0.8, color=COLOR_PHISHING,
            label=f"Hidden real edges ({len(hidden_scores)})", density=True)

    ax.set_xlabel("Quantum walk score (multi-t averaged)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(f"Score distribution: hidden edges vs non-edges\n"
                 f"Good separation = method reliably identifies missing links",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return _save(fig, "fig10_score_distribution")


# ═══════════════════════════════════════════════════════════════════
# GENERATE ALL — demo mode or with real data
# ═══════════════════════════════════════════════════════════════════

def generate_all_edge_estimation_figures(G: nx.Graph = None, verbose: bool = True):
    """
    Generates all 4 edge-estimation figures (fig07-10) from one graph.
    If no graph provided, builds a synthetic campaign graph for demo.
    """
    if G is None:
        G = _build_demo_campaign_graph()
        if verbose:
            print("  Using synthetic campaign graph for demo")

    if verbose:
        print(f"\n  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    from src.quantum.estimate_edges import estimate_missing_edges, evaluate_edge_recovery

    # Fig 07 — predictions overlay
    predictions = estimate_missing_edges(G, top_k=10, filter_by_type=True)
    fig07 = figure_edge_predictions(G, predictions, max_show=5)

    # Fig 08 — recovery comparison
    eval_result = evaluate_edge_recovery(G, hide_fraction=0.2, verbose=verbose)
    fig08 = figure_recovery_comparison(eval_result)

    # Fig 09 — stability across seeds
    fig09 = figure_recovery_stability(G, n_seeds=10, hide_fraction=0.2)

    # Fig 10 — score distribution
    fig10 = figure_score_distribution(G, hide_fraction=0.2)

    paths = {"fig07": fig07, "fig08": fig08, "fig09": fig09, "fig10": fig10}
    if verbose:
        print(f"\n  ✓ All 4 edge estimation figures generated")
    return paths


def _build_demo_campaign_graph():
    """Builds a realistic 19-node phishing campaign graph for demos."""
    G = nx.Graph()
    domains = [f"phish{i}.xyz" for i in range(5)]
    for d in domains:
        G.add_node(d, type="domain")
    ips = ["185.220.101.47", "185.220.101.48"]
    for ip in ips:
        G.add_node(ip, type="ip")
    for d in domains[:3]:
        G.add_edge(d, ips[0], relation="resolves-to")
    for d in domains[3:]:
        G.add_edge(d, ips[1], relation="resolves-to")
    G.add_node("wildcard_cert", type="cert_peer")
    for d in domains[:4]:
        G.add_edge(d, "wildcard_cert", relation="shares-cert")
    G.add_node("separate_cert", type="cert_peer")
    G.add_edge(domains[4], "separate_cert", relation="shares-cert")
    G.add_node("NameCheap", type="registrar")
    G.add_node("Porkbun", type="registrar")
    for d in domains[:3]:
        G.add_edge(d, "NameCheap", relation="registered-by")
    for d in domains[3:]:
        G.add_edge(d, "Porkbun", relation="registered-by")
    G.add_node("favicon:a1b2c3", type="favicon")
    for d in domains[:3]:
        G.add_edge(d, "favicon:a1b2c3", relation="uses-favicon")
    G.add_node("AS44477|Stark", type="asn")
    G.add_edge(ips[0], "AS44477|Stark", relation="hosted-in")
    G.add_edge(ips[1], "AS44477|Stark", relation="hosted-in")
    G.add_node("Frankfurt|DE", type="geo")
    G.add_node("Amsterdam|NL", type="geo")
    G.add_edge(ips[0], "Frankfurt|DE", relation="located-in")
    G.add_edge(ips[1], "Amsterdam|NL", relation="located-in")
    for i, co in enumerate(["legit1.com", "legit2.com", "legit3.org"]):
        G.add_node(co, type="co_host")
        G.add_edge(ips[i % 2], co, relation="also-hosts")
    G.add_node("admin.phish0.xyz", type="subdomain")
    G.add_edge(domains[0], "admin.phish0.xyz", relation="has-subdomain")
    G.graph["domain"] = "phish0.xyz"
    return G


# ── CLI ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("\n" + "=" * 58)
    print("  EVELYN — paper_figures()")
    print("=" * 58)

    if "--edge-demo" in sys.argv:
        print("\n  Generating all edge estimation figures (demo mode)...\n")
        generate_all_edge_estimation_figures(verbose=True)

    elif "--blind-spot-demo" in sys.argv:
        print("\n  Generating blind spot + numerical validation figures...\n")
        from src.quantum.compare_blind_spots import run_synthetic_blind_spot_suite
        results = run_synthetic_blind_spot_suite()
        figure_blind_spot_comparison(results)
        figure_numerical_validation(
            eigenvalue_diffs=[(0.5, 1e-15), (1.0, 8.4e-16), (5.0, 3e-14),
                              (10.0, 5e-13), (100.0, 2e-10)],
            unitarity_checks={0.5: True, 1.0: True, 5.0: True,
                              10.0: True, 100.0: True, 1000.0: True},
        )

    elif "--all-demo" in sys.argv:
        print("\n  Generating ALL figures (demo mode)...\n")
        from src.quantum.compare_blind_spots import run_synthetic_blind_spot_suite
        results = run_synthetic_blind_spot_suite()
        figure_blind_spot_comparison(results)
        figure_numerical_validation(
            eigenvalue_diffs=[(0.5, 1e-15), (1.0, 8.4e-16), (5.0, 3e-14),
                              (10.0, 5e-13), (100.0, 2e-10)],
            unitarity_checks={0.5: True, 1.0: True, 5.0: True,
                              10.0: True, 100.0: True, 1000.0: True},
        )
        generate_all_edge_estimation_figures(verbose=True)

    else:
        print("\n  Usage:")
        print("    python -m src.viz.paper_figures --edge-demo")
        print("    python -m src.viz.paper_figures --blind-spot-demo")
        print("    python -m src.viz.paper_figures --all-demo")
        print("\n  Or import individual functions:")
        print("    from src.viz.paper_figures import figure_edge_predictions")
        print("    from src.viz.paper_figures import generate_all_edge_estimation_figures")