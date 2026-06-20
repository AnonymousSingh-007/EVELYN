# src/viz/paper_figures.py
#
# PURPOSE: Generate publication-quality (Q1 venue style) figures
# directly from EVELYN's pipeline outputs, one function per figure,
# called incrementally as each stage produces new data. This is NOT
# a one-shot "make all figures at the end" script — call individual
# functions as you go, so you catch visualization/data issues early.
#
# STYLE CONVENTIONS (consistent across every figure, IEEE/USENIX style):
#   - White background, no gridlines unless they aid reading values
#   - Serif font for figure text (matches LaTeX paper body text)
#   - Consistent color mapping: phishing=red-ish, benign=blue-ish,
#     quantum-walk=amber, GNN-baseline=steel-blue (matches your
#     earlier Stage-0 widget palette for visual continuity across
#     your whole research narrative)
#   - All figures saved as BOTH .png (300dpi, for drafts/slides) and
#     .pdf (vector, for the actual LaTeX submission) — venues require
#     vector graphics for camera-ready figures.

import matplotlib
matplotlib.use("Agg")   # no display needed — saving directly to file
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import networkx as nx
from pathlib import Path
import pickle
import json

FIGURES_DIR = Path("results/figures")

# Consistent palette across the whole paper
COLOR_PHISHING = "#C0392B"
COLOR_BENIGN   = "#2E5C8A"
COLOR_QUANTUM  = "#BA7517"
COLOR_GNN      = "#3F70AC"
COLOR_NEUTRAL  = "#888780"

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.edgecolor":   "#444444",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def _save(fig, name: str) -> None:
    """Saves a figure as both PNG (300dpi) and PDF (vector) — the dual
    format every Q1 venue submission checklist asks for."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURES_DIR / f"{name}.png"
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {png_path}")
    print(f"  ✓ Saved {pdf_path}")


# ═══════════════════════════════════════════════════════════════════
# STAGE 1 FIGURE — Dataset composition & graph size distribution
# ═══════════════════════════════════════════════════════════════════

def figure_dataset_composition(results_csv: str = "data/processed/batch_results.csv") -> None:
    """
    Figure 1 in most infrastructure-graph papers: shows your dataset's
    basic composition — class balance, and the node-count distribution
    that motivated your truncation/recursion design decisions. This is
    almost always Figure 1 or 2 in the paper, establishing "here is the
    data we're working with" before any method results appear.
    """
    import pandas as pd
    df = pd.read_csv(results_csv)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # Left panel: class balance bar chart
    ax = axes[0]
    counts = df["label"].value_counts().sort_index()
    labels = ["Benign", "Phishing"]
    colors = [COLOR_BENIGN, COLOR_PHISHING]
    bars = ax.bar(labels, [counts.get(0, 0), counts.get(1, 0)], color=colors, width=0.55)
    for bar, val in zip(bars, [counts.get(0, 0), counts.get(1, 0)]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(val), ha="center", fontsize=9)
    ax.set_ylabel("Domains")
    ax.set_title("(a) Class balance", fontsize=10, loc="left")

    # Right panel: node-count distribution, split by class — this is
    # the figure that JUSTIFIES your node-budget/truncation design in
    # the methods section, by showing the real size distribution.
    ax = axes[1]
    phishing_nodes = df[df["label"] == 1]["node_count"]
    benign_nodes   = df[df["label"] == 0]["node_count"]
    bins = np.linspace(0, max(df["node_count"].max(), 10), 15)
    ax.hist(benign_nodes, bins=bins, alpha=0.6, label="Benign", color=COLOR_BENIGN)
    ax.hist(phishing_nodes, bins=bins, alpha=0.6, label="Phishing", color=COLOR_PHISHING)
    ax.set_xlabel("Nodes per ego-graph")
    ax.set_ylabel("Count")
    ax.set_title("(b) Graph size distribution", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    _save(fig, "fig01_dataset_composition")


# ═══════════════════════════════════════════════════════════════════
# STAGE 1 FIGURE — Example ego-graph visualization
# ═══════════════════════════════════════════════════════════════════

def figure_example_graph(graph_pkl_path: str, title: str = None) -> None:
    """
    Figure showing one real saved graph from data/graphs/, node-type
    colored, exactly like your Stage 0 hypergraph diagram but using
    REAL collected data instead of an illustrative mockup. Papers
    almost always include 1-2 of these as qualitative examples
    alongside the quantitative results.
    """
    with open(graph_pkl_path, "rb") as f:
        G = pickle.load(f)

    type_colors = {
        "domain":    "#D85A30",
        "ip":        "#D4537E",
        "registrar": "#1D9E75",
        "cert_peer": "#378ADD",
        "asn":       "#7F77DD",
        "geo":       "#5BA88A",
        "co_host":   "#C9A227",
    }

    fig, ax = plt.subplots(figsize=(6, 5.5))
    pos = nx.spring_layout(G, seed=7, k=0.6)

    for node_type, color in type_colors.items():
        nodes_of_type = [n for n, d in G.nodes(data=True) if d.get("type") == node_type]
        if nodes_of_type:
            nx.draw_networkx_nodes(G, pos, nodelist=nodes_of_type, node_color=color,
                                    node_size=220, ax=ax, label=node_type, alpha=0.9)

    nx.draw_networkx_edges(G, pos, edge_color="#999999", width=0.8, alpha=0.6, ax=ax)

    # Only label the root domain to avoid visual clutter
    domain = G.graph.get("domain")
    if domain and domain in pos:
        ax.annotate(domain, pos[domain], fontsize=7, ha="center",
                    xytext=(0, 12), textcoords="offset points")

    ax.legend(loc="upper left", fontsize=7, frameon=False, markerscale=0.7)
    ax.set_title(title or f"Ego-graph: {domain}", fontsize=10)
    ax.axis("off")

    fig.tight_layout()
    safe_name = (domain or "graph").replace(".", "_")
    _save(fig, f"fig02_example_graph_{safe_name}")


# ═══════════════════════════════════════════════════════════════════
# STAGE 2 FIGURE — Numerical validation summary (hermiticity, unitarity)
# ═══════════════════════════════════════════════════════════════════

def figure_numerical_validation(eigenvalue_diffs: list, unitarity_checks: dict) -> None:
    """
    A "trust the method" figure for the methods section — shows that
    your eigendecomposition-vs-scipy cross-validation stays near
    machine epsilon across many t values, and that unitarity holds
    even at large t. Reviewers respect seeing this proven visually,
    not just asserted in prose.

    Parameters:
        eigenvalue_diffs : list of (t, max_diff) tuples from running
                           verify_against_scipy() at multiple t values
        unitarity_checks : dict {t: bool} from quantum_walk()'s is_unitary
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))

    ax = axes[0]
    ts = [t for t, _ in eigenvalue_diffs]
    diffs = [d for _, d in eigenvalue_diffs]
    ax.semilogy(ts, diffs, "o-", color=COLOR_QUANTUM, markersize=4)
    ax.axhline(1e-9, color=COLOR_NEUTRAL, linestyle="--", linewidth=0.8,
              label="Tolerance threshold")
    ax.set_xlabel("Time parameter t")
    ax.set_ylabel("Max |U_ours − U_scipy|")
    ax.set_title("(a) Cross-validation vs scipy.expm", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ts2 = sorted(unitarity_checks.keys())
    vals = [1 if unitarity_checks[t] else 0 for t in ts2]
    ax.bar([str(t) for t in ts2], vals, color=COLOR_QUANTUM, width=0.5)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Fails", "Passes"])
    ax.set_xlabel("Time parameter t")
    ax.set_title("(b) Unitarity check (U·U† = I)", fontsize=10, loc="left")

    fig.tight_layout()
    _save(fig, "fig03_numerical_validation")


# ═══════════════════════════════════════════════════════════════════
# STAGE 2 FIGURE — Blind spot comparison (quantum walk vs GNN baseline)
# ═══════════════════════════════════════════════════════════════════

def figure_blind_spot_comparison(comparison_results: list) -> None:
    """
    THE signature figure of your "beyond the concept" contribution.
    Side-by-side bar chart: for each tested graph pair, shows the
    quantum-walk distance vs the GNN-baseline distance. Cases where
    quantum walk says "different" but GNN says "same" are highlighted
    — this IS your blind-spot evidence, visualized.

    Parameters:
        comparison_results : list of dicts, exactly as returned by
                             compare_pair() in compare_blind_spots.py
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    labels = [f"{r['label_a']}\nvs\n{r['label_b']}" for r in comparison_results]
    quantum_dists = [r["quantum_distance"] for r in comparison_results]
    gnn_dists     = [r["gnn_distance"] for r in comparison_results]
    blind_spots   = [r["is_blind_spot"] for r in comparison_results]

    x = np.arange(len(labels))
    width = 0.35

    bars_q = ax.bar(x - width/2, quantum_dists, width, label="Quantum walk φ(G)",
                    color=COLOR_QUANTUM)
    bars_g = ax.bar(x + width/2, gnn_dists, width, label="GraphSAGE baseline",
                    color=COLOR_GNN)

    # Highlight blind-spot cases with a marker above the bar pair
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
    _save(fig, "fig04_blind_spot_comparison")


# ═══════════════════════════════════════════════════════════════════
# STAGE 2 FIGURE — Eigenvalue spectrum comparison (phishing vs benign)
# ═══════════════════════════════════════════════════════════════════

def figure_spectrum_comparison(phishing_eigenvalues_list: list,
                                benign_eigenvalues_list: list) -> None:
    """
    Overlays the eigenvalue spectra of multiple phishing graphs against
    multiple benign graphs. If phishing infrastructure really does have
    a distinct "shape," this should show visually separated spectral
    distributions — your first real qualitative evidence of class
    separability, before any clustering numbers exist.

    Parameters:
        phishing_eigenvalues_list : list of np.ndarray (one per phishing graph)
        benign_eigenvalues_list   : list of np.ndarray (one per benign graph)
    """
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
    _save(fig, "fig05_spectrum_comparison")


# ── SELF-TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("\n" + "=" * 58)
    print("  EVELYN — paper_figures()")
    print("=" * 58)

    if "--demo" in sys.argv:
        # Demo mode: generates all figures using synthetic/toy data so
        # you can see the STYLE and verify matplotlib/output paths work,
        # before wiring in your real saved graphs and results CSV.
        print("\n  Demo mode: generating figures from synthetic data\n")

        # Fake a results CSV in memory for figure 1
        import pandas as pd
        demo_df = pd.DataFrame({
            "label": [1]*12 + [0]*15,
            "node_count": list(np.random.randint(12, 28, 12)) + list(np.random.randint(2, 90, 15)),
        })
        demo_path = Path("data/processed/_demo_results.csv")
        demo_path.parent.mkdir(parents=True, exist_ok=True)
        demo_df.to_csv(demo_path, index=False)
        figure_dataset_composition(str(demo_path))

        # Demo numerical validation
        figure_numerical_validation(
            eigenvalue_diffs=[(0.5, 1e-15), (1.0, 8.4e-16), (5.0, 3e-14), (10.0, 5e-13), (100.0, 2e-10)],
            unitarity_checks={0.5: True, 1.0: True, 5.0: True, 10.0: True, 100.0: True, 1000.0: True},
        )

        # Demo blind spot comparison (using your actual real results!)
        from src.quantum.compare_blind_spots import run_synthetic_blind_spot_suite
        results = run_synthetic_blind_spot_suite()
        figure_blind_spot_comparison(results)

        # Demo spectrum comparison
        rng = np.random.default_rng(1)
        phishing_eigs = [rng.normal(0, 1.5, 15) for _ in range(8)]
        benign_eigs   = [rng.normal(0, 3.0, 25) for _ in range(8)]
        figure_spectrum_comparison(phishing_eigs, benign_eigs)

        print(f"\n  All demo figures saved to {FIGURES_DIR}/")
        print(f"  Once you have real data, call these functions directly with")
        print(f"  your actual CSV/graph paths instead of --demo mode.")
    else:
        print("\n  Usage:")
        print("    python -m src.viz.paper_figures --demo   (generate style-check demo figures)")
        print("\n  Or import individual functions into your own scripts:")
        print("    from src.viz.paper_figures import figure_dataset_composition")
        print("    figure_dataset_composition('data/processed/batch_results.csv')")