# src/viz/stage_figures.py
#
# PURPOSE: Given a single domain, generate FOUR connected, paper-quality
# 2D figures tracing it through the entire EVELYN pipeline:
#
#   Figure A — Hypergraph        : the actual node-link graph, clean
#                                   2D layout, labeled edges, no 3D
#                                   depth ambiguity to fight with.
#   Figure B — Hamiltonian heatmap: the literal H = -A matrix, as a
#                                   color-coded grid with node labels
#                                   on both axes — makes the abstract
#                                   "matrix" concrete and readable.
#   Figure C — Eigenvalue spectrum: the raw eigenvalues of H, plotted
#                                   on a line — the "ingredients" that
#                                   feed the fingerprint.
#   Figure D — Fingerprint bars   : the final phi(G) vector, as a
#                                   bar chart — the actual number
#                                   sequence used for clustering/comparison.
#
# WHY 2D INSTEAD OF FLATTENING THE 3D SCENE:
#   A 3D scatter plot rendered to a static JPEG loses exactly the two
#   things that made it useful (rotation or hover) and gains nothing
#   in return — depth gets misjudged, labels overlap unpredictably
#   depending on camera angle, and edges crossing "behind" nodes read
#   as crossing "through" them. A purpose-built 2D layout for each
#   STAGE of the pipeline is more legible, easier to caption precisely,
#   and matches how these are actually presented in published work —
#   one focused figure per concept, not one crowded scene.
#
#   The interactive 3D HTML (graph_3d.py) remains the right tool for
#   live, exploratory hover-based investigation. This module is the
#   one that produces what actually goes in the paper / report.

from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.pipeline.build_graph import build_graph
from src.quantum.hamiltonian import build_hamiltonian
from src.quantum.fingerprint import extract_fingerprint


FIGURES_DIR = Path("results/figures")

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.edgecolor":    "#444444",
    "figure.facecolor":  "white",
    "savefig.facecolor": "white",
})

TYPE_COLORS = {
    "domain":    "#D85A30",
    "ip":        "#D4537E",
    "registrar": "#1D9E75",
    "cert_peer": "#378ADD",
    "asn":       "#7F77DD",
    "geo":       "#5BA88A",
    "co_host":   "#C9A227",
    "favicon":   "#9B59B6",
    "jarm":      "#E67E22",
    "subdomain": "#16A085",
    "unknown":   "#888780",
}

EDGE_PLAIN_LABEL = {
    "resolves-to":        "resolves to",
    "registered-by":      "registered by",
    "shares-cert":        "shares cert with",
    "hosted-in":          "hosted on",
    "located-in":         "located in",
    "also-hosts":         "shares server with",
    "uses-favicon":       "uses icon",
    "has-subdomain":      "subdomain of",
    "server-fingerprint": "server fingerprint",
}


def generate_all_stage_figures(domain_or_url: str, label: int = None,
                                verbose: bool = True) -> dict:
    """
    Runs the full pipeline once, then generates all four stage figures
    from the SAME underlying graph and Hamiltonian — so every figure
    in the set is guaranteed to describe the exact same investigation,
    not four separately-run, possibly-inconsistent snapshots.

    Returns dict of {"graph", "hamiltonian", "spectrum", "fingerprint"}
    mapped to saved PNG paths (PDFs are saved alongside automatically).
    """
    if verbose:
        print(f"\n  [stage_figures] Running pipeline for: {domain_or_url}")

    G = build_graph(domain_or_url, label=label, verbose=verbose, save=True)

    if G.number_of_nodes() == 0:
        if verbose:
            print("  ⚠ Empty graph — nothing to visualize")
        return {}

    domain = G.graph.get("domain", "unknown").replace(".", "_")
    ham = build_hamiltonian(G, weighted=True, variant="adjacency")
    fp = extract_fingerprint(G, method="combined")

    paths = {}
    paths["graph"]       = _figure_hypergraph(G, domain, label)
    paths["hamiltonian"] = _figure_hamiltonian_heatmap(ham, domain, label)
    paths["spectrum"]    = _figure_eigenvalue_spectrum(ham, domain, label)
    paths["fingerprint"] = _figure_fingerprint_bars(fp, domain, label)

    if verbose:
        print(f"\n  ✓ Generated 4 stage figures for {domain}:")
        for name, p in paths.items():
            print(f"      {name}: {p}")

    return paths


def _save(fig, name: str) -> str:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURES_DIR / f"{name}.png"
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return str(png_path)


# ═══════════════════════════════════════════════════════════════════
# FIGURE A — Hypergraph, clean 2D, labeled edges
# ═══════════════════════════════════════════════════════════════════

def _figure_hypergraph(G: nx.Graph, domain: str, label) -> str:
    """
    A genuinely readable 2D node-link diagram. Node color = type
    (legend on the side). Edge labels sit at the midpoint of each
    edge, deduplicated per (source, relation) so a hub node doesn't
    repeat the same label many times.
    """
    fig, ax = plt.subplots(figsize=(9, 7.5))

    try:
        pos = nx.kamada_kawai_layout(G)
    except Exception:
        pos = nx.spring_layout(G, seed=7, k=0.6)

    nx.draw_networkx_edges(G, pos, edge_color="#aaaaaa", width=0.9,
                            alpha=0.7, ax=ax)

    types_present = sorted(set(d.get("type", "unknown") for _, d in G.nodes(data=True)))
    for t in types_present:
        nodes_of_type = [n for n, d in G.nodes(data=True) if d.get("type") == t]
        sizes = [180 + 60 * G.degree(n) for n in nodes_of_type]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes_of_type,
                                node_color=TYPE_COLORS.get(t, "#888780"),
                                node_size=sizes, alpha=0.9, ax=ax,
                                edgecolors="white", linewidths=0.8, label=t)

    root = G.graph.get("domain")
    if root in pos:
        ax.annotate(root, pos[root], fontsize=9, fontweight="bold", ha="center",
                    xytext=(0, 14), textcoords="offset points")

    by_source_relation = {}
    for u, v, data in G.edges(data=True):
        relation = data.get("relation", "connected to")
        by_source_relation.setdefault((u, relation), []).append(v)

    for (u, relation), targets in by_source_relation.items():
        if u not in pos or targets[0] not in pos:
            continue
        v = targets[0]
        mx = (pos[u][0] + pos[v][0]) / 2
        my = (pos[u][1] + pos[v][1]) / 2
        plain = EDGE_PLAIN_LABEL.get(relation, relation.replace("-", " "))
        text = plain if len(targets) == 1 else f"{plain} (+{len(targets)-1})"
        ax.text(mx, my, text, fontsize=6.5, color="#555555", ha="center",
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

    label_str = {1: "PHISHING", 0: "BENIGN", None: "UNKNOWN"}.get(label, "UNKNOWN")
    ax.set_title(f"{domain}  —  infrastructure hypergraph  [{label_str}]\n"
                 f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges",
                 fontsize=12)
    ax.legend(loc="upper left", fontsize=7.5, frameon=False, markerscale=0.6,
              bbox_to_anchor=(1.0, 1.0))
    ax.axis("off")

    fig.tight_layout()
    return _save(fig, f"stageA_hypergraph_{domain}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE B — Hamiltonian matrix heatmap
# ═══════════════════════════════════════════════════════════════════

def _figure_hamiltonian_heatmap(ham: dict, domain: str, label) -> str:
    """
    The Hamiltonian H = -A as an actual color-coded grid, with real
    node names on both axes (truncated if long).
    """
    H = ham["H"]
    node_order = ham["node_order"]
    n = ham["n_nodes"]

    show_labels = n <= 30

    fig_size = max(6, min(0.35 * n + 2, 12))
    fig, ax = plt.subplots(figsize=(fig_size + 1, fig_size))

    im = ax.imshow(H, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Matrix value", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    if show_labels:
        short_names = [nm[:18] + "…" if len(nm) > 18 else nm for nm in node_order]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(short_names, rotation=90, fontsize=6.5)
        ax.set_yticklabels(short_names, fontsize=6.5)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"{n} nodes (labels hidden — too many to fit)", fontsize=9)

    label_str = {1: "PHISHING", 0: "BENIGN", None: "UNKNOWN"}.get(label, "UNKNOWN")
    ax.set_title(f"{domain}  —  Hamiltonian  H = −A  [{label_str}]\n"
                 f"Blue cells = connected nodes (−1), white = no connection (0)",
                 fontsize=11)

    fig.tight_layout()
    return _save(fig, f"stageB_hamiltonian_{domain}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE C — Eigenvalue spectrum
# ═══════════════════════════════════════════════════════════════════

def _figure_eigenvalue_spectrum(ham: dict, domain: str, label) -> str:
    """
    The raw eigenvalues of H, sorted and plotted as a stem plot —
    the literal "ingredients" U(t) is built from.
    """
    eigenvalues = np.sort(ham["eigenvalues"])
    n = len(eigenvalues)

    fig, ax = plt.subplots(figsize=(8, 4))
    markerline, stemlines, baseline = ax.stem(range(n), eigenvalues, basefmt=" ")
    markerline.set_color("#BA7517")
    markerline.set_markersize(5)
    stemlines.set_color("#BA7517")
    stemlines.set_linewidth(1)
    ax.axhline(0, color="#888780", linewidth=0.7, linestyle="--")

    ax.set_xlabel("Eigenvalue index (sorted ascending)", fontsize=10)
    ax.set_ylabel("Eigenvalue of H", fontsize=10)

    label_str = {1: "PHISHING", 0: "BENIGN", None: "UNKNOWN"}.get(label, "UNKNOWN")
    ax.set_title(f"{domain}  —  Hamiltonian eigenvalue spectrum  [{label_str}]\n"
                 f"These {n} values are the raw ingredients of the quantum walk",
                 fontsize=11)

    fig.tight_layout()
    return _save(fig, f"stageC_spectrum_{domain}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE D — Fingerprint phi(G) bar chart
# ═══════════════════════════════════════════════════════════════════

def _figure_fingerprint_bars(fp: dict, domain: str, label) -> str:
    """
    The final phi(G) feature vector as a bar chart, with a visual
    divider between its two component halves (spectral-shape portion
    vs return-probability portion) when method="combined".
    """
    phi = fp["phi"]
    n = len(phi)

    eigen_hist_len = 20 if n > 20 else n // 2
    split = eigen_hist_len if fp["method"] == "combined" else n

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(n)
    colors = ["#BA7517" if i < split else "#3F70AC" for i in range(n)]
    ax.bar(x, phi, color=colors, width=0.8)

    if fp["method"] == "combined" and split < n and len(phi) > 0:
        ymax = max(phi) if len(phi) else 1
        ax.axvline(split - 0.5, color="#888780", linewidth=1, linestyle="--")
        ax.text(split / 2, ymax * 1.05, "spectral shape", ha="center", fontsize=8, color="#BA7517")
        ax.text(split + (n - split) / 2, ymax * 1.05, "return probability", ha="center", fontsize=8, color="#3F70AC")

    ax.set_xlabel("Fingerprint dimension", fontsize=10)
    ax.set_ylabel("Value", fontsize=10)

    label_str = {1: "PHISHING", 0: "BENIGN", None: "UNKNOWN"}.get(label, "UNKNOWN")
    ax.set_title(f"{domain}  —  quantum walk fingerprint φ(G)  [{label_str}]\n"
                 f"{n}-dimensional vector, fixed length regardless of graph size",
                 fontsize=11)

    fig.tight_layout()
    return _save(fig, f"stageD_fingerprint_{domain}")


# ── SELF-TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "https://github.com"
    label = int(sys.argv[2]) if len(sys.argv) > 2 else None

    print("\n" + "=" * 58)
    print("  EVELYN — stage_figures()")
    print("=" * 58)

    paths = generate_all_stage_figures(target, label=label, verbose=True)