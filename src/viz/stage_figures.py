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


def _save(fig, name: str) -> str:
    """
    Saves a matplotlib figure as both PNG (300dpi, for quick viewing)
    and PDF (vector, for LaTeX/print) into results/figures/. Returns
    the PNG path as the canonical "here's where it went" reference.

    This function was previously missing from the module entirely —
    an editing mistake during an earlier patch dropped its definition
    while keeping every call site intact, which is exactly the kind
    of bug Python's lazy function-body execution lets slip through
    until the function is actually invoked at runtime, not at import
    time. Defining it here, immediately after FIGURES_DIR, guarantees
    it exists before any of the _figure_* functions below can call it.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURES_DIR / f"{name}.png"
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return str(png_path)

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


def generate_comparison_figures(domain_a_url: str, label_a: int,
                                 domain_b_url: str, label_b: int,
                                 verbose: bool = True) -> dict:
    """
    THE figure set that actually tells your research story: runs TWO
    domains through the pipeline and produces side-by-side comparison
    figures for the Hamiltonian spectrum and the fingerprint — because
    a single graph's spectrum or fingerprint means little in isolation;
    what matters is the CONTRAST between a phishing graph and a benign
    one. This directly addresses the "figures B/C/D are uninformative"
    problem: nothing in a single bar chart says "different from what?"
    Putting two side by side does.
    """
    if verbose:
        print(f"\n  [stage_figures] Comparison: {domain_a_url}  vs  {domain_b_url}")

    G_a = build_graph(domain_a_url, label=label_a, verbose=verbose, save=True)
    G_b = build_graph(domain_b_url, label=label_b, verbose=verbose, save=True)

    if G_a.number_of_nodes() == 0 or G_b.number_of_nodes() == 0:
        if verbose:
            print("  ⚠ One or both graphs empty — cannot compare")
        return {}

    name_a = G_a.graph.get("domain", "unknown").replace(".", "_")
    name_b = G_b.graph.get("domain", "unknown").replace(".", "_")

    ham_a = build_hamiltonian(G_a, weighted=True, variant="adjacency")
    ham_b = build_hamiltonian(G_b, weighted=True, variant="adjacency")
    fp_a = extract_fingerprint(G_a, method="combined")
    fp_b = extract_fingerprint(G_b, method="combined")

    paths = {}
    paths["graph_a"] = _figure_hypergraph(G_a, name_a, label_a)
    paths["graph_b"] = _figure_hypergraph(G_b, name_b, label_b)
    paths["spectrum_compare"] = _figure_spectrum_comparison(ham_a, ham_b, name_a, name_b, label_a, label_b)
    paths["fingerprint_compare"] = _figure_fingerprint_comparison(fp_a, fp_b, name_a, name_b, label_a, label_b)
    paths["distance_summary"] = _figure_distance_summary(fp_a, fp_b, G_a, G_b, name_a, name_b, label_a, label_b)

    if verbose:
        print(f"\n  ✓ Generated comparison figures:")
        for name, p in paths.items():
            print(f"      {name}: {p}")

    return paths


def _figure_spectrum_comparison(ham_a: dict, ham_b: dict, name_a: str, name_b: str,
                                  label_a, label_b) -> str:
    """
    Two eigenvalue spectra plotted on the SAME axes, directly overlaid.
    This is the fix for Figure C: a single spectrum tells you nothing
    without something to compare it against. Seeing a phishing
    spectrum's shape next to a benign one's — same axes, same scale —
    is what actually communicates "these are structurally different."
    """
    eig_a = np.sort(ham_a["eigenvalues"])
    eig_b = np.sort(ham_b["eigenvalues"])

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Normalize x-axis to [0,1] by RANK PERCENTILE rather than raw index,
    # since the two graphs almost certainly have different node counts —
    # this lets us compare SHAPE of the spectrum, not just raw position.
    x_a = np.linspace(0, 1, len(eig_a))
    x_b = np.linspace(0, 1, len(eig_b))

    color_a = "#C0392B" if label_a == 1 else "#2E6E9E"
    color_b = "#C0392B" if label_b == 1 else "#2E6E9E"
    tag_a = "PHISHING" if label_a == 1 else "benign" if label_a == 0 else "?"
    tag_b = "PHISHING" if label_b == 1 else "benign" if label_b == 0 else "?"

    ax.plot(x_a, eig_a, "o-", color=color_a, markersize=5, linewidth=1.5,
            label=f"{name_a}  [{tag_a}]  (n={len(eig_a)})")
    ax.plot(x_b, eig_b, "s--", color=color_b, markersize=5, linewidth=1.5,
            label=f"{name_b}  [{tag_b}]  (n={len(eig_b)})")
    ax.axhline(0, color="#888780", linewidth=0.6, linestyle=":")

    ax.set_xlabel("Position in spectrum (0 = smallest eigenvalue, 1 = largest)", fontsize=10)
    ax.set_ylabel("Eigenvalue of H", fontsize=10)
    ax.set_title("Hamiltonian spectrum shape — overlaid for direct comparison\n"
                 "Different shapes here = different infrastructure topology",
                 fontsize=11)
    ax.legend(loc="best", fontsize=9, frameon=False)

    fig.tight_layout()
    return _save(fig, f"stageC_spectrum_compare_{name_a}_vs_{name_b}")


def _figure_fingerprint_comparison(fp_a: dict, fp_b: dict, name_a: str, name_b: str,
                                     label_a, label_b) -> str:
    """
    Two fingerprints as GROUPED bars (side by side per dimension) rather
    than two separate charts — this is the fix for Figure D: the actual
    research question is "how different are these two vectors," and a
    grouped bar chart makes the gaps between bars the visual story,
    instead of asking the reader to mentally diff two separate images.
    """
    phi_a, phi_b = fp_a["phi"], fp_b["phi"]
    n = min(len(phi_a), len(phi_b))   # guard against mismatched lengths
    phi_a, phi_b = phi_a[:n], phi_b[:n]

    color_a = "#C0392B" if label_a == 1 else "#2E6E9E"
    color_b = "#C0392B" if label_b == 1 else "#2E6E9E"
    tag_a = "PHISHING" if label_a == 1 else "benign" if label_a == 0 else "?"
    tag_b = "PHISHING" if label_b == 1 else "benign" if label_b == 0 else "?"

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(n)
    width = 0.4
    ax.bar(x - width/2, phi_a, width, color=color_a, alpha=0.85, label=f"{name_a} [{tag_a}]")
    ax.bar(x + width/2, phi_b, width, color=color_b, alpha=0.85, label=f"{name_b} [{tag_b}]")

    cosine_dist = 1.0 - float(np.dot(phi_a, phi_b) /
                              ((np.linalg.norm(phi_a) + 1e-12) * (np.linalg.norm(phi_b) + 1e-12)))

    ax.set_xlabel("Fingerprint dimension", fontsize=10)
    ax.set_ylabel("Value", fontsize=10)
    ax.set_title(f"φ(G) fingerprint comparison — cosine distance = {cosine_dist:.4f}\n"
                 f"Larger gaps between bars = more structurally distinct",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=9, frameon=False)

    fig.tight_layout()
    return _save(fig, f"stageD_fingerprint_compare_{name_a}_vs_{name_b}")


def _figure_distance_summary(fp_a: dict, fp_b: dict, G_a, G_b, name_a: str, name_b: str,
                               label_a, label_b) -> str:
    """
    A single, plain-language summary card: the actual cosine distance,
    rendered as a simple gauge/scale rather than a chart — this is the
    figure a non-technical reader (or a paper's abstract figure) needs:
    one number, one sentence, no decoding required.
    """
    from src.quantum.gnn_baseline import graphsage_embed, embedding_distance

    phi_a, phi_b = fp_a["phi"], fp_b["phi"]
    quantum_dist = 1.0 - float(np.dot(phi_a, phi_b) /
                               ((np.linalg.norm(phi_a) + 1e-12) * (np.linalg.norm(phi_b) + 1e-12)))

    emb_a = graphsage_embed(G_a, seed=1)["graph_embedding"]
    emb_b = graphsage_embed(G_b, seed=1)["graph_embedding"]
    gnn_dist = embedding_distance(emb_a, emb_b, metric="cosine")

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")

    tag_a = "PHISHING" if label_a == 1 else "benign" if label_a == 0 else "?"
    tag_b = "PHISHING" if label_b == 1 else "benign" if label_b == 0 else "?"

    ax.text(0.5, 0.92, f"{name_a} [{tag_a}]   vs   {name_b} [{tag_b}]",
            ha="center", fontsize=13, fontweight="bold", transform=ax.transAxes)

    # Two horizontal "distance gauges" — visual scale from 0 (identical)
    # to a reasonable max (0.5) for cosine distance in this context.
    max_scale = 0.5
    for i, (label_text, dist, color) in enumerate([
        ("Quantum walk φ(G)", quantum_dist, "#BA7517"),
        ("GraphSAGE baseline", gnn_dist, "#3F70AC"),
    ]):
        y = 0.6 - i * 0.35
        ax.add_patch(plt.Rectangle((0.05, y - 0.06), 0.9, 0.12, transform=ax.transAxes,
                                    facecolor="#eeeeee", edgecolor="#cccccc"))
        fill_width = min(abs(dist) / max_scale, 1.0) * 0.9
        ax.add_patch(plt.Rectangle((0.05, y - 0.06), fill_width, 0.12, transform=ax.transAxes,
                                    facecolor=color))
        ax.text(0.05, y + 0.10, f"{label_text}:  distance = {dist:.4f}",
                fontsize=10, transform=ax.transAxes)

    fig.tight_layout()
    return _save(fig, f"summary_distance_{name_a}_vs_{name_b}")



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

    if len(sys.argv) >= 5 and sys.argv[3] not in ("0", "1"):
        # Fallback won't trigger normally; kept simple on purpose.
        pass

    if len(sys.argv) >= 5:
        # Comparison mode: domain_a label_a domain_b label_b
        domain_a, label_a, domain_b, label_b = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
        print("\n" + "=" * 58)
        print("  EVELYN — stage_figures() COMPARISON mode")
        print("=" * 58)
        generate_comparison_figures(domain_a, label_a, domain_b, label_b, verbose=True)
    else:
        target = sys.argv[1] if len(sys.argv) > 1 else "https://github.com"
        label = int(sys.argv[2]) if len(sys.argv) > 2 else None
        print("\n" + "=" * 58)
        print("  EVELYN — stage_figures()  (single-domain mode)")
        print("=" * 58)
        generate_all_stage_figures(target, label=label, verbose=True)