# src/viz/graph_3d.py
#
# PURPOSE: Given a single domain (or URL), run the FULL EVELYN pipeline
# end-to-end — build_graph() → hamiltonian() → layout — and produce a
# 3D visualization of the resulting infrastructure hypergraph. This is
# the "pass a domain, get a paper-ready figure" entry point requested.
#
# WHY 3D AND NOT JUST 2D SPRING LAYOUT:
#   Your graphs now have up to 9 distinct node types (domain, ip,
#   registrar, cert_peer, asn, geo, co_host, favicon, jarm, subdomain).
#   A 2D spring layout gets visually crowded fast once you have this
#   many categories overlapping in a flat plane. A 3D layout gives the
#   graph room to separate by TYPE along one axis (so all "cert_peer"
#   nodes naturally cluster at one height, all "asn" nodes at another)
#   while still using force-directed X/Y placement WITHIN each type
#   layer — this produces a far more readable figure for a graph this
#   rich, and is genuinely common in security-visualization research
#   for exactly this reason (separating heterogeneous node types along
#   a third dimension rather than cramming everything into one plane).
#
# WHY WE ALSO USE THE HAMILTONIAN HERE:
#   We don't just lay out the raw graph — we additionally compute the
#   Hamiltonian's eigenvector structure and use the TOP 2 eigenvectors
#   as an alternative "spectral layout" option. Nodes that are
#   structurally similar (same role in the topology, per the quantum
#   walk's own mathematical lens) end up positioned near each other.
#   This means the 3D figure visually previews the SAME structural
#   information your quantum walk fingerprint is built from — a
#   reviewer can SEE the spectral structure your method exploits,
#   not just take your equation's word for it.
#
# OUTPUTS:
#   - An interactive standalone HTML file (rotate/zoom/hover in browser)
#   - A static PNG/PDF for direct paper inclusion (camera angle fixed)

import pickle
from pathlib import Path

import numpy as np
import networkx as nx

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from src.pipeline.build_graph import build_graph
from src.quantum.hamiltonian import build_hamiltonian


FIGURES_DIR = Path("results/figures")
GRAPHS_DIR  = Path("data/graphs")

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

TYPE_LAYER_HEIGHT = {
    "domain": 0, "ip": 1, "registrar": 2, "asn": 3, "geo": 4,
    "cert_peer": 5, "subdomain": 5.5, "favicon": 6, "jarm": 6.5,
    "co_host": 7, "unknown": 8,
}


def build_and_visualize_3d(domain_or_url: str, label: int = None,
                            layout: str = "layered",
                            verbose: bool = True,
                            save: bool = True) -> dict:
    """
    Full pipeline: domain/URL in, 3D figure out.

    Parameters:
        domain_or_url : a bare domain OR a full URL — build_graph()
                        handles redirect resolution internally either way
        label         : 1=phishing, 0=benign, None=unknown
        layout        : "layered"  — group by node type along Z-axis
                                      (default, most readable for
                                      heterogeneous graphs)
                        "spectral" — use Hamiltonian eigenvectors as
                                      X/Y/Z directly (shows the
                                      quantum-walk-relevant structure)
        verbose       : print pipeline progress
        save          : write HTML + PNG/PDF to results/figures/

    Returns:
    {
        "graph":        the built nx.Graph,
        "html_path":    str or None,
        "png_path":     str or None,
        "pdf_path":     str or None,
    }
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "plotly is required for 3D visualization. Install with:\n"
            "  pip install plotly --break-system-packages"
        )

    if verbose:
        print(f"\n  [graph_3d] Running full pipeline for: {domain_or_url}")

    G = build_graph(domain_or_url, label=label, verbose=verbose, save=True)

    if G.number_of_nodes() == 0:
        if verbose:
            print(f"  ⚠ Empty graph — nothing to visualize")
        return {"graph": G, "html_path": None, "png_path": None, "pdf_path": None}

    if layout == "spectral":
        positions = _spectral_layout_3d(G)
    else:
        positions = _layered_layout_3d(G)

    fig = _build_plotly_figure(G, positions, domain_or_url, label)

    result = {"graph": G, "html_path": None, "png_path": None, "pdf_path": None}

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        domain = G.graph.get("domain", "unknown").replace(".", "_")
        base_name = f"fig3d_{domain}_{layout}"

        html_path = FIGURES_DIR / f"{base_name}.html"
        fig.write_html(str(html_path))
        result["html_path"] = str(html_path)

        try:
            png_path = FIGURES_DIR / f"{base_name}.png"
            pdf_path = FIGURES_DIR / f"{base_name}.pdf"
            fig.write_image(str(png_path), scale=2)
            fig.write_image(str(pdf_path))
            result["png_path"] = str(png_path)
            result["pdf_path"] = str(pdf_path)
        except Exception as e:
            if verbose:
                print(f"  ⚠ Static image export skipped (install 'kaleido' "
                      f"for PNG/PDF): {e}")

        if verbose:
            print(f"\n  ✓ Saved interactive 3D figure → {html_path}")
            if result["png_path"]:
                print(f"  ✓ Saved static figure → {result['png_path']} / {result['pdf_path']}")

    return result


def _layered_layout_3d(G: nx.Graph) -> dict:
    """
    Z-axis = fixed height per node type (TYPE_LAYER_HEIGHT).
    X/Y within each layer = force-directed (spring) layout computed
    on the FULL graph first, so structurally-connected nodes still
    pull toward each other horizontally even across layers.
    """
    pos_2d = nx.spring_layout(G, seed=13, k=0.6, iterations=100, dim=2)

    positions = {}
    for node, data in G.nodes(data=True):
        node_type = data.get("type", "unknown")
        x, y = pos_2d[node]
        z = TYPE_LAYER_HEIGHT.get(node_type, TYPE_LAYER_HEIGHT["unknown"])
        z += np.random.default_rng(hash(node) % (2**32)).uniform(-0.15, 0.15)
        positions[node] = (x * 3, y * 3, z)

    return positions


def _spectral_layout_3d(G: nx.Graph) -> dict:
    """
    Uses the Hamiltonian's eigenvectors directly as 3D coordinates —
    specifically the eigenvectors for the 2nd, 3rd, and 4th smallest
    eigenvalues (skipping the smallest, which for Laplacian-style
    Hamiltonians is typically uninformative — standard spectral graph
    theory convention). This shows the SAME mathematical structure
    the quantum walk fingerprint is built from, as a spatial layout.
    """
    ham = build_hamiltonian(G, weighted=True, variant="laplacian")
    eigenvectors = ham["eigenvectors"]
    node_order = ham["node_order"]
    n = ham["n_nodes"]

    idx_x = 1 if n > 1 else 0
    idx_y = 2 if n > 2 else 0
    idx_z = 3 if n > 3 else 0

    positions = {}
    for i, node in enumerate(node_order):
        x = eigenvectors[i, idx_x] * 10
        y = eigenvectors[i, idx_y] * 10
        z = eigenvectors[i, idx_z] * 10
        positions[node] = (x, y, z)

    return positions


def _build_plotly_figure(G: nx.Graph, positions: dict, domain_or_url: str, label) -> "go.Figure":
    """Builds the interactive 3D Plotly figure from a graph and a position dict."""

    edge_x, edge_y, edge_z = [], [], []
    for u, v in G.edges():
        x0, y0, z0 = positions[u]
        x1, y1, z1 = positions[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode="lines",
        line=dict(color="rgba(150,150,150,0.4)", width=1.5),
        hoverinfo="none",
        showlegend=False,
    )

    traces = [edge_trace]
    types_present = sorted(set(data.get("type", "unknown") for _, data in G.nodes(data=True)))

    for node_type in types_present:
        nodes_of_type = [n for n, d in G.nodes(data=True) if d.get("type") == node_type]
        xs = [positions[n][0] for n in nodes_of_type]
        ys = [positions[n][1] for n in nodes_of_type]
        zs = [positions[n][2] for n in nodes_of_type]

        hover_texts = []
        for n in nodes_of_type:
            attrs = G.nodes[n]
            extra = ""
            if node_type == "asn" and attrs.get("reputation_tier") not in (None, "unknown"):
                extra = f"<br>⚠ {attrs.get('reputation_tier')}: {attrs.get('reputation_label')}"
            hover_texts.append(f"<b>{n}</b><br>type: {node_type}{extra}")

        degree_sizes = [6 + 3 * G.degree(n) for n in nodes_of_type]

        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers",
            marker=dict(
                size=degree_sizes,
                color=TYPE_COLORS.get(node_type, TYPE_COLORS["unknown"]),
                opacity=0.85,
                line=dict(color="white", width=0.5),
            ),
            text=hover_texts,
            hoverinfo="text",
            name=node_type,
        ))

    label_str = {1: "PHISHING", 0: "BENIGN", None: "UNKNOWN"}.get(label, "UNKNOWN")
    domain = G.graph.get("domain", domain_or_url)

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"EVELYN infrastructure hypergraph — {domain}  [{label_str}]"
              f"<br><sub>{G.number_of_nodes()} nodes · {G.number_of_edges()} edges</sub>",
        showlegend=True,
        scene=dict(
            xaxis=dict(showticklabels=False, title=""),
            yaxis=dict(showticklabels=False, title=""),
            zaxis=dict(showticklabels=False, title=""),
            bgcolor="white",
        ),
        margin=dict(l=0, r=0, b=0, t=60),
        paper_bgcolor="white",
    )

    return fig


# ── SELF-TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if not PLOTLY_AVAILABLE:
        print("\n  ⚠ plotly not installed. Run:")
        print("    pip install plotly --break-system-packages")
        print("    pip install kaleido --break-system-packages   (for PNG/PDF export)")
        sys.exit(0)

    print("\n" + "=" * 58)
    print("  EVELYN — graph_3d()")
    print("=" * 58)

    target = sys.argv[1] if len(sys.argv) > 1 else "https://github.com"
    label = int(sys.argv[2]) if len(sys.argv) > 2 else None
    layout = sys.argv[3] if len(sys.argv) > 3 else "layered"

    print(f"  Target: {target}")
    print(f"  Layout: {layout}\n")

    result = build_and_visualize_3d(target, label=label, layout=layout, verbose=True)

    print(f"\n  Nodes: {result['graph'].number_of_nodes()}")
    print(f"  Open the .html file in any browser to rotate/zoom/explore.")