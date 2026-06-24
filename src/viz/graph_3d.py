# src/viz/graph_3d.py
#
# PURPOSE: Given a domain (or URL), run the full EVELYN pipeline and
# produce a research-grade, NON-TECHNICAL-READABLE 3D visualization
# of the resulting infrastructure hypergraph — interactive HTML and a
# high-resolution static JPEG/PDF for direct paper use.
#
# DESIGN BRIEF (why this version looks different from the first pass):
#   Audience: a reviewer OR a non-technical stakeholder must understand
#   the figure WITHOUT having read the codebase first.
#
#   Four real design moves, not just cosmetics:
#     1. SEMANTIC COLOR GROUPING — node types are grouped into three
#        meaning-categories (ownership / network / fingerprint) rather
#        than ten arbitrary colors. A reader learns 3 things, not 10.
#     2. THE Z-AXIS IS THE LEGEND — vertical layering by category is
#        labeled directly on the plot as a reading guide.
#     3. PLAIN-LANGUAGE HEADER — domain, verdict, and a one-line plain
#        summary appear directly on the figure in full sentences.
#     4. CALLOUT ANNOTATIONS — the most important 1-3 findings are
#        called out directly on the plot, like an annotated infographic.
#
# OUTPUTS:
#   - results/figures/fig3d_<domain>_<layout>.html   (interactive)
#   - results/figures/fig3d_<domain>_<layout>.jpg    (high-res, labeled)
#   - results/figures/fig3d_<domain>_<layout>.pdf    (vector, for LaTeX)

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

CATEGORY_OF_TYPE = {
    "domain":    "ownership",
    "registrar": "ownership",
    "subdomain": "ownership",
    "ip":        "network",
    "asn":       "network",
    "geo":       "network",
    "co_host":   "network",
    "cert_peer":   "fingerprint",
    "favicon":     "fingerprint",
    "jarm":        "fingerprint",
}

CATEGORY_COLORS = {
    "ownership":   "#C0392B",
    "network":     "#2E6E9E",
    "fingerprint": "#9B59B6",
}

TYPE_SHADES = {
    "domain":    "#C0392B", "registrar": "#D9684F", "subdomain": "#E89380",
    "ip":        "#2E6E9E", "asn":        "#5A93BE", "geo":       "#88B6D6", "co_host": "#B7D3E8",
    "cert_peer": "#9B59B6", "favicon":    "#B07CC6", "jarm":      "#C9A0D9",
}

TYPE_PLAIN_LABEL = {
    "domain":    "Website address",
    "registrar": "Who registered it",
    "subdomain": "Related sub-page",
    "ip":        "Server address",
    "asn":       "Hosting company",
    "geo":       "Server location",
    "co_host":   "Other site, same server",
    "cert_peer": "Shares security certificate",
    "favicon":   "Shares page icon",
    "jarm":      "Shares server fingerprint",
}

CATEGORY_LAYER_CENTER = {"ownership": 0, "network": 3.5, "fingerprint": 7}
TYPE_LAYER_OFFSET = {
    "domain": -0.3, "registrar": 0.3, "subdomain": 0,
    "ip": -0.4, "asn": 0, "geo": 0.4, "co_host": 0.8,
    "cert_peer": -0.3, "favicon": 0, "jarm": 0.3,
}

FONT_FAMILY_DISPLAY = "Georgia, 'Times New Roman', serif"
FONT_FAMILY_BODY = "'Segoe UI', Helvetica, Arial, sans-serif"


def build_and_visualize_3d(domain_or_url: str, label: int = None,
                            layout: str = "layered",
                            verbose: bool = True,
                            save: bool = True) -> dict:
    """
    Full pipeline: domain/URL in, research-grade 3D figure out.

    Parameters:
        domain_or_url : a bare domain OR a full URL
        label         : 1=phishing, 0=benign, None=unknown
        layout        : "layered" (default) or "spectral" (positions
                        nodes using Hamiltonian eigenvectors)
        verbose       : print pipeline progress
        save          : write HTML + JPG + PDF to results/figures/

    Returns dict with "graph", "html_path", "jpg_path", "pdf_path".
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "plotly is required. Install with:\n"
            "  pip install plotly kaleido --break-system-packages"
        )

    if verbose:
        print(f"\n  [graph_3d] Running full pipeline for: {domain_or_url}")

    G = build_graph(domain_or_url, label=label, verbose=verbose, save=True)

    if G.number_of_nodes() == 0:
        if verbose:
            print(f"  ⚠ Empty graph — nothing to visualize")
        return {"graph": G, "html_path": None, "jpg_path": None, "pdf_path": None}

    positions = _spectral_layout_3d(G) if layout == "spectral" else _layered_layout_3d(G)
    findings = _extract_key_findings(G)
    fig = _build_plotly_figure(G, positions, domain_or_url, label, findings, layout)

    result = {"graph": G, "html_path": None, "jpg_path": None, "pdf_path": None}

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        domain = G.graph.get("domain", "unknown").replace(".", "_")
        base_name = f"fig3d_{domain}_{layout}"

        html_path = FIGURES_DIR / f"{base_name}.html"
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        result["html_path"] = str(html_path)

        try:
            jpg_path = FIGURES_DIR / f"{base_name}.jpg"
            pdf_path = FIGURES_DIR / f"{base_name}.pdf"
            fig.write_image(str(jpg_path), width=2400, height=1500, scale=1)
            fig.write_image(str(pdf_path), width=2400, height=1500)
            result["jpg_path"] = str(jpg_path)
            result["pdf_path"] = str(pdf_path)
        except Exception as e:
            if verbose:
                print(f"  ⚠ Static image export skipped (needs 'kaleido'): {e}")

        if verbose:
            print(f"\n  ✓ Interactive (rotate/zoom/hover) → {html_path}")
            if result["jpg_path"]:
                print(f"  ✓ High-res JPEG (paper/slides)    → {result['jpg_path']}")
                print(f"  ✓ Vector PDF (LaTeX)              → {result['pdf_path']}")

    return result


def _layered_layout_3d(G: nx.Graph) -> dict:
    """Z = category layer (+ small type offset). X/Y = force-directed across the whole graph."""
    pos_2d = nx.spring_layout(G, seed=13, k=0.6, iterations=100, dim=2)

    positions = {}
    for node, data in G.nodes(data=True):
        node_type = data.get("type", "unknown")
        category = CATEGORY_OF_TYPE.get(node_type, "network")
        x, y = pos_2d[node]
        z = CATEGORY_LAYER_CENTER.get(category, 3.5) + TYPE_LAYER_OFFSET.get(node_type, 0)
        z += np.random.default_rng(hash(node) % (2**32)).uniform(-0.12, 0.12)
        positions[node] = (x * 3.2, y * 3.2, z)

    return positions


def _spectral_layout_3d(G: nx.Graph) -> dict:
    """Positions nodes using the Hamiltonian's own eigenvectors."""
    ham = build_hamiltonian(G, weighted=True, variant="laplacian")
    eigenvectors = ham["eigenvectors"]
    node_order = ham["node_order"]
    n = ham["n_nodes"]

    idx_x = 1 if n > 1 else 0
    idx_y = 2 if n > 2 else 0
    idx_z = 3 if n > 3 else 0

    positions = {}
    for i, node in enumerate(node_order):
        positions[node] = (
            eigenvectors[i, idx_x] * 10,
            eigenvectors[i, idx_y] * 10,
            eigenvectors[i, idx_z] * 10,
        )
    return positions


def _extract_key_findings(G: nx.Graph) -> list:
    """Scans the graph for 1-3 plain-language findings worth calling out on the figure."""
    findings = []

    for node, data in G.nodes(data=True):
        if data.get("type") == "asn" and data.get("reputation_tier") not in (None, "unknown"):
            tier = data.get("reputation_tier", "").replace("_", " ")
            findings.append({"node": node, "text": f"⚠ Hosted by a provider flagged as: {tier}"})

    domain_data = G.nodes.get(G.graph.get("domain"), {})
    if domain_data.get("brand_mismatch"):
        findings.append({
            "node": G.graph.get("domain"),
            "text": f"⚠ Page claims to be \"{domain_data['brand_mismatch'][0].title()}\" "
                    f"but the address doesn't match",
        })
    if domain_data.get("is_self_signed"):
        findings.append({"node": G.graph.get("domain"),
                          "text": "⚠ Uses a self-signed certificate (not independently verified)"})

    cert_peers = [n for n, d in G.nodes(data=True) if d.get("type") == "cert_peer"]
    if len(cert_peers) >= 3:
        findings.append({
            "node": G.graph.get("domain"),
            "text": f"🔗 Shares a security certificate with {len(cert_peers)} other sites",
        })

    return findings[:3]


def _build_plotly_figure(G: nx.Graph, positions: dict, domain_or_url: str,
                          label, findings: list, layout: str) -> "go.Figure":

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
        line=dict(color="rgba(140,140,140,0.35)", width=1.5),
        hoverinfo="none", showlegend=False,
    )
    traces = [edge_trace]

    types_present = sorted(
        set(data.get("type", "unknown") for _, data in G.nodes(data=True)),
        key=lambda t: (CATEGORY_OF_TYPE.get(t, "network"), t),
    )

    for node_type in types_present:
        nodes_of_type = [n for n, d in G.nodes(data=True) if d.get("type") == node_type]
        xs = [positions[n][0] for n in nodes_of_type]
        ys = [positions[n][1] for n in nodes_of_type]
        zs = [positions[n][2] for n in nodes_of_type]

        plain = TYPE_PLAIN_LABEL.get(node_type, node_type)
        hover_texts = []
        for n in nodes_of_type:
            attrs = G.nodes[n]
            extra = ""
            if node_type == "asn" and attrs.get("reputation_tier") not in (None, "unknown"):
                extra = f"<br><b>⚠ {attrs.get('reputation_tier', '').replace('_',' ')}</b>"
            display_name = n if len(n) < 40 else n[:37] + "..."
            hover_texts.append(f"<b>{display_name}</b><br><i>{plain}</i>{extra}")

        degree_sizes = [8 + 3 * G.degree(n) for n in nodes_of_type]

        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers",
            marker=dict(
                size=degree_sizes,
                color=TYPE_SHADES.get(node_type, "#888780"),
                opacity=0.9,
                line=dict(color="white", width=0.8),
            ),
            text=hover_texts,
            hoverinfo="text",
            name=f"{plain}",
        ))

    for finding in findings:
        node = finding["node"]
        if node not in positions:
            continue
        x, y, z = positions[node]
        traces.append(go.Scatter3d(
            x=[x], y=[y], z=[z + 0.6],
            mode="text",
            text=[finding["text"]],
            textfont=dict(size=13, color="#7A1F1F", family=FONT_FAMILY_BODY),
            showlegend=False,
            hoverinfo="skip",
        ))

    label_str = {1: "⚠ FLAGGED AS PHISHING", 0: "✓ BENIGN", None: "? UNKNOWN"}.get(label, "? UNKNOWN")
    label_color = {1: "#C0392B", 0: "#1D9E75", None: "#888780"}.get(label, "#888780")
    domain = G.graph.get("domain", domain_or_url)

    n_nodes, n_edges = G.number_of_nodes(), G.number_of_edges()
    plain_summary = (
        f"This map shows {n_nodes} pieces of infrastructure connected to "
        f"<b>{domain}</b> — its server, host, certificates, and related sites."
    )

    axis_guide = (
        "Read top to bottom:  <b>Fingerprint</b> (shared icons/certs) → "
        "<b>Network</b> (servers/hosting) → <b>Ownership</b> (who registered it)"
        if layout == "layered" else
        "Position reflects mathematical structure — nodes placed close together "
        "play a similar structural role in the network"
    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=f"<span style='font-family:{FONT_FAMILY_DISPLAY}; font-size:26px'>"
                 f"{domain}</span>"
                 f"<span style='font-family:{FONT_FAMILY_BODY}; font-size:16px; color:{label_color}'>"
                 f"   {label_str}</span>"
                 f"<br><span style='font-family:{FONT_FAMILY_BODY}; font-size:13px; color:#555'>"
                 f"{plain_summary}</span>"
                 f"<br><span style='font-family:{FONT_FAMILY_BODY}; font-size:11px; color:#888'>"
                 f"{axis_guide}</span>",
            x=0.02, xanchor="left",
        ),
        showlegend=True,
        legend=dict(
            title=dict(text="<b>What each color means</b>", font=dict(size=12)),
            font=dict(size=11, family=FONT_FAMILY_BODY),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#ddd", borderwidth=1,
        ),
        scene=dict(
            xaxis=dict(showticklabels=False, title="", showbackground=False, showgrid=False),
            yaxis=dict(showticklabels=False, title="", showbackground=False, showgrid=False),
            zaxis=dict(showticklabels=False, title="", showbackground=False, showgrid=False),
            bgcolor="white",
        ),
        margin=dict(l=0, r=0, b=0, t=140),
        paper_bgcolor="white",
        font=dict(family=FONT_FAMILY_BODY),
    )

    return fig


# ── SELF-TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if not PLOTLY_AVAILABLE:
        print("\n  ⚠ plotly not installed. Run:")
        print("    pip install plotly kaleido --break-system-packages")
        sys.exit(0)

    print("\n" + "=" * 58)
    print("  EVELYN — graph_3d()")
    print("=" * 58)

    target = sys.argv[1] if len(sys.argv) > 1 else "https://github.com"
    label = int(sys.argv[2]) if len(sys.argv) > 2 else None
    layout = sys.argv[3] if len(sys.argv) > 3 else "layered"

    print(f"  Target: {target}\n  Layout: {layout}\n")

    result = build_and_visualize_3d(target, label=label, layout=layout, verbose=True)

    print(f"\n  Nodes: {result['graph'].number_of_nodes()}")
    print(f"  Open the .html in a browser to rotate/zoom/explore with full hover detail.")
    print(f"  Use the .jpg for slides/reports, the .pdf for the paper itself.")