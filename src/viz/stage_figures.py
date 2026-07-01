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

def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURES_DIR / f"{name}.png"
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return str(png_path)

plt.rcParams.update({"font.family":"serif","font.size":10,"axes.spines.top":False,
    "axes.spines.right":False,"axes.edgecolor":"#444444","figure.facecolor":"white"})

TYPE_COLORS = {"domain":"#D85A30","ip":"#D4537E","registrar":"#1D9E75",
    "cert_peer":"#378ADD","asn":"#7F77DD","geo":"#5BA88A","co_host":"#C9A227",
    "favicon":"#9B59B6","jarm":"#E67E22","subdomain":"#16A085","unknown":"#888780"}
EDGE_PLAIN_LABEL = {"resolves-to":"resolves to","registered-by":"registered by",
    "shares-cert":"shares cert with","hosted-in":"hosted on","located-in":"located in",
    "also-hosts":"shares server with","uses-favicon":"uses icon",
    "has-subdomain":"subdomain of","server-fingerprint":"server fingerprint"}

def generate_all_stage_figures(domain_or_url, label=None, verbose=True,
                                include_edge_estimation=True):
    if verbose: print(f"\n  [stage_figures] Running pipeline for: {domain_or_url}")
    G = build_graph(domain_or_url, label=label, verbose=verbose, save=True)
    if G.number_of_nodes() == 0:
        if verbose: print("  ⚠ Empty graph")
        return {}
    domain = G.graph.get("domain","unknown").replace(".","_")
    ham = build_hamiltonian(G, weighted=True, variant="adjacency")
    fp = extract_fingerprint(G, method="combined")
    paths = {}
    paths["graph"] = _figure_hypergraph(G, domain, label)
    paths["hamiltonian"] = _figure_hamiltonian_heatmap(ham, domain, label)
    paths["spectrum"] = _figure_eigenvalue_spectrum(ham, domain, label)
    paths["fingerprint"] = _figure_fingerprint_bars(fp, domain, label)

    # Stage E — Edge estimation (the pivot contribution)
    if include_edge_estimation and G.number_of_edges() >= 6:
        try:
            from src.quantum.estimate_edges import estimate_missing_edges, evaluate_edge_recovery
            from src.viz.paper_figures import figure_edge_predictions, figure_recovery_comparison

            predictions = estimate_missing_edges(G, top_k=10, filter_by_type=True)
            if predictions:
                paths["edge_predictions"] = figure_edge_predictions(
                    G, predictions, max_show=5,
                    title=f"{domain} — predicted missing infrastructure [{label}]")
                if verbose:
                    print(f"\n  Top 3 predicted missing edges:")
                    for p in predictions[:3]:
                        print(f"    → {p['explanation']}")

            if G.number_of_edges() >= 8 and nx.is_connected(G):
                eval_result = evaluate_edge_recovery(G, hide_fraction=0.2, verbose=verbose)
                if "quantum_walk" in eval_result:
                    paths["edge_recovery"] = figure_recovery_comparison(eval_result)
        except Exception as e:
            if verbose: print(f"  ⚠ Edge estimation figures skipped: {e}")

    if verbose:
        n_figs = len(paths)
        print(f"\n  ✓ Generated {n_figs} stage figures for {domain}:")
        for name, p in paths.items(): print(f"      {name}: {p}")
    return paths

def generate_comparison_figures(domain_a_url, label_a, domain_b_url, label_b, verbose=True):
    if verbose: print(f"\n  [stage_figures] Comparison: {domain_a_url} vs {domain_b_url}")
    G_a = build_graph(domain_a_url, label=label_a, verbose=verbose, save=True)
    G_b = build_graph(domain_b_url, label=label_b, verbose=verbose, save=True)
    if G_a.number_of_nodes() == 0 or G_b.number_of_nodes() == 0: return {}
    name_a = G_a.graph.get("domain","unknown").replace(".","_")
    name_b = G_b.graph.get("domain","unknown").replace(".","_")
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
    return paths

def _figure_spectrum_comparison(ham_a, ham_b, name_a, name_b, label_a, label_b):
    eig_a, eig_b = np.sort(ham_a["eigenvalues"]), np.sort(ham_b["eigenvalues"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x_a, x_b = np.linspace(0, 1, len(eig_a)), np.linspace(0, 1, len(eig_b))
    color_a = "#C0392B" if label_a == 1 else "#2E6E9E"
    color_b = "#C0392B" if label_b == 1 else "#2E6E9E"
    tag_a = "PHISHING" if label_a == 1 else "benign"
    tag_b = "PHISHING" if label_b == 1 else "benign"
    ax.plot(x_a, eig_a, "o-", color=color_a, markersize=5, label=f"{name_a} [{tag_a}]")
    ax.plot(x_b, eig_b, "s--", color=color_b, markersize=5, label=f"{name_b} [{tag_b}]")
    ax.set_xlabel("Position in spectrum"); ax.set_ylabel("Eigenvalue of H")
    ax.set_title("Hamiltonian spectrum comparison"); ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, f"stageC_spectrum_compare_{name_a}_vs_{name_b}")

def _figure_fingerprint_comparison(fp_a, fp_b, name_a, name_b, label_a, label_b):
    phi_a, phi_b = fp_a["phi"], fp_b["phi"]
    n = min(len(phi_a), len(phi_b))
    phi_a, phi_b = phi_a[:n], phi_b[:n]
    color_a = "#C0392B" if label_a == 1 else "#2E6E9E"
    color_b = "#C0392B" if label_b == 1 else "#2E6E9E"
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(n); width = 0.4
    ax.bar(x - width/2, phi_a, width, color=color_a, alpha=0.85, label=name_a)
    ax.bar(x + width/2, phi_b, width, color=color_b, alpha=0.85, label=name_b)
    cosine_dist = 1.0 - float(np.dot(phi_a, phi_b) / ((np.linalg.norm(phi_a)+1e-12)*(np.linalg.norm(phi_b)+1e-12)))
    ax.set_title(f"Fingerprint comparison — cosine distance = {cosine_dist:.4f}")
    ax.legend(frameon=False); fig.tight_layout()
    return _save(fig, f"stageD_fingerprint_compare_{name_a}_vs_{name_b}")

def _figure_distance_summary(fp_a, fp_b, G_a, G_b, name_a, name_b, label_a, label_b):
    from src.quantum.gnn_baseline import graphsage_embed, embedding_distance
    phi_a, phi_b = fp_a["phi"], fp_b["phi"]
    quantum_dist = 1.0 - float(np.dot(phi_a, phi_b)/((np.linalg.norm(phi_a)+1e-12)*(np.linalg.norm(phi_b)+1e-12)))
    emb_a = graphsage_embed(G_a, seed=1)["graph_embedding"]
    emb_b = graphsage_embed(G_b, seed=1)["graph_embedding"]
    gnn_dist = embedding_distance(emb_a, emb_b, metric="cosine")
    fig, ax = plt.subplots(figsize=(8, 3)); ax.axis("off")
    ax.text(0.5, 0.85, f"{name_a} vs {name_b}", ha="center", fontsize=13, fontweight="bold", transform=ax.transAxes)
    ax.text(0.05, 0.55, f"Quantum walk: {quantum_dist:.4f}", fontsize=11, transform=ax.transAxes)
    ax.text(0.05, 0.25, f"GraphSAGE:    {gnn_dist:.4f}", fontsize=11, transform=ax.transAxes)
    fig.tight_layout()
    return _save(fig, f"summary_distance_{name_a}_vs_{name_b}")

def _figure_hypergraph(G, domain, label):
    fig, ax = plt.subplots(figsize=(9, 7.5))
    try: pos = nx.kamada_kawai_layout(G)
    except: pos = nx.spring_layout(G, seed=7, k=0.6)
    nx.draw_networkx_edges(G, pos, edge_color="#aaaaaa", width=0.9, alpha=0.7, ax=ax)
    types_present = sorted(set(d.get("type","unknown") for _, d in G.nodes(data=True)))
    for t in types_present:
        nodes = [n for n, d in G.nodes(data=True) if d.get("type") == t]
        sizes = [180 + 60 * G.degree(n) for n in nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=TYPE_COLORS.get(t,"#888780"),
                                node_size=sizes, alpha=0.9, ax=ax, edgecolors="white", linewidths=0.8, label=t)
    root = G.graph.get("domain")
    if root in pos:
        ax.annotate(root, pos[root], fontsize=9, fontweight="bold", ha="center", xytext=(0,14), textcoords="offset points")
    label_str = {1:"PHISHING",0:"BENIGN",None:"UNKNOWN"}.get(label,"UNKNOWN")
    ax.set_title(f"{domain} — infrastructure hypergraph [{label_str}]\n{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    ax.legend(loc="upper left", fontsize=7.5, frameon=False, markerscale=0.6, bbox_to_anchor=(1.0,1.0))
    ax.axis("off"); fig.tight_layout()
    return _save(fig, f"stageA_hypergraph_{domain}")

def _figure_hamiltonian_heatmap(ham, domain, label):
    H, node_order, n = ham["H"], ham["node_order"], ham["n_nodes"]
    fig_size = max(6, min(0.35*n+2, 12))
    fig, ax = plt.subplots(figsize=(fig_size+1, fig_size))
    im = ax.imshow(H, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if n <= 30:
        short = [nm[:18]+"…" if len(nm)>18 else nm for nm in node_order]
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(short, rotation=90, fontsize=6.5); ax.set_yticklabels(short, fontsize=6.5)
    label_str = {1:"PHISHING",0:"BENIGN",None:"UNKNOWN"}.get(label,"UNKNOWN")
    ax.set_title(f"{domain} — Hamiltonian H = −A [{label_str}]")
    fig.tight_layout()
    return _save(fig, f"stageB_hamiltonian_{domain}")

def _figure_eigenvalue_spectrum(ham, domain, label):
    eigenvalues = np.sort(ham["eigenvalues"]); n = len(eigenvalues)
    fig, ax = plt.subplots(figsize=(8, 4))
    ml, sl, bl = ax.stem(range(n), eigenvalues, basefmt=" ")
    ml.set_color("#BA7517"); ml.set_markersize(5); sl.set_color("#BA7517"); sl.set_linewidth(1)
    ax.axhline(0, color="#888780", linewidth=0.7, linestyle="--")
    ax.set_xlabel("Eigenvalue index"); ax.set_ylabel("Eigenvalue of H")
    label_str = {1:"PHISHING",0:"BENIGN",None:"UNKNOWN"}.get(label,"UNKNOWN")
    ax.set_title(f"{domain} — eigenvalue spectrum [{label_str}]")
    fig.tight_layout()
    return _save(fig, f"stageC_spectrum_{domain}")

def _figure_fingerprint_bars(fp, domain, label):
    phi = fp["phi"]; n = len(phi)
    eigen_hist_len = 20 if n > 20 else n // 2
    split = eigen_hist_len if fp["method"] == "combined" else n
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#BA7517" if i < split else "#3F70AC" for i in range(n)]
    ax.bar(np.arange(n), phi, color=colors, width=0.8)
    ax.set_xlabel("Fingerprint dimension"); ax.set_ylabel("Value")
    label_str = {1:"PHISHING",0:"BENIGN",None:"UNKNOWN"}.get(label,"UNKNOWN")
    ax.set_title(f"{domain} — φ(G) fingerprint [{label_str}]")
    fig.tight_layout()
    return _save(fig, f"stageD_fingerprint_{domain}")


# ── SELF-TEST / CLI ────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 5 and sys.argv[3] not in ("0", "1"):
        pass

    if len(sys.argv) >= 5:
        domain_a, label_a = sys.argv[1], int(sys.argv[2])
        domain_b, label_b = sys.argv[3], int(sys.argv[4])
        print("\n" + "=" * 58)
        print("  EVELYN — stage_figures() COMPARISON mode")
        print("=" * 58)
        generate_comparison_figures(domain_a, label_a, domain_b, label_b, verbose=True)
    elif len(sys.argv) >= 2:
        target = sys.argv[1]
        label = int(sys.argv[2]) if len(sys.argv) > 2 else None
        print("\n" + "=" * 58)
        print("  EVELYN — stage_figures()  (single-domain mode)")
        print("=" * 58)
        generate_all_stage_figures(target, label=label, verbose=True)
    else:
        print("\n  Usage:")
        print("    python -m src.viz.stage_figures <URL> [label]")
        print("    python -m src.viz.stage_figures <URL_A> <label_A> <URL_B> <label_B>")
        print("\n  Examples:")
        print("    python -m src.viz.stage_figures https://suspicious.xyz 1")
        print("    python -m src.viz.stage_figures https://phish.xyz 1 https://google.com 0")