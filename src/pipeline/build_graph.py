# src/pipeline/build_graph.py
#
# PURPOSE: Take a single URL and build its complete ego-graph as a
# labeled NetworkX object. Now includes:
#   - SSL metadata as node features (not just edges)
#   - Edge weights encoding signal strength
#   - Graph-level metadata dictionary for the paper's feature table
#   - Serialization to disk (data/graphs/)
#   - Graceful partial-graph returns — if 3/7 modules fail, you still
#     get a graph with the 4 that succeeded, not an empty failure

import networkx as nx
import pickle
from pathlib import Path

from src.pipeline.parse_url          import parse_url
from src.pipeline.resolve_dns        import resolve_dns
from src.pipeline.fetch_whois        import fetch_whois
from src.pipeline.fetch_cert         import fetch_cert
from src.pipeline.fetch_asn          import fetch_asn
from src.pipeline.fetch_geo          import fetch_geo
from src.pipeline.check_shared_hosting import check_shared_hosting
from src.pipeline.fetch_ssl_meta     import fetch_ssl_meta
from src.pipeline.fetch_page_meta import fetch_page_meta

GRAPHS_DIR = Path("data/graphs")


def build_graph(url: str,
                label: int = None,
                verbose: bool = True,
                save: bool = False) -> nx.Graph:
    """
    Builds a complete labeled NetworkX ego-graph for one URL.

    Parameters:
        url     : the full URL to analyse
        label   : 1 = phishing, 0 = benign, None = unknown
        verbose : print step-by-step progress
        save    : if True, serialise the graph to data/graphs/

    Node attributes (always present, None if unavailable):
        type, domain_age_days, cert_count, ttl,
        is_self_signed, ssl_validity_days, ssl_days_to_expiry

    Graph-level attributes (G.graph dict):
        url, domain, label, node_count, edge_count,
        modules_succeeded, modules_failed

    Returns a nx.Graph. Even on partial failure, returns whatever
    nodes were collected — never an empty graph from a partial failure.
    """

    G = nx.Graph()
    modules_succeeded = []
    modules_failed    = []

    # ── Parse URL ──────────────────────────────────────────────────────
    parsed = parse_url(url)
    domain = parsed["full_domain"]

    if not domain:
        if verbose:
            print(f"  ⚠ Could not extract domain from: {url}")
        return G

    # The domain node is the root of the entire ego-graph.
    # We attach feature attributes directly to it here — these become
    # node features in the adjacency matrix later.
    G.add_node(domain, type="domain",
               suffix=parsed["suffix"],
               is_ip=parsed["is_ip"],
               domain_age_days=None,
               cert_count=None,
               is_self_signed=None,
               ssl_validity_days=None,
               ssl_days_to_expiry=None)

    if verbose:
        print(f"\n  [build_graph] {domain}  label={'phishing' if label==1 else 'benign' if label==0 else '?'}")
        print(f"  {'─'*52}")

    # ── DNS ────────────────────────────────────────────────────────────
    dns = resolve_dns(domain)
    if dns["resolved"]:
        for ip in dns["ips"]:
            # TTL as edge weight: low TTL = weaker/more evasive connection
            # We normalize to 0-1 range by capping at 3600s (1 hour)
            ttl_weight = min(dns["ttl"], 3600) / 3600 if dns["ttl"] else 0.5
            G.add_node(ip, type="ip", ttl=dns["ttl"])
            G.add_edge(domain, ip,
                       relation="resolves-to",
                       weight=ttl_weight)
        modules_succeeded.append("dns")
        if verbose:
            print(f"  ✓ DNS:         {len(dns['ips'])} IP(s)")
    else:
        modules_failed.append(f"dns:{dns['error']}")
        if verbose:
            print(f"  ✗ DNS:         {dns['error']}")

    # ── WHOIS ──────────────────────────────────────────────────────────
    whois = fetch_whois(domain)
    if whois["resolved"] and whois["registrar"]:
        reg = whois["registrar"]
        G.add_node(reg, type="registrar")
        # Edge weight: younger domain = higher suspicion = lower weight
        # A 1-day-old domain has weight ~0. A 10-year-old domain has weight ~1.
        age = whois["domain_age_days"] or 0
        age_weight = min(age, 3650) / 3650  # cap at 10 years
        G.add_edge(domain, reg,
                   relation="registered-by",
                   weight=age_weight)
        G.nodes[domain]["domain_age_days"] = whois["domain_age_days"]
        modules_succeeded.append("whois")
        if verbose:
            print(f"  ✓ WHOIS:       {reg}, age={whois['domain_age_days']}d")
    else:
        modules_failed.append(f"whois:{whois.get('error')}")
        if verbose:
            print(f"  ✗ WHOIS:       {whois.get('error')}")

    # ── Certificate transparency ───────────────────────────────────────
    cert = fetch_cert(domain)
    if cert["resolved"]:
        for peer in cert["shared_domains"]:
            G.add_node(peer, type="cert_peer")
            # Shared cert peers all have weight 1.0 — this is a
            # cryptographic certainty, not a probabilistic inference.
            G.add_edge(domain, peer,
                       relation="shares-cert",
                       weight=1.0)
        G.nodes[domain]["cert_count"] = cert["cert_count"]
        modules_succeeded.append("cert")
        if verbose:
            print(f"  ✓ Cert:        {cert['cert_count']} certs, "
                  f"{len(cert['shared_domains'])} peers (via {cert.get('source')})")
    else:
        modules_failed.append(f"cert:{cert.get('error')}")
        if verbose:
            print(f"  ✗ Cert:        {cert.get('error')}")

    # ── SSL metadata (node features, not new graph nodes) ─────────────
    ssl = fetch_ssl_meta(domain)
    if ssl["resolved"]:
        G.nodes[domain]["is_self_signed"]      = ssl["is_self_signed"]
        G.nodes[domain]["ssl_validity_days"]   = ssl["validity_days"]
        G.nodes[domain]["ssl_days_to_expiry"]  = ssl["days_until_expiry"]
        modules_succeeded.append("ssl")
        if verbose:
            print(f"  ✓ SSL:         issuer={ssl['issuer']}, "
                  f"validity={ssl['validity_days']}d, "
                  f"self_signed={ssl['is_self_signed']}")
    else:
        modules_failed.append(f"ssl:{ssl.get('error')}")
        if verbose:
            print(f"  ✗ SSL:         {ssl.get('error')}")

    # ── ASN + Geo + shared hosting (per IP node) ───────────────────────
    for node, data in list(G.nodes(data=True)):
        if data.get("type") != "ip":
            continue
        ip = node

        asn = fetch_asn(ip)
        if asn["resolved"] and asn["asn"]:
            asn_node = f"{asn['asn']}|{asn['as_name']}"
            G.add_node(asn_node, type="asn",
                       country=asn["country"])
            G.add_edge(ip, asn_node,
                       relation="hosted-in",
                       weight=1.0)
            if "asn" not in modules_succeeded:
                modules_succeeded.append("asn")
        else:
            if "asn" not in modules_failed:
                modules_failed.append(f"asn:{asn.get('error')}")

        geo = fetch_geo(ip)
        if geo["resolved"] and geo["country"]:
            geo_node = f"{geo['city']}|{geo['country']}"
            G.add_node(geo_node, type="geo",
                       lat=geo["lat"], lon=geo["lon"],
                       isp=geo["isp"])
            G.add_edge(ip, geo_node,
                       relation="located-in",
                       weight=1.0)
            if "geo" not in modules_succeeded:
                modules_succeeded.append("geo")
        else:
            if "geo" not in modules_failed:
                modules_failed.append(f"geo:{geo.get('error')}")

        shared = check_shared_hosting(ip)
        if shared["resolved"]:
            MAX_CO_HOST = 15
            for co in shared["co_hosted_domains"][:MAX_CO_HOST]:
                G.add_node(co, type="co_host")
                G.add_edge(ip, co,
                           relation="also-hosts",
                           weight=0.5)
            if "shared" not in modules_succeeded:
                modules_succeeded.append("shared")
        else:
            if "shared" not in modules_failed:
                modules_failed.append(f"shared:{shared.get('error')}")

    # ── Enforce hard node budget ────────────────────────────────────────
    # Some domains (Google, Cloudflare, major CDNs) produce graphs with
    # 500-1000+ nodes because they have massive cert footprints and
    # thousands of co-hosted domains on shared infrastructure. This is
    # an architectural fact about hyperscaler infrastructure, not signal
    # about phishing/benign status — and it makes the quantum walk
    # computationally infeasible (O(n^3) on n=1000 is ~1 billion ops
    # PER time step, per graph).
    #
    # We cap every graph at MAX_NODES total, keeping high-priority node
    # types (domain, ip, registrar, asn, geo — the "core identity" of
    # the infrastructure) and trimming low-priority, high-volume types
    # (cert_peer, co_host) first. This is local ego-graph extraction,
    # not full graph truncation — it's a deliberate, documented design
    # choice, not data loss.
    MAX_NODES = 50
    TRIM_PRIORITY = ["co_host", "cert_peer"]   # trimmed first, in this order
    MAX_PER_TRIMMED_TYPE = {"cert_peer": 5, "co_host": 3}

    if G.number_of_nodes() > MAX_NODES:
        truncated_counts = {}

        for node_type in TRIM_PRIORITY:
            if G.number_of_nodes() <= MAX_NODES:
                break

            # Find all nodes of this type, EXCLUDING the root domain
            # (which is never trimmed regardless of its own type tag)
            nodes_of_type = [
                n for n, d in G.nodes(data=True)
                if d.get("type") == node_type and n != domain
            ]

            keep_n = MAX_PER_TRIMMED_TYPE.get(node_type, 5)
            to_remove = nodes_of_type[keep_n:]   # keep first `keep_n`, drop rest

            G.remove_nodes_from(to_remove)
            if to_remove:
                truncated_counts[node_type] = len(to_remove)

        if truncated_counts and verbose:
            print(f"  ⚠ TRUNCATED for node budget (max={MAX_NODES}): {truncated_counts}")

        # Record this truncation event — essential for the paper's
        # methods/limitations section, and for filtering analysis later
        G.graph["truncated"] = bool(truncated_counts)
        G.graph["truncated_counts"] = truncated_counts
    else:
        G.graph["truncated"] = False
        G.graph["truncated_counts"] = {}

    # ── Attach graph-level metadata ────────────────────────────────────
    # G.graph is a dictionary that travels with the graph object everywhere.
    # This metadata is what your batch pipeline and evaluation code will
    # read later — it's the "label" and "quality" info for this graph.
    G.graph.update({
        "url":               url,
        "domain":            domain,
        "label":             label,
        "node_count":        G.number_of_nodes(),
        "edge_count":        G.number_of_edges(),
        "modules_succeeded": modules_succeeded,
        "modules_failed":    modules_failed,
        # "truncated" and "truncated_counts" already set above — this
        # update() call will not overwrite them since they're not keys here
    })

    # ── Serialise to disk ──────────────────────────────────────────────
    if save:
        GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = domain.replace(".", "_").replace("/", "_")
        label_tag = f"_L{label}" if label is not None else ""
        path = GRAPHS_DIR / f"{safe_name}{label_tag}.pkl"
        with open(path, "wb") as f:
            pickle.dump(G, f)
        G.graph["saved_path"] = str(path)
        if verbose:
            print(f"\n  Saved → {path}")

    return G


def print_graph_summary(G: nx.Graph) -> None:
    print(f"\n  {'─'*52}")
    print(f"  GRAPH SUMMARY — {G.graph.get('domain', '?')}")
    print(f"  label:   {G.graph.get('label')} "
          f"({'phishing' if G.graph.get('label')==1 else 'benign' if G.graph.get('label')==0 else 'unknown'})")
    print(f"  nodes:   {G.number_of_nodes()}")
    print(f"  edges:   {G.number_of_edges()}")
    if G.graph.get("truncated"):
        print(f"  ⚠ truncated: {G.graph.get('truncated_counts')}")
    print(f"  modules: ✓ {G.graph.get('modules_succeeded')}  "
          f"✗ {G.graph.get('modules_failed')}")
    type_counts = {}
    for _, data in G.nodes(data=True):
        t = data.get("type", "?")
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, c in sorted(type_counts.items()):
        print(f"    {t:14s}: {c}")
    print(f"  {'─'*52}")


# ── SELF-TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "https://github.com"
    label = int(sys.argv[2]) if len(sys.argv) > 2 else None

    print("\n" + "=" * 58)
    print("  EVELYN — build_graph()")
    print("=" * 58)
    G = build_graph(url, label=label, verbose=True, save=True)
    print_graph_summary(G)