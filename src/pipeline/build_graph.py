# src/pipeline/build_graph.py
#
# PURPOSE: Take a single URL and build its complete "ego-graph" — a
# NetworkX graph containing every node type we've collected (domain,
# IP, registrar, certificate, shared-cert domains, ASN, geolocation,
# co-hosted domains) and the edges connecting them. This is G_i — the
# exact object the quantum walk in Stage 2 will consume.
#
# This file doesn't introduce NEW data collection logic. It is purely
# an ORCHESTRATOR: it calls the seven functions you've already built
# and tested, and wires their outputs together into one graph.

import networkx as nx

from src.pipeline.parse_url import parse_url
from src.pipeline.resolve_dns import resolve_dns
from src.pipeline.fetch_whois import fetch_whois
from src.pipeline.fetch_cert import fetch_cert
from src.pipeline.fetch_asn import fetch_asn
from src.pipeline.fetch_geo import fetch_geo
from src.pipeline.check_shared_hosting import check_shared_hosting


def build_graph(url: str, verbose: bool = True) -> nx.Graph:
    """
    Builds a NetworkX graph for one URL by calling every pipeline
    module and wiring their results into nodes and edges.

    Node types added (each tagged with a "type" attribute so later
    code, like hamiltonian.py, can distinguish them if needed):
      - domain    : the URL's registered domain
      - ip        : resolved IP address(es)
      - registrar : WHOIS registrar
      - cert_peer : other domains sharing a TLS certificate
      - asn       : hosting ASN
      - geo       : country/city of the IP
      - co_host   : other domains sharing the same IP

    Returns a networkx.Graph object. Use G.nodes(data=True) and
    G.edges(data=True) to inspect it, or nx.write_gpickle() to save it.
    """

    G = nx.Graph()

    # ── Step 1: parse the URL ──────────────────────────────────────────
    parsed = parse_url(url)
    domain = parsed["full_domain"]

    if not domain:
        if verbose:
            print(f"  ⚠ Could not extract a domain from this URL — likely a raw IP. Skipping.")
        return G

    G.add_node(domain, type="domain", suffix=parsed["suffix"], is_ip=parsed["is_ip"])

    if verbose:
        print(f"\n  Building graph for: {domain}")

    # ── Step 2: DNS resolution ──────────────────────────────────────────
    dns_result = resolve_dns(domain)
    if dns_result["resolved"]:
        for ip in dns_result["ips"]:
            G.add_node(ip, type="ip", ttl=dns_result["ttl"])
            G.add_edge(domain, ip, relation="resolves-to")
        if verbose:
            print(f"  ✓ DNS:        {len(dns_result['ips'])} IP(s) added")
    elif verbose:
        print(f"  ✗ DNS failed: {dns_result['error']}")

    # ── Step 3: WHOIS ────────────────────────────────────────────────────
    whois_result = fetch_whois(domain)
    if whois_result["resolved"] and whois_result["registrar"]:
        registrar = whois_result["registrar"]
        G.add_node(registrar, type="registrar")
        G.add_edge(domain, registrar, relation="registered-by")
        # Store domain age directly on the domain node — this is a
        # NODE FEATURE, not a separate node, since age is a property
        # of the domain itself, not a distinct entity it connects to.
        G.nodes[domain]["domain_age_days"] = whois_result["domain_age_days"]
        if verbose:
            print(f"  ✓ WHOIS:      registrar = {registrar}, age = {whois_result['domain_age_days']} days")
    elif verbose:
        print(f"  ✗ WHOIS failed or no registrar: {whois_result['error']}")

    # ── Step 4: Certificate transparency ────────────────────────────────
    cert_result = fetch_cert(domain)
    if cert_result["resolved"]:
        for shared_domain in cert_result["shared_domains"]:
            G.add_node(shared_domain, type="cert_peer")
            G.add_edge(domain, shared_domain, relation="shares-cert")
        G.nodes[domain]["cert_count"] = cert_result["cert_count"]
        if verbose:
            print(f"  ✓ Cert:       {cert_result['cert_count']} certs, "
                  f"{len(cert_result['shared_domains'])} shared-cert peer(s) "
                  f"(source: {cert_result.get('source')})")
    elif verbose:
        print(f"  ✗ Cert failed: {cert_result['error']}")

    # ── Step 5: ASN + Geo + shared hosting (per IP) ─────────────────────
    if dns_result["resolved"]:
        for ip in dns_result["ips"]:

            asn_result = fetch_asn(ip)
            if asn_result["resolved"] and asn_result["asn"]:
                asn_node = f"{asn_result['asn']} ({asn_result['as_name']})"
                G.add_node(asn_node, type="asn")
                G.add_edge(ip, asn_node, relation="hosted-in")
                if verbose:
                    print(f"  ✓ ASN ({ip}): {asn_node}")
            elif verbose:
                print(f"  ✗ ASN failed for {ip}: {asn_result['error']}")

            geo_result = fetch_geo(ip)
            if geo_result["resolved"] and geo_result["country"]:
                geo_node = f"{geo_result['city']}, {geo_result['country']}"
                G.add_node(geo_node, type="geo")
                G.add_edge(ip, geo_node, relation="located-in")
                if verbose:
                    print(f"  ✓ Geo ({ip}): {geo_node}")
            elif verbose:
                print(f"  ✗ Geo failed for {ip}: {geo_result['error']}")

            host_result = check_shared_hosting(ip)
            if host_result["resolved"]:
                # Cap co-hosted domains we add as nodes — some IPs (like
                # 8.8.8.8) return hundreds, which would bloat the graph
                # with noise rather than signal. We keep this configurable
                # via a constant so it's easy to tune later.
                MAX_CO_HOST_NODES = 15
                for co_domain in host_result["co_hosted_domains"][:MAX_CO_HOST_NODES]:
                    G.add_node(co_domain, type="co_host")
                    G.add_edge(ip, co_domain, relation="also-hosts")
                if verbose:
                    print(f"  ✓ Shared hosting ({ip}): {host_result['count']} found, "
                          f"{min(host_result['count'], MAX_CO_HOST_NODES)} added to graph")
            elif verbose:
                print(f"  ✗ Shared hosting failed for {ip}: {host_result['error']}")

    return G


def print_graph_summary(G: nx.Graph) -> None:
    """Prints a quick summary of a built graph — node/edge counts by type."""
    print(f"\n  {'─'*52}")
    print(f"  GRAPH SUMMARY")
    print(f"  {'─'*52}")
    print(f"  Total nodes: {G.number_of_nodes()}")
    print(f"  Total edges: {G.number_of_edges()}")

    # Count nodes by their "type" attribute
    type_counts = {}
    for node, data in G.nodes(data=True):
        node_type = data.get("type", "unknown")
        type_counts[node_type] = type_counts.get(node_type, 0) + 1

    for node_type, count in sorted(type_counts.items()):
        print(f"    {node_type:12s}: {count}")
    print(f"  {'─'*52}")


# ── SELF-TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        url = sys.argv[1]
        print("\n" + "=" * 58)
        print("  EVELYN — build_graph()")
        print("=" * 58)
        G = build_graph(url)
        print_graph_summary(G)

        print(f"\n  All nodes:")
        for node, data in G.nodes(data=True):
            print(f"    {node}  [{data.get('type', '?')}]")

    else:
        print("\n" + "=" * 58)
        print("  EVELYN — build_graph() test suite")
        print("=" * 58)
        print("  NOTE: this calls all 7 pipeline modules — will take 10-30 seconds")

        G = build_graph("https://github.com")
        print_graph_summary(G)