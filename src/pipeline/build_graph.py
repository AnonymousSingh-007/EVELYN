# src/pipeline/build_graph.py
#
# PURPOSE: Take a single URL and build its complete ego-graph as a
# labeled NetworkX object — now with FULL threat-hunting depth:
#   - Redirect chain resolution FIRST (find the TRUE final domain
#     before running anything else, since cloaking/shorteners/free-
#     host front-doors mean the input URL's domain may not be real target)
#   - DNS, WHOIS, cert (crt.sh+CertSpotter fallback), SSL metadata
#   - Page metadata: HTTP headers, favicon hash, title/brand mismatch
#   - Subdomain enumeration (CT logs + targeted DNS wordlist)
#   - JARM TLS server fingerprinting (catches shared phishing-kit
#     server configs even when certs/domains differ completely)
#   - ASN reputation scoring against curated bulletproof-hosting list
#   - ASN + Geo + shared hosting per IP
#   - Hard node budget enforcement (prevents hyperscaler graph blowup)
#
# Even on partial failure, returns whatever was collected — never an
# empty graph from a partial failure. Every module failure is logged
# to modules_failed for full audit transparency.

import networkx as nx
import pickle
from pathlib import Path

from src.pipeline.parse_url            import parse_url
from src.pipeline.fetch_redirect_chain import fetch_redirect_chain
from src.pipeline.resolve_dns          import resolve_dns
from src.pipeline.fetch_whois          import fetch_whois
from src.pipeline.fetch_cert           import fetch_cert
from src.pipeline.fetch_asn            import fetch_asn
from src.pipeline.fetch_geo            import fetch_geo
from src.pipeline.check_shared_hosting import check_shared_hosting
from src.pipeline.fetch_ssl_meta       import fetch_ssl_meta
from src.pipeline.fetch_page_meta      import fetch_page_meta
from src.pipeline.fetch_subdomains     import fetch_subdomains
from src.pipeline.fetch_jarm           import fetch_jarm, JARM_AVAILABLE
from src.pipeline.asn_reputation       import score_asn

GRAPHS_DIR = Path("data/graphs")
MAX_NODES = 50
TRIM_PRIORITY = ["co_host", "cert_peer", "subdomain"]
MAX_PER_TRIMMED_TYPE = {"cert_peer": 5, "co_host": 3, "subdomain": 8}

def _looks_like_real_domain(value: str) -> bool:
    if not value or not isinstance(value, str): return False
    if " " in value: return False
    if "." not in value: return False
    if len(value) > 100: return False
    if value[0].isupper() and len(value) > 1 and value[1].islower(): return False
    return True

def build_graph(url: str, label: int = None, verbose: bool = True, save: bool = False) -> nx.Graph:
    G = nx.Graph()
    modules_succeeded = []
    modules_failed    = []

    redirect = fetch_redirect_chain(url)
    if redirect["resolved"]:
        effective_url = redirect["final_url"]
        modules_succeeded.append("redirect_chain")
        if verbose and redirect["hop_count"] > 0:
            print(f"  ↪ Redirect chain ({redirect['hop_count']} hop(s)): {url} → {effective_url}")
    else:
        effective_url = url
        modules_failed.append(f"redirect_chain:{redirect.get('error')}")

    parsed = parse_url(effective_url)
    domain = parsed["full_domain"]
    if not domain:
        if verbose: print(f"  ⚠ Could not extract domain from: {effective_url}")
        return G

    G.add_node(domain, type="domain", suffix=parsed["suffix"], is_ip=parsed["is_ip"],
               domain_age_days=None, cert_count=None, is_self_signed=None,
               ssl_validity_days=None, ssl_days_to_expiry=None,
               server_header=None, redirect_count=None,
               missing_sec_headers=None, brand_mismatch=None, jarm=None)

    if verbose:
        print(f"\n  [build_graph] {domain}  label={'phishing' if label==1 else 'benign' if label==0 else '?'}")
        print(f"  {'─'*52}")

    # DNS
    dns = resolve_dns(domain)
    if dns["resolved"]:
        for ip in dns["ips"]:
            ttl_weight = min(dns["ttl"], 3600) / 3600 if dns["ttl"] else 0.5
            G.add_node(ip, type="ip", ttl=dns["ttl"])
            G.add_edge(domain, ip, relation="resolves-to", weight=ttl_weight)
        modules_succeeded.append("dns")
        if verbose: print(f"  ✓ DNS:         {len(dns['ips'])} IP(s)")
    else:
        modules_failed.append(f"dns:{dns['error']}")
        if verbose: print(f"  ✗ DNS:         {dns['error']}")

    # WHOIS
    whois_r = fetch_whois(domain)
    if whois_r["resolved"] and whois_r["registrar"]:
        reg = whois_r["registrar"]
        G.add_node(reg, type="registrar")
        age = whois_r["domain_age_days"] or 0
        age_weight = min(age, 3650) / 3650
        G.add_edge(domain, reg, relation="registered-by", weight=age_weight)
        G.nodes[domain]["domain_age_days"] = whois_r["domain_age_days"]
        modules_succeeded.append("whois")
        if verbose: print(f"  ✓ WHOIS:       {reg}, age={whois_r['domain_age_days']}d")
    else:
        modules_failed.append(f"whois:{whois_r.get('error')}")
        if verbose: print(f"  ✗ WHOIS:       {whois_r.get('error')}")

    # Cert
    cert = fetch_cert(domain)
    if cert["resolved"]:
        for peer in cert["shared_domains"]:
            if not _looks_like_real_domain(peer): continue
            G.add_node(peer, type="cert_peer")
            G.add_edge(domain, peer, relation="shares-cert", weight=1.0)
        G.nodes[domain]["cert_count"] = cert["cert_count"]
        modules_succeeded.append("cert")
        if verbose: print(f"  ✓ Cert:        {cert['cert_count']} certs, {len(cert['shared_domains'])} peers (via {cert.get('source')})")
    else:
        modules_failed.append(f"cert:{cert.get('error')}")
        if verbose: print(f"  ✗ Cert:        {cert.get('error')}")

    # SSL
    ssl_r = fetch_ssl_meta(domain)
    if ssl_r["resolved"]:
        G.nodes[domain]["is_self_signed"]     = ssl_r["is_self_signed"]
        G.nodes[domain]["ssl_validity_days"]  = ssl_r["validity_days"]
        G.nodes[domain]["ssl_days_to_expiry"] = ssl_r["days_until_expiry"]
        modules_succeeded.append("ssl")
        if verbose: print(f"  ✓ SSL:         issuer={ssl_r['issuer']}, validity={ssl_r['validity_days']}d")
    else:
        modules_failed.append(f"ssl:{ssl_r.get('error')}")
        if verbose: print(f"  ✗ SSL:         {ssl_r.get('error')}")

    # Page meta
    page_meta = fetch_page_meta(domain)
    if page_meta["resolved"]:
        if page_meta["favicon_hash"]:
            favicon_node = f"favicon:{page_meta['favicon_hash']}"
            G.add_node(favicon_node, type="favicon")
            G.add_edge(domain, favicon_node, relation="uses-favicon", weight=1.0)
        G.nodes[domain]["server_header"]      = page_meta["server_header"]
        G.nodes[domain]["redirect_count"]     = page_meta["redirect_count"]
        G.nodes[domain]["missing_sec_headers"] = len(page_meta["security_headers_missing"])
        G.nodes[domain]["page_title"]         = page_meta["page_title"]
        domain_lower = domain.lower()
        brand_mismatch = [b for b in page_meta["brand_keywords_in_title"] if b not in domain_lower]
        G.nodes[domain]["brand_mismatch"] = brand_mismatch
        modules_succeeded.append("page_meta")
        if verbose:
            fav_short = page_meta["favicon_hash"][:12] if page_meta["favicon_hash"] else None
            print(f"  ✓ PageMeta:    favicon={fav_short}...")
    else:
        modules_failed.append(f"page_meta:{page_meta.get('error')}")
        if verbose: print(f"  ✗ PageMeta:    {page_meta.get('error')}")

    # Subdomains
    subdomains = fetch_subdomains(domain)
    if subdomains["resolved"] and subdomains["count"] > 0:
        for sub in subdomains["subdomains"]:
            if not _looks_like_real_domain(sub): continue
            G.add_node(sub, type="subdomain")
            G.add_edge(domain, sub, relation="has-subdomain", weight=1.0)
        modules_succeeded.append("subdomains")
        if verbose: print(f"  ✓ Subdomains:  {subdomains['count']} found")
    else:
        if subdomains["resolved"]:
            modules_succeeded.append("subdomains")
        else:
            modules_failed.append(f"subdomains:{subdomains.get('error')}")

    # JARM
    if JARM_AVAILABLE:
        jarm = fetch_jarm(domain)
        if jarm["resolved"] and not jarm["is_null_jarm"]:
            jarm_node = f"jarm:{jarm['jarm']}"
            G.add_node(jarm_node, type="jarm")
            G.add_edge(domain, jarm_node, relation="server-fingerprint", weight=1.0)
            G.nodes[domain]["jarm"] = jarm["jarm"]
            modules_succeeded.append("jarm")
            if verbose: print(f"  ✓ JARM:        {jarm['jarm'][:20]}...")
        else:
            modules_failed.append(f"jarm:{jarm.get('error') or 'null_jarm'}")

    # ASN + Geo + shared hosting per IP
    for node, data in list(G.nodes(data=True)):
        if data.get("type") != "ip": continue
        ip = node
        asn = fetch_asn(ip)
        if asn["resolved"] and asn["asn"]:
            asn_node = f"{asn['asn']}|{asn['as_name']}"
            reputation = score_asn(asn["asn"])
            G.add_node(asn_node, type="asn", country=asn["country"],
                       reputation_tier=reputation["tier"], reputation_label=reputation["label"],
                       risk_score=reputation["risk_score"])
            G.add_edge(ip, asn_node, relation="hosted-in", weight=1.0)
            if "asn" not in modules_succeeded: modules_succeeded.append("asn")
            if verbose and reputation["is_flagged"]:
                print(f"  ⚠ ASN REPUTATION ({ip}): {reputation['tier'].upper()} — {reputation['label']}")
        else:
            if "asn" not in modules_failed: modules_failed.append(f"asn:{asn.get('error')}")

        geo = fetch_geo(ip)
        if geo["resolved"] and geo["country"]:
            geo_node = f"{geo['city']}|{geo['country']}"
            G.add_node(geo_node, type="geo", lat=geo["lat"], lon=geo["lon"], isp=geo["isp"])
            G.add_edge(ip, geo_node, relation="located-in", weight=1.0)
            if "geo" not in modules_succeeded: modules_succeeded.append("geo")
        else:
            if "geo" not in modules_failed: modules_failed.append(f"geo:{geo.get('error')}")

        shared = check_shared_hosting(ip)
        if shared["resolved"]:
            MAX_CO_HOST = 15
            for co in shared["co_hosted_domains"][:MAX_CO_HOST]:
                if not _looks_like_real_domain(co): continue
                G.add_node(co, type="co_host")
                G.add_edge(ip, co, relation="also-hosts", weight=0.5)
            if "shared" not in modules_succeeded: modules_succeeded.append("shared")
        else:
            if "shared" not in modules_failed:
                raw_error = str(shared.get("error", ""))
                truncated_error = raw_error[:200] + ("...[truncated]" if len(raw_error) > 200 else "")
                modules_failed.append(f"shared:{truncated_error}")

    # Enforce hard node budget
    if G.number_of_nodes() > MAX_NODES:
        truncated_counts = {}
        for node_type in TRIM_PRIORITY:
            if G.number_of_nodes() <= MAX_NODES: break
            nodes_of_type = [n for n, d in G.nodes(data=True) if d.get("type") == node_type and n != domain]
            keep_n = MAX_PER_TRIMMED_TYPE.get(node_type, 5)
            to_remove = nodes_of_type[keep_n:]
            G.remove_nodes_from(to_remove)
            if to_remove: truncated_counts[node_type] = len(to_remove)
        if truncated_counts and verbose: print(f"  ⚠ TRUNCATED for node budget (max={MAX_NODES}): {truncated_counts}")
        G.graph["truncated"] = bool(truncated_counts)
        G.graph["truncated_counts"] = truncated_counts
    else:
        G.graph["truncated"] = False
        G.graph["truncated_counts"] = {}

    G.graph.update({"url": effective_url, "original_url": url,
                    "redirect_hop_count": redirect.get("hop_count", 0),
                    "domain": domain, "label": label,
                    "node_count": G.number_of_nodes(), "edge_count": G.number_of_edges(),
                    "modules_succeeded": modules_succeeded, "modules_failed": modules_failed})

    if save:
        GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = domain.replace(".", "_").replace("/", "_")
        label_tag = f"_L{label}" if label is not None else ""
        path = GRAPHS_DIR / f"{safe_name}{label_tag}.pkl"
        with open(path, "wb") as f: pickle.dump(G, f)
        G.graph["saved_path"] = str(path)
        if verbose: print(f"\n  Saved → {path}")

    return G

def print_graph_summary(G: nx.Graph) -> None:
    print(f"\n  {'─'*52}")
    print(f"  GRAPH SUMMARY — {G.graph.get('domain', '?')}")
    print(f"  nodes: {G.number_of_nodes()}  edges: {G.number_of_edges()}")
    print(f"  modules: ✓ {G.graph.get('modules_succeeded')}")
    print(f"           ✗ {G.graph.get('modules_failed')}")
    type_counts = {}
    for _, data in G.nodes(data=True):
        t = data.get("type", "?")
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, c in sorted(type_counts.items()):
        print(f"    {t:14s}: {c}")
    print(f"  {'─'*52}")


# ── SELF-TEST / CLI ────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "https://github.com"
    label = int(sys.argv[2]) if len(sys.argv) > 2 else None

    print("\n" + "=" * 58)
    print("  EVELYN — build_graph()")
    print("=" * 58)
    G = build_graph(url, label=label, verbose=True, save=True)
    print_graph_summary(G)