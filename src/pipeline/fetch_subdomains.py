# src/pipeline/fetch_subdomains.py
#
# PURPOSE: Discover OTHER subdomains of a suspicious domain — e.g. if
# the seed is "secure-bank-login.xyz", find out if "admin.secure-bank-
# login.xyz", "api.secure-bank-login.xyz", or "panel.secure-bank-
# login.xyz" also exist. Phishing operators frequently stand up an
# admin/control panel subdomain alongside the public-facing phishing
# page — finding it expands your view of the FULL attack surface,
# not just the one URL a victim clicked.
#
# TWO PASSIVE SOURCES, ZERO COST:
#   1. Certificate transparency (we already query this in fetch_cert.py
#      for shares-cert PEERS — this file re-uses the SAME raw cert
#      data to extract subdomains of THIS domain specifically, which
#      is a different question: "what subdomains does crt.sh know
#      about for snyc.com" vs "what OTHER domains share my cert")
#   2. A small, curated wordlist bruteforce via plain DNS resolution —
#      genuinely free (uses resolve_dns.py you already built), no
#      external API, just systematic checking of common phishing-
#      relevant subdomain names.
#
# WE DO NOT DO ACTIVE BRUTEFORCE AGAINST LARGE WORDLISTS. A 50,000-word
# subdomain bruteforce is aggressive active reconnaissance that starts
# to resemble what the attacker's own monitoring might flag, generates
# huge numbers of DNS queries, and is overkill for this research's
# purposes. A short, targeted list of phishing-relevant names is both
# more ethical and more efficient for what you're actually looking for.

import requests
from src.pipeline.resolve_dns import resolve_dns


CRT_SH_URL = "https://crt.sh/"

# Short, TARGETED wordlist — chosen specifically because these are the
# subdomain names most associated with phishing kit control panels,
# admin interfaces, and staging environments, per common threat-intel
# writeups on phishing kit deployment patterns (not an exhaustive
# general-purpose subdomain list — that would be a different, much
# more aggressive tool).
PHISHING_RELEVANT_SUBDOMAINS = [
    "admin", "panel", "login", "secure", "verify", "account",
    "update", "confirm", "api", "cpanel", "webmail", "mail",
    "staging", "test", "dev", "portal", "auth", "sso",
]


def fetch_subdomains_via_ct(domain: str, timeout: float = 10.0) -> dict:
    """
    Queries crt.sh for ALL certificate-logged subdomains of this exact
    domain (different from fetch_cert.py's shared_domains, which finds
    OTHER domains on the SAME cert — this finds subdomains of THIS one).

    Example output:
    {
        "domain":      "secure-bank-login.xyz",
        "subdomains":  ["admin.secure-bank-login.xyz", "api.secure-bank-login.xyz"],
        "method":      "certificate_transparency",
        "resolved":    True,
        "error":       None
    }
    """
    if "://" in domain or "/" in domain:
        return _failure(domain, "InvalidInput: pass a bare domain, not a full URL")

    try:
        response = requests.get(
            CRT_SH_URL,
            params={"q": f"%.{domain}", "output": "json"},
            timeout=timeout,
            headers={"User-Agent": "EVELYN-research-tool/0.1"},
        )
        response.raise_for_status()
        entries = response.json()

        subdomains = set()
        for entry in entries:
            for name in entry.get("name_value", "").split("\n"):
                name = name.strip()
                # Keep only names that are ACTUAL SUBDOMAINS of this
                # domain (ends with .domain, isn't the domain itself,
                # and isn't a wildcard entry like "*.domain")
                if name.endswith(f".{domain}") and not name.startswith("*"):
                    subdomains.add(name)

        return {
            "domain": domain, "subdomains": sorted(subdomains),
            "method": "certificate_transparency", "resolved": True, "error": None,
        }

    except requests.exceptions.Timeout:
        return _failure(domain, "Timeout")
    except requests.exceptions.RequestException as e:
        return _failure(domain, f"RequestError: {e}")
    except ValueError as e:
        return _failure(domain, f"InvalidJSON: {e}")
    except Exception as e:
        return _failure(domain, f"UnknownError: {e}")


def fetch_subdomains_via_dns(domain: str) -> dict:
    """
    Checks a SHORT, targeted list of phishing-relevant subdomain names
    via direct DNS resolution. This catches subdomains that exist but
    were never logged in any public certificate (e.g. an admin panel
    that uses plain HTTP, or a self-signed cert that doesn't get
    logged the same way).
    """
    found = []
    for prefix in PHISHING_RELEVANT_SUBDOMAINS:
        candidate = f"{prefix}.{domain}"
        result = resolve_dns(candidate, timeout=2.0)   # short timeout — we're checking MANY candidates
        if result["resolved"]:
            found.append({"subdomain": candidate, "ips": result["ips"]})

    return {
        "domain": domain, "subdomains": found,
        "method": "dns_wordlist", "resolved": True, "error": None,
    }


def fetch_subdomains(domain: str) -> dict:
    """
    Combines both passive sources into one result. This is the function
    build_graph.py should call.
    """
    ct_result = fetch_subdomains_via_ct(domain)
    dns_result = fetch_subdomains_via_dns(domain)

    ct_subdomains = set(ct_result["subdomains"]) if ct_result["resolved"] else set()
    dns_subdomains = {item["subdomain"] for item in dns_result["subdomains"]}

    all_subdomains = sorted(ct_subdomains | dns_subdomains)

    return {
        "domain":          domain,
        "subdomains":      all_subdomains,
        "count":           len(all_subdomains),
        "via_ct":          sorted(ct_subdomains),
        "via_dns":         sorted(dns_subdomains),
        "resolved":        ct_result["resolved"] or dns_result["resolved"],
        "error":           ct_result.get("error") if not ct_result["resolved"] else None,
    }


def _failure(domain: str, error_msg: str) -> dict:
    return {"domain": domain, "subdomains": [], "method": None,
            "resolved": False, "error": error_msg}


def _print_result(result: dict) -> None:
    print(f"\n  Domain:        {result['domain']}")
    if result["resolved"]:
        print(f"  Status:        ✓ RESOLVED")
        print(f"  Subdomains found: {result['count']}")
        for sub in result["subdomains"][:15]:
            via = "CT" if sub in result["via_ct"] else "DNS"
            print(f"      → {sub}  [{via}]")
        if result["count"] > 15:
            print(f"      ... and {result['count'] - 15} more")
    else:
        print(f"  Status:        ✗ FAILED  ({result['error']})")
    print(f"  {'─'*52}")


# ── SELF-TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        print("\n" + "=" * 58)
        print("  EVELYN — fetch_subdomains()")
        print("=" * 58)
        _print_result(fetch_subdomains(sys.argv[1]))
    else:
        print("\n" + "=" * 58)
        print("  EVELYN — fetch_subdomains() test suite")
        print("=" * 58)
        print("  NOTE: DNS wordlist check makes ~18 sequential DNS queries —")
        print("  expect a few seconds per domain\n")
        for domain in ["github.com"]:
            _print_result(fetch_subdomains(domain))