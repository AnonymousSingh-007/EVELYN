# src/pipeline/fetch_cert.py
#
# PURPOSE: Query Certificate Transparency logs to find TLS certificates
# issued for a domain, and find OTHER domains that share the same
# certificate. This is your strongest signal in the entire pipeline —
# a shared cert is cryptographic proof of common infrastructure.
#
# TWO-SOURCE STRATEGY:
#   Primary:  crt.sh        — most complete, but frequently overloaded (502s)
#   Fallback: CertSpotter    — smaller free tier, but far more reliable uptime
#
# This dual-source pattern is itself worth documenting in your paper's
# methods/limitations section: free CT-log search tools are inherently
# best-effort infrastructure, and a production system should budget for
# a paid tier. We mitigate this with redundancy instead of giving up.

import requests
import time


CRT_SH_URL        = "https://crt.sh/"
CERTSPOTTER_URL   = "https://api.certspotter.com/v1/issuances"


def fetch_cert(domain: str, timeout: float = 10.0, max_retries: int = 1) -> dict:
    """
    Queries Certificate Transparency logs for certs issued to this domain.
    Tries crt.sh first (most complete), falls back to CertSpotter if
    crt.sh fails. Returns which source actually succeeded so you can
    track this in your dataset for transparency in the paper.

    Example output:
    {
        "domain":          "secure-bank-login.xyz",
        "cert_count":      2,
        "shared_domains":  ["other-phish-domain.xyz"],
        "issuer":          "Let's Encrypt",
        "source":          "crt.sh",
        "resolved":        True,
        "error":           None
    }
    """

    if "://" in domain or "/" in domain:
        return _failure(domain, "InvalidInput: pass a bare domain, not a full URL")

    # Try crt.sh first — it's the most complete public CT log search.
    result = _query_crtsh(domain, timeout, max_retries)
    if result["resolved"]:
        result["source"] = "crt.sh"
        return result

    crtsh_error = result["error"]

    # crt.sh failed — fall back to CertSpotter.
    result = _query_certspotter(domain, timeout)
    if result["resolved"]:
        result["source"] = "certspotter"
        return result

    # Both sources failed. Report both errors so you can see the full
    # picture when debugging a batch run later.
    return _failure(domain, f"Both sources failed. crt.sh: {crtsh_error} | certspotter: {result['error']}")


def _query_crtsh(domain: str, timeout: float, max_retries: int) -> dict:
    """Queries crt.sh. Retries once on transient failure (502/timeout)."""
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            response = requests.get(
                CRT_SH_URL,
                params={"q": f"%.{domain}", "output": "json"},
                timeout=timeout,
                headers={"User-Agent": "EVELYN-research-tool/0.1"},
            )
            response.raise_for_status()
            cert_entries = response.json()

            if not cert_entries:
                return {"domain": domain, "cert_count": 0, "shared_domains": [],
                        "issuer": None, "resolved": True, "error": None}

            all_names, issuers = set(), set()
            for entry in cert_entries:
                names = entry.get("name_value", "").split("\n")
                all_names.update(n.strip() for n in names if n.strip())
                issuers.add(entry.get("issuer_name", "unknown"))

            shared = sorted(n for n in all_names if not n.endswith(domain) and domain not in n)

            return {"domain": domain, "cert_count": len(cert_entries),
                     "shared_domains": shared,
                     "issuer": list(issuers)[0] if issuers else None,
                     "resolved": True, "error": None}

        except requests.exceptions.Timeout:
            last_error = "Timeout"
        except requests.exceptions.RequestException as e:
            last_error = f"RequestError: {e}"
        except ValueError as e:
            last_error = f"InvalidJSON: {e}"
        except Exception as e:
            last_error = f"UnknownError: {e}"

        if attempt < max_retries:
            time.sleep(1.5)

    return _failure(domain, last_error)


def _query_certspotter(domain: str, timeout: float) -> dict:
    """
    Queries CertSpotter as a fallback. Unauthenticated requests are
    rate-limited (a handful per hour) — fine for development/research
    scale, but for a full 1000-URL batch run later you'd want to either
    register a free API key or stagger requests with delays.
    """
    try:
        response = requests.get(
            CERTSPOTTER_URL,
            params={
                "domain": domain,
                "include_subdomains": "true",
                "expand": "dns_names",
            },
            timeout=timeout,
            headers={"User-Agent": "EVELYN-research-tool/0.1"},
        )
        response.raise_for_status()
        issuances = response.json()

        if not issuances:
            return {"domain": domain, "cert_count": 0, "shared_domains": [],
                     "issuer": None, "resolved": True, "error": None}

        all_names = set()
        for entry in issuances:
            all_names.update(_clean_domain_name(n) for n in entry.get("dns_names", []))

        shared = sorted(n for n in all_names if not n.endswith(domain) and domain not in n)

        # CertSpotter's basic (unauthenticated) response doesn't include
        # issuer name unless you add &expand=issuer — keeping this call
        # lean (no issuer expand) reduces the chance of hitting the free
        # rate limit, since issuer detail isn't essential to the fallback.
        return {"domain": domain, "cert_count": len(issuances),
                 "shared_domains": shared, "issuer": "unknown (certspotter basic)",
                 "resolved": True, "error": None}

    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 429:
            return _failure(domain, "RateLimited: CertSpotter unauthenticated quota exceeded")
        return _failure(domain, f"HTTPError: {e}")
    except requests.exceptions.Timeout:
        return _failure(domain, "Timeout")
    except requests.exceptions.RequestException as e:
        return _failure(domain, f"RequestError: {e}")
    except ValueError as e:
        return _failure(domain, f"InvalidJSON: {e}")
    except Exception as e:
        return _failure(domain, f"UnknownError: {e}")

def _clean_domain_name(name: str) -> str:
    """
    CertSpotter occasionally returns markdown-style links instead of
    plain domain strings in some fields. Strip that down to a clean
    domain name so downstream graph code never has to deal with it.
    """
    if name.startswith("[") and "](" in name:
        name = name.split("](")[0].lstrip("[")
    return name

def _failure(domain: str, error_msg: str) -> dict:
    return {
        "domain": domain, "cert_count": 0, "shared_domains": [],
        "issuer": None, "source": None, "resolved": False, "error": error_msg,
    }


def _print_result(result: dict) -> None:
    print(f"\n  Domain:          {result['domain']}")
    if result["resolved"]:
        print(f"  Status:          ✓ RESOLVED  (source: {result.get('source', '?')})")
        print(f"  Certs found:     {result['cert_count']}")
        print(f"  Issuer:          {result['issuer']}")
        if result["shared_domains"]:
            print(f"  ⚠ SHARED CERT WITH {len(result['shared_domains'])} OTHER DOMAIN(S):")
            for d in result["shared_domains"][:10]:
                print(f"      → {d}")
            if len(result["shared_domains"]) > 10:
                print(f"      ... and {len(result['shared_domains']) - 10} more")
        else:
            print(f"  Shared domains:  none found")
    else:
        print(f"  Status:          ✗ FAILED  ({result['error']})")
    print(f"  {'─'*52}")


# ── SELF-TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        print("\n" + "=" * 58)
        print("  EVELYN — fetch_cert()")
        print("=" * 58)
        _print_result(fetch_cert(sys.argv[1]))
    else:
        TEST_DOMAINS = ["example.com", "x.com"]
        print("\n" + "=" * 58)
        print("  EVELYN — fetch_cert() test suite")
        print("=" * 58)
        print("  Trying crt.sh first, falling back to CertSpotter on failure")
        for domain in TEST_DOMAINS:
            _print_result(fetch_cert(domain))