# src/pipeline/fetch_cert.py
#
# PURPOSE: Query crt.sh (Certificate Transparency log search) to find
# TLS certificates issued for a domain, and critically, find OTHER
# domains that share the same certificate. This is your strongest
# signal in the entire pipeline — a shared cert is cryptographic proof
# of common infrastructure, not a probabilistic guess.
#
# Background: since 2018, ALL publicly-trusted TLS certificates must be
# logged in public, append-only Certificate Transparency logs. crt.sh
# provides a free search interface over these logs. This means we can
# ask "what other domains has this certificate been issued for?" and
# get a real, auditable answer.

import requests


CRT_SH_URL = "https://crt.sh/"


def fetch_cert(domain: str, timeout: float = 10.0) -> dict:
    """
    Queries crt.sh for certificates issued to this domain.

    Example output:
    {
        "domain":          "secure-bank-login.xyz",
        "cert_count":      2,
        "shared_domains":  ["other-phish-domain.xyz", "third-domain.cc"],
        "issuer":          "Let's Encrypt",
        "resolved":        True,
        "error":           None
    }
    """

    # GUARD CLAUSE — same pattern as fetch_whois.py and resolve_dns.py.
    # This function must receive a bare domain, never a full URL.
    if "://" in domain or "/" in domain:
        return _failure(domain, "InvalidInput: pass a bare domain, not a full URL")

    try:
        # crt.sh supports a JSON output mode via ?output=json
        # We query with a wildcard prefix "%25." which means "this domain
        # OR any subdomain of it" in crt.sh's search syntax (%25 = URL-encoded %)
        response = requests.get(
            CRT_SH_URL,
            params={"q": f"%.{domain}", "output": "json"},
            timeout=timeout,
            headers={"User-Agent": "EVELYN-research-tool/0.1"},
        )

        # crt.sh returns HTTP 200 even when there are zero results —
        # it just returns an empty JSON list. We still check the status
        # code in case the service itself is down or rate-limiting us.
        response.raise_for_status()

        # The response can sometimes be empty text even with status 200
        # if crt.sh is having a bad day. json() will throw if so, and
        # we catch that in the except block below.
        cert_entries = response.json()

        if not cert_entries:
            return {
                "domain":         domain,
                "cert_count":     0,
                "shared_domains": [],
                "issuer":         None,
                "resolved":       True,
                "error":          None,
            }

        # Each entry in cert_entries is one certificate record. The
        # "name_value" field contains ALL domain names that certificate
        # covers — this can include Subject Alternative Names (SANs),
        # which is EXACTLY where we find "other domains sharing this cert".
        # name_value often contains multiple domains separated by newlines.
        all_names = set()
        issuers = set()

        for entry in cert_entries:
            names_in_this_cert = entry.get("name_value", "").split("\n")
            all_names.update(n.strip() for n in names_in_this_cert if n.strip())
            issuers.add(entry.get("issuer_name", "unknown"))

        # Remove the domain itself and any of ITS OWN subdomains from the
        # "shared domains" list — we only care about OTHER, unrelated
        # domains that happen to share a cert (that's the suspicious signal)
        shared_domains = sorted(
            name for name in all_names
            if not name.endswith(domain) and domain not in name
        )

        return {
            "domain":         domain,
            "cert_count":     len(cert_entries),
            "shared_domains": shared_domains,
            "issuer":         list(issuers)[0] if issuers else None,
            "resolved":       True,
            "error":          None,
        }

    except requests.exceptions.Timeout:
        return _failure(domain, "Timeout")

    except requests.exceptions.RequestException as e:
        # Covers connection errors, DNS failures for crt.sh itself,
        # HTTP error status codes raised by raise_for_status(), etc.
        return _failure(domain, f"RequestError: {e}")

    except ValueError as e:
        # response.json() raises ValueError if the body isn't valid JSON
        return _failure(domain, f"InvalidJSON: {e}")

    except Exception as e:
        return _failure(domain, f"UnknownError: {e}")


def _failure(domain: str, error_msg: str) -> dict:
    return {
        "domain":         domain,
        "cert_count":     0,
        "shared_domains": [],
        "issuer":         None,
        "resolved":       False,
        "error":          error_msg,
    }


def _print_result(result: dict) -> None:
    """Pretty-print one fetch_cert() result to the terminal."""
    print(f"\n  Domain:          {result['domain']}")
    if result["resolved"]:
        print(f"  Status:          ✓ RESOLVED")
        print(f"  Certs found:     {result['cert_count']}")
        print(f"  Issuer:          {result['issuer']}")
        if result["shared_domains"]:
            print(f"  ⚠ SHARED CERT WITH {len(result['shared_domains'])} OTHER DOMAIN(S):")
            for d in result["shared_domains"][:10]:   # cap display at 10
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
        TEST_DOMAINS = [
            "github.com",     # well-known domain, should show normal cert pattern
            "google.com",     # massive cert footprint — good edge case to observe
        ]

        print("\n" + "=" * 58)
        print("  EVELYN — fetch_cert() test suite")
        print("=" * 58)
        print("  NOTE: querying crt.sh — may take several seconds, it's a big public log")

        for domain in TEST_DOMAINS:
            _print_result(fetch_cert(domain))