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
import time

CRT_SH_URL = "https://crt.sh/"


def fetch_cert(domain: str, timeout: float = 10.0, max_retries: int = 2) -> dict:
    """
    Queries crt.sh for certificates issued to this domain.
    Retries on timeout/502 because crt.sh is a free public service that
    frequently struggles under load — this is normal, not exceptional,
    behavior for this particular data source.
    """

    if "://" in domain or "/" in domain:
        return _failure(domain, "InvalidInput: pass a bare domain, not a full URL")

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
                return {
                    "domain": domain, "cert_count": 0,
                    "shared_domains": [], "issuer": None,
                    "resolved": True, "error": None,
                }

            all_names = set()
            issuers = set()
            for entry in cert_entries:
                names_in_this_cert = entry.get("name_value", "").split("\n")
                all_names.update(n.strip() for n in names_in_this_cert if n.strip())
                issuers.add(entry.get("issuer_name", "unknown"))

            shared_domains = sorted(
                name for name in all_names
                if not name.endswith(domain) and domain not in name
            )

            return {
                "domain": domain, "cert_count": len(cert_entries),
                "shared_domains": shared_domains,
                "issuer": list(issuers)[0] if issuers else None,
                "resolved": True, "error": None,
            }

        except requests.exceptions.Timeout:
            last_error = "Timeout"
        except requests.exceptions.RequestException as e:
            last_error = f"RequestError: {e}"
        except ValueError as e:
            last_error = f"InvalidJSON: {e}"
        except Exception as e:
            last_error = f"UnknownError: {e}"

        # If we have retries left, wait a moment before trying again.
        # crt.sh failures are often transient — a 2-second pause and
        # retry succeeds more often than you'd expect.
        if attempt < max_retries:
            time.sleep(2)

    return _failure(domain, last_error)


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
        result = fetch_cert(sys.argv[1])
        _print_result(result)
        if not result["resolved"] and "RequestError" in str(result["error"]):
            print("\n  NOTE: crt.sh is a known-flaky free public service.")
            print("  This is a documented limitation, not a bug in EVELYN.")
            print("  Production systems would use a paid CT-log API (e.g. Censys).\n")

    else:
        TEST_DOMAINS = ["example.com", "x.com"]
        print("\n" + "=" * 58)
        print("  EVELYN — fetch_cert() test suite")
        print("=" * 58)
        for domain in TEST_DOMAINS:
            _print_result(fetch_cert(domain))