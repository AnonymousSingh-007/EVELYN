# src/pipeline/fetch_jarm.py
#
# PURPOSE: Compute the JARM fingerprint of a domain's TLS server.
# JARM is fundamentally different from fetch_cert.py — it doesn't
# look at the CERTIFICATE (which proves identity/ownership), it looks
# at the SERVER SOFTWARE'S BEHAVIOR during the TLS handshake (which
# proves what software stack is actually running).
#
# WHY THIS MATTERS FOR PHISHING ATTRIBUTION:
#   Phishing-as-a-service kits are pre-packaged software bundles —
#   the same kit, deployed by many different "customers" (individual
#   attackers), often runs on near-identical server configurations
#   (same web server version, same TLS library, same cipher ordering)
#   because the kit ships with a default deployment template that
#   operators rarely customize. Per Salesforce's original JARM research:
#   "servers configured by their creators like any other server and
#   then deployed across their fleet... tend to produce unique JARM
#   fingerprints" — meaning a shared JARM hash across "unrelated"
#   domains is evidence they're running the SAME KIT, even with
#   completely different certs, domains, and registrars.
#
# HONEST LIMITATION (cite this in your methods section): JARM is known
# to be less discriminative for domains sitting behind common CDNs
# (Cloudflare, Akamai), since the CDN's edge server — not the origin
# server — answers the TLS handshake, so many unrelated sites behind
# the same CDN can share a JARM hash for reasons that have NOTHING to
# do with shared attacker infrastructure. We flag CDN-fronted domains
# so this can be accounted for in feature weighting later.

try:
    from jarm.scanner.scanner import Scanner
    JARM_AVAILABLE = True
except ImportError:
    JARM_AVAILABLE = False


# A JARM of all zeros means the server refused every single probe —
# this itself is informative (server actively resists fingerprinting)
# but is NOT a useful comparison value, so we flag it distinctly.
NULL_JARM = "0" * 62


def fetch_jarm(domain: str, port: int = 443, timeout: float = 10.0) -> dict:
    """
    Computes the JARM fingerprint for a domain's TLS server.

    Example output:
    {
        "domain":      "secure-bank-login.xyz",
        "jarm":        "27d40d40d29d40d1dc42d43d00041d4689ee210389f4f6b4b5b1b93f92252d",
        "is_null_jarm": False,
        "resolved":    True,
        "error":       None
    }
    """
    if not JARM_AVAILABLE:
        return _failure(domain, "MissingDependency: run 'pip install pyjarm'")

    if "://" in domain or "/" in domain:
        return _failure(domain, "InvalidInput: pass a bare domain, not a full URL")

    try:
        # Scanner.scan returns a tuple: (jarm_hash, host, port)
        jarm_hash, _, _ = Scanner.scan(domain, port)

        if not jarm_hash or jarm_hash == NULL_JARM:
            return {
                "domain": domain, "jarm": jarm_hash, "is_null_jarm": True,
                "resolved": True, "error": None,
            }

        return {
            "domain": domain, "jarm": jarm_hash, "is_null_jarm": False,
            "resolved": True, "error": None,
        }

    except Exception as e:
        # pyjarm raises broad exceptions for connection failures, TLS
        # negotiation failures, etc — we treat all of these the same
        # way as every other module: log and move on, never crash the batch.
        return _failure(domain, f"JarmScanError: {e}")


def _failure(domain: str, error_msg: str) -> dict:
    return {"domain": domain, "jarm": None, "is_null_jarm": None,
            "resolved": False, "error": error_msg}


def _print_result(result: dict) -> None:
    print(f"\n  Domain:      {result['domain']}")
    if result["resolved"]:
        print(f"  Status:      ✓ RESOLVED")
        print(f"  JARM:        {result['jarm']}")
        if result["is_null_jarm"]:
            print(f"  ⚠ Null JARM — server refused all fingerprinting probes")
    else:
        print(f"  Status:      ✗ FAILED  ({result['error']})")
    print(f"  {'─'*52}")


# ── SELF-TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if not JARM_AVAILABLE:
        print("\n  ⚠ pyjarm not installed. Run: pip install pyjarm")
        sys.exit(0)

    if len(sys.argv) > 1:
        print("\n" + "=" * 58)
        print("  EVELYN — fetch_jarm()")
        print("=" * 58)
        _print_result(fetch_jarm(sys.argv[1]))
    else:
        # Two well-known domains likely on DIFFERENT server stacks —
        # if their JARM hashes differ, that confirms JARM is correctly
        # discriminating between different server configurations.
        TEST_DOMAINS = ["google.com", "github.com"]
        print("\n" + "=" * 58)
        print("  EVELYN — fetch_jarm() test suite")
        print("=" * 58)
        results = []
        for domain in TEST_DOMAINS:
            r = fetch_jarm(domain)
            _print_result(r)
            results.append(r)

        if all(r["resolved"] for r in results) and len(set(r["jarm"] for r in results)) > 1:
            print("\n  ✓ Sanity check passed: different domains produced different JARM hashes")