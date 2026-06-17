# src/pipeline/resolve_dns.py
#
# PURPOSE: Take a domain name and resolve it to its actual IP address(es)
# using a real DNS query. This is the second node-type in our graph:
# domain --resolves-to--> IP
#
# This is your first file that reaches out to the real internet.
# Errors here are normal and expected — DNS fails constantly in the
# real world (timeouts, NXDOMAIN, dead domains). We handle that gracefully.

import dns.resolver   # part of the dnspython package
import dns.exception


def resolve_dns(domain: str, timeout: float = 3.0) -> dict:
    """
    Resolves a domain name to its IPv4 address(es) via an A record query.

    Example input:  "google.com"
    Example output: {
        "domain":     "google.com",
        "ips":        ["142.250.193.46"],
        "ttl":        300,
        "resolved":   True,
        "error":      None
    }

    On failure (domain doesn't exist, times out, etc.):
    {
        "domain":     "this-domain-does-not-exist-xyz123.com",
        "ips":        [],
        "ttl":        None,
        "resolved":   False,
        "error":      "NXDOMAIN"
    }
    """

    # dns.resolver.Resolver() creates a fresh resolver object.
    # We set a short timeout because phishing domains are often dead,
    # parked, or hosted on slow/sketchy infrastructure. Without a timeout,
    # one bad domain could hang your entire pipeline for 30+ seconds.
    resolver = dns.resolver.Resolver()
    resolver.timeout = timeout      # how long to wait for ONE DNS server
    resolver.lifetime = timeout     # how long to wait in total (all retries)

    try:
        # resolve() sends the actual DNS query over the network.
        # "A" means we want IPv4 address records specifically.
        # (There's also "AAAA" for IPv6 — we'll add that later if needed.)
        answer = resolver.resolve(domain, "A")

        # answer is iterable — a domain can have MULTIPLE IPs
        # (this happens with load balancing, CDNs, and — importantly —
        # phishing campaigns that round-robin across several servers)
        ips = [rdata.address for rdata in answer]

        # TTL = Time To Live, in seconds. This tells us how long this
        # DNS record is supposed to be cached. Phishing infrastructure
        # often uses very LOW TTLs (60s, 300s) so they can switch IPs
        # fast if they get blocked. This is itself a signal we can use
        # later as a graph edge weight or node feature.
        ttl = answer.rrset.ttl

        return {
            "domain":   domain,
            "ips":      ips,
            "ttl":      ttl,
            "resolved": True,
            "error":    None,
        }

    except dns.resolver.NXDOMAIN:
        # NXDOMAIN = "this domain does not exist at all"
        # Common for phishing domains that have been taken down
        return _failure(domain, "NXDOMAIN")

    except dns.resolver.NoAnswer:
        # The domain exists but has no A record (maybe only has MX or TXT)
        return _failure(domain, "NoAnswer")

    except dns.exception.Timeout:
        # The DNS server didn't respond in time
        return _failure(domain, "Timeout")

    except Exception as e:
        # Catch-all for anything unexpected (network down, malformed
        # domain string, etc.). We never want resolve_dns() to CRASH
        # your pipeline — a single bad domain should not stop you from
        # processing the other 999 in your dataset.
        return _failure(domain, f"UnknownError: {e}")


def _failure(domain: str, error_msg: str) -> dict:
    """
    Helper to build a consistent failure response.
    Keeping this in one place means every failure path returns
    the exact same shape — no surprises downstream in build_graph.py.
    """
    return {
        "domain":   domain,
        "ips":      [],
        "ttl":      None,
        "resolved": False,
        "error":    error_msg,
    }


def _print_result(result: dict) -> None:
    """
    Pretty-print one resolve_dns() result to the terminal.

    NOTE ON TTL THRESHOLD:
    A low TTL alone is NOT a phishing signal — major CDNs (Google,
    Cloudflare, Akamai) use low TTLs (30-300s) for legitimate global
    load-balancing. We lowered our "suspicious" threshold to <60s and
    relabeled it clearly as "CDN-or-evasion" rather than implying guilt.
    The real signal only emerges when LOW TTL is combined with other
    features (new domain age, suspicious TLD, no CDN ASN) in build_graph.py.
    This function just reports the raw fact — it does not judge intent.
    """
    print(f"\n  Domain:    {result['domain']}")
    if result["resolved"]:
        print(f"  Status:    ✓ RESOLVED")
        print(f"  IP(s):     {', '.join(result['ips'])}")
        print(f"  TTL:       {result['ttl']}s", end="")
        if result["ttl"] is not None and result["ttl"] < 60:
            print("  ℹ very low TTL — could be CDN load-balancing OR evasive infra")
        else:
            print()
    else:
        print(f"  Status:    ✗ FAILED  ({result['error']})")
    print(f"  {'─'*52}")

# ── SELF-TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # User supplied a domain directly: python resolve_dns.py google.com
        print("\n" + "=" * 58)
        print("  EVELYN — resolve_dns()")
        print("=" * 58)
        _print_result(resolve_dns(sys.argv[1]))

    else:
        # Built-in test domains: benign CDN-backed sites (to show low-TTL-is-normal),
        # a dead domain (NXDOMAIN path), and a cheap-TLD domain (TLD signal, separate
        # from TTL — this is what actually correlates with phishing in research data)
        TEST_DOMAINS = [
            "google.com",                                              # benign, CDN, low TTL — proves TTL alone is noise
            "github.com",                                              # benign, CDN, low TTL
            "this-domain-absolutely-does-not-exist-evelyn-test.com",   # NXDOMAIN path
            "freenom.com",                                             # known free-domain provider, abused historically
        ]

        print("\n" + "=" * 58)
        print("  EVELYN — resolve_dns() test suite")
        print("=" * 58)

        for domain in TEST_DOMAINS:
            _print_result(resolve_dns(domain))