# src/pipeline/fetch_asn.py
#
# PURPOSE: Given an IP address, find out which ASN (Autonomous System
# Number) it belongs to — i.e. which hosting provider / ISP actually
# owns and routes that IP. This is a NEW node type in our graph:
# IP --hosted-in--> ASN
#
# Why this matters: attackers tend to return to the SAME few hosting
# providers repeatedly, because they need providers who accept their
# payment method and don't ask many questions. Seeing 50 "different"
# phishing domains all hosted in the same ASN is a very strong signal
# of a single operator, even if every domain/IP/cert looks unrelated.

import requests


# We use the free RDAP (Registration Data Access Protocol) service run by
# the RIPE NCC, which works for ASN lookups across ALL regional registries
# (ARIN, RIPE, APNIC, etc.) — not just European IPs. RDAP is the modern,
# structured-JSON replacement for old-school WHOIS, so we get clean data
# without the messy text-parsing problem fetch_whois.py had to deal with.
RDAP_URL = "https://rdap.org/ip/{ip}"


def fetch_asn(ip: str, timeout: float = 8.0) -> dict:
    """
    Looks up the ASN (hosting provider) for a given IP address.

    Example output:
    {
        "ip":          "185.220.101.47",
        "asn":         "AS4766",
        "as_name":     "Korea Telecom",
        "country":     "KR",
        "resolved":    True,
        "error":       None
    }
    """

    # GUARD CLAUSE: this function takes an IP, not a domain or URL.
    # A quick sanity check — does it look roughly like an IPv4 address?
    if not _looks_like_ip(ip):
        return _failure(ip, "InvalidInput: pass an IP address, e.g. '185.220.101.47'")

    try:
        response = requests.get(
            RDAP_URL.format(ip=ip),
            timeout=timeout,
            headers={"User-Agent": "EVELYN-research-tool/0.1"},
        )
        response.raise_for_status()
        data = response.json()

        # RDAP responses nest the ASN info inside an "autnums" related-link
        # structure OR sometimes directly. The field names vary slightly
        # by registry, so we check a few likely places. This kind of
        # defensive multi-path lookup is normal when consuming RDAP data
        # because each regional registry implements it slightly differently.
        asn = data.get("autnum") or data.get("handle")

        # The "name" field is usually the organisation/ISP name
        as_name = data.get("name")

        # Country is often in the "country" field directly
        country = data.get("country")

        # If we didn't find an ASN number directly, look in remarks/entities
        # (this is the messier fallback path — RDAP for IP ranges doesn't
        # always include autnum directly, sometimes you need a SEPARATE
        # query to a dedicated ASN-lookup endpoint for full reliability)
        if not asn:
            entities = data.get("entities", [])
            for entity in entities:
                if entity.get("roles") and "registrant" in entity.get("roles", []):
                    as_name = as_name or entity.get("handle")

        return {
            "ip":       ip,
            "asn":      asn,
            "as_name":  as_name,
            "country":  country,
            "resolved": True,
            "error":    None,
        }

    except requests.exceptions.Timeout:
        return _failure(ip, "Timeout")
    except requests.exceptions.RequestException as e:
        return _failure(ip, f"RequestError: {e}")
    except ValueError as e:
        return _failure(ip, f"InvalidJSON: {e}")
    except Exception as e:
        return _failure(ip, f"UnknownError: {e}")


def _looks_like_ip(value: str) -> bool:
    """Quick structural check — 4 dot-separated numeric parts, 0-255 each."""
    parts = value.split(".")
    if len(parts) != 4:
        return False
    for part in parts:
        if not part.isdigit() or not (0 <= int(part) <= 255):
            return False
    return True


def _failure(ip: str, error_msg: str) -> dict:
    return {
        "ip": ip, "asn": None, "as_name": None,
        "country": None, "resolved": False, "error": error_msg,
    }


def _print_result(result: dict) -> None:
    print(f"\n  IP:        {result['ip']}")
    if result["resolved"]:
        print(f"  Status:    ✓ RESOLVED")
        print(f"  ASN:       {result['asn']}")
        print(f"  Org name:  {result['as_name']}")
        print(f"  Country:   {result['country']}")
    else:
        print(f"  Status:    ✗ FAILED  ({result['error']})")
    print(f"  {'─'*52}")


# ── SELF-TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        print("\n" + "=" * 58)
        print("  EVELYN — fetch_asn()")
        print("=" * 58)
        _print_result(fetch_asn(sys.argv[1]))
    else:
        TEST_IPS = [
            "8.8.8.8",          # Google public DNS — well-known ASN
            "1.1.1.1",          # Cloudflare public DNS — well-known ASN
        ]
        print("\n" + "=" * 58)
        print("  EVELYN — fetch_asn() test suite")
        print("=" * 58)
        for ip in TEST_IPS:
            _print_result(fetch_asn(ip))