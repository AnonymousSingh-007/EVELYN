# src/pipeline/check_shared_hosting.py
#
# PURPOSE: Given an IP, find OTHER domains that resolve to (or have
# historically resolved to) that same IP. This directly builds the
# IP --hosts--> Domain edges that are central to your whole thesis:
# "different" phishing domains sharing one IP is a primary campaign signal.
#
# We use HackerTarget's free reverse-IP API, which queries passive DNS
# history. Like crt.sh, this is a free community tool — expect occasional
# rate limiting and treat results as best-effort, not guaranteed-complete.

import requests


HACKERTARGET_URL = "https://api.hackertarget.com/reverseiplookup/"


def check_shared_hosting(ip: str, timeout: float = 10.0) -> dict:
    """
    Finds other domains hosted on the same IP address.

    Example output:
    {
        "ip":             "185.220.101.47",
        "co_hosted_domains": ["other-phish.xyz", "third-site.cc"],
        "count":          2,
        "resolved":       True,
        "error":          None
    }
    """

    if not _looks_like_ip(ip):
        return _failure(ip, "InvalidInput: pass an IP address, e.g. '185.220.101.47'")

    try:
        response = requests.get(
            HACKERTARGET_URL,
            params={"q": ip},
            timeout=timeout,
        )
        response.raise_for_status()

        # HackerTarget's free API returns PLAIN TEXT, not JSON — one
        # domain per line. It also returns specific text messages for
        # error conditions (rate limit, no results) INSTEAD OF an error
        # HTTP code, similar to the ip-api.com situation in fetch_geo.py.
        # This inconsistency across free APIs is exactly why every single
        # one of our pipeline files needs its own custom error detection —
        # there's no universal pattern, you have to read each API's docs.
        text = response.text.strip()

        if "API count exceeded" in text or "error" in text.lower():
            return _failure(ip, f"APIError: {text}")

        if not text or text == "":
            domains = []
        else:
            domains = [line.strip() for line in text.split("\n") if line.strip()]

        return {
            "ip":                ip,
            "co_hosted_domains": domains,
            "count":             len(domains),
            "resolved":          True,
            "error":             None,
        }

    except requests.exceptions.Timeout:
        return _failure(ip, "Timeout")
    except requests.exceptions.RequestException as e:
        return _failure(ip, f"RequestError: {e}")
    except Exception as e:
        return _failure(ip, f"UnknownError: {e}")


def _looks_like_ip(value: str) -> bool:
    parts = value.split(".")
    if len(parts) != 4:
        return False
    for part in parts:
        if not part.isdigit() or not (0 <= int(part) <= 255):
            return False
    return True


def _failure(ip: str, error_msg: str) -> dict:
    return {
        "ip": ip, "co_hosted_domains": [], "count": 0,
        "resolved": False, "error": error_msg,
    }


def _print_result(result: dict) -> None:
    print(f"\n  IP:        {result['ip']}")
    if result["resolved"]:
        print(f"  Status:    ✓ RESOLVED")
        print(f"  Co-hosted: {result['count']} domain(s)")
        for d in result["co_hosted_domains"][:10]:
            print(f"      → {d}")
        if result["count"] > 10:
            print(f"      ... and {result['count'] - 10} more")
        if result["count"] > 20:
            print(f"  ⚠ HIGH DENSITY — likely shared/bulk hosting infrastructure")
    else:
        print(f"  Status:    ✗ FAILED  ({result['error']})")
    print(f"  {'─'*52}")


# ── SELF-TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        print("\n" + "=" * 58)
        print("  EVELYN — check_shared_hosting()")
        print("=" * 58)
        _print_result(check_shared_hosting(sys.argv[1]))
    else:
        TEST_IPS = ["8.8.8.8"]
        print("\n" + "=" * 58)
        print("  EVELYN — check_shared_hosting() test suite")
        print("=" * 58)
        for ip in TEST_IPS:
            _print_result(check_shared_hosting(ip))