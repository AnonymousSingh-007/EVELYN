# src/pipeline/fetch_geo.py
#
# PURPOSE: Given an IP address, find its approximate geographic
# location (country, city, lat/long). New node type:
# IP --located-in--> Country/City
#
# Why this matters: phishing campaigns targeting a specific country's
# bank (e.g. an Indian bank) but hosted on servers in an unrelated
# country is itself a signal. Also, attackers running multiple
# campaigns often cluster their hosting in 2-3 favourite countries
# regardless of who they're targeting — this becomes a campaign
# fingerprint feature just like ASN.

import requests


# ip-api.com offers a free tier (no API key needed) for non-commercial
# research use, rate-limited to 45 requests/minute. For a research
# project processing hundreds of IPs, you respect this rate limit by
# adding a small delay between calls in your batch-processing script
# later — not in this file itself.
IP_API_URL = "http://ip-api.com/json/{ip}"


def fetch_geo(ip: str, timeout: float = 8.0) -> dict:
    """
    Looks up geolocation data for an IP address.

    Example output:
    {
        "ip":        "185.220.101.47",
        "country":   "Germany",
        "city":      "Frankfurt",
        "lat":       50.1109,
        "lon":       8.6821,
        "isp":       "Some Hosting GmbH",
        "resolved":  True,
        "error":     None
    }
    """

    if not _looks_like_ip(ip):
        return _failure(ip, "InvalidInput: pass an IP address, e.g. '185.220.101.47'")

    try:
        response = requests.get(
            IP_API_URL.format(ip=ip),
            timeout=timeout,
            params={"fields": "status,message,country,city,lat,lon,isp"},
        )
        response.raise_for_status()
        data = response.json()

        # ip-api.com returns {"status": "fail", "message": "..."} on errors
        # INSTEAD of an HTTP error code — so we have to check this field
        # ourselves rather than relying on raise_for_status() alone.
        # This is a common pattern with free geo-IP APIs: success/failure
        # is reported INSIDE a 200 OK body, not via HTTP status.
        if data.get("status") == "fail":
            return _failure(ip, f"APIError: {data.get('message', 'unknown')}")

        return {
            "ip":       ip,
            "country":  data.get("country"),
            "city":     data.get("city"),
            "lat":      data.get("lat"),
            "lon":      data.get("lon"),
            "isp":      data.get("isp"),
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
    parts = value.split(".")
    if len(parts) != 4:
        return False
    for part in parts:
        if not part.isdigit() or not (0 <= int(part) <= 255):
            return False
    return True


def _failure(ip: str, error_msg: str) -> dict:
    return {
        "ip": ip, "country": None, "city": None, "lat": None,
        "lon": None, "isp": None, "resolved": False, "error": error_msg,
    }


def _print_result(result: dict) -> None:
    print(f"\n  IP:        {result['ip']}")
    if result["resolved"]:
        print(f"  Status:    ✓ RESOLVED")
        print(f"  Location:  {result['city']}, {result['country']}")
        print(f"  Coords:    ({result['lat']}, {result['lon']})")
        print(f"  ISP:       {result['isp']}")
    else:
        print(f"  Status:    ✗ FAILED  ({result['error']})")
    print(f"  {'─'*52}")


# ── SELF-TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        print("\n" + "=" * 58)
        print("  EVELYN — fetch_geo()")
        print("=" * 58)
        _print_result(fetch_geo(sys.argv[1]))
    else:
        TEST_IPS = ["8.8.8.8", "1.1.1.1"]
        print("\n" + "=" * 58)
        print("  EVELYN — fetch_geo() test suite")
        print("=" * 58)
        for ip in TEST_IPS:
            _print_result(fetch_geo(ip))