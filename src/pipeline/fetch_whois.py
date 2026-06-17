# src/pipeline/fetch_whois.py
#
# PURPOSE: Take a domain and fetch its WHOIS registration data.
# This gives us the REGISTRAR node in our graph: domain --registered-by--> registrar
# It also gives us domain AGE — one of the strongest phishing signals that exists.
# Phishing domains are overwhelmingly registered within days of being used in attacks.
# Legitimate domains (google.com, your bank's real site) are years or decades old.

import whois          # this is the python-whois package
from datetime import datetime, timezone


def fetch_whois(domain: str) -> dict:
    """
    ... (existing docstring stays the same) ...
    """

    # GUARD CLAUSE: reject anything that isn't a bare domain.
    # This function must ONLY receive output from parse_url()'s
    # "full_domain" field — never a raw URL with scheme/path/query.
    # We enforce this here so the bug surfaces immediately and loudly,
    # instead of silently working by accident (like x.com/SpaceX did)
    # and then breaking mysteriously on a different input later.
    if "://" in domain or "/" in domain:
        return {
            "domain":          domain,
            "registrar":       None,
            "creation_date":   None,
            "domain_age_days": None,
            "resolved":        False,
            "error":           "InvalidInput: pass a bare domain (e.g. 'x.com'), not a full URL. Use parse_url() first.",
        }


    try:
        # whois.whois() sends the actual WHOIS protocol query.
        # WHOIS is an OLD protocol (1982) — every registrar runs a server
        # that responds to it, but the RESPONSE FORMAT IS NOT STANDARDIZED.
        # This is why the python-whois library has to do a lot of messy
        # text-parsing internally — different registrars format their
        # responses completely differently. This is also why your error
        # handling here needs to be more forgiving than resolve_dns.py.
        w = whois.whois(domain)

        # w.registrar can be None, a string, or (annoyingly) sometimes
        # a list of strings if multiple records were returned. We
        # normalize all three cases into one clean string or None.
        registrar = _normalize_field(w.registrar)

        # w.creation_date has the SAME problem — it can be a single
        # datetime object, a list of datetime objects, or None.
        creation_date = _normalize_field(w.creation_date)

        # Calculate domain age in days — this is gold for phishing detection.
        # A domain registered 3 days ago that's already hosting a "bank login"
        # page is about as suspicious as it gets.
        domain_age_days = None
        if creation_date is not None and isinstance(creation_date, datetime):
            now = datetime.now(timezone.utc)
            # Some WHOIS responses return naive datetimes (no timezone info).
            # We have to handle both cases or this comparison crashes.
            if creation_date.tzinfo is None:
                creation_date = creation_date.replace(tzinfo=timezone.utc)
            domain_age_days = (now - creation_date).days

        return {
            "domain":          domain,
            "registrar":       registrar,
            "creation_date":   creation_date.strftime("%Y-%m-%d") if isinstance(creation_date, datetime) else None,
            "domain_age_days": domain_age_days,
            "resolved":        True,
            "error":           None,
        }

    except Exception as e:
        # WHOIS fails A LOT in practice: rate limiting, registrars that
        # don't respond, privacy-protected records, malformed responses.
        # We catch broadly here because the python-whois library doesn't
        # give us specific exception types to catch individually like
        # dnspython does — it's a much messier ecosystem.
        return {
            "domain":          domain,
            "registrar":       None,
            "creation_date":   None,
            "domain_age_days": None,
            "resolved":        False,
            "error":           f"WhoisLookupError: {e}",
        }


def _normalize_field(value):
    """
    WHOIS libraries sometimes return a list when multiple records exist
    (e.g. registrar appears twice in the raw response). We always want
    ONE clean value, so we take the first item if it's a list.
    """
    if isinstance(value, list):
        return value[0] if len(value) > 0 else None
    return value


def _print_result(result: dict) -> None:
    """Pretty-print one fetch_whois() result to the terminal."""
    print(f"\n  Domain:          {result['domain']}")
    if result["resolved"]:
        print(f"  Status:          ✓ RESOLVED")
        print(f"  Registrar:       {result['registrar']}")
        print(f"  Creation date:   {result['creation_date']}")
        if result["domain_age_days"] is not None:
            age = result["domain_age_days"]
            print(f"  Domain age:      {age} days", end="")
            if age < 30:
                print("  ⚠ VERY NEW — strong phishing signal")
            elif age < 365:
                print("  ⚠ under 1 year old")
            else:
                print()
        else:
            print(f"  Domain age:      unknown")
    else:
        print(f"  Status:          ✗ FAILED  ({result['error']})")
    print(f"  {'─'*52}")


# ── SELF-TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        print("\n" + "=" * 58)
        print("  EVELYN — fetch_whois()")
        print("=" * 58)
        _print_result(fetch_whois(sys.argv[1]))

    else:
        TEST_DOMAINS = [
            "google.com",      # extremely old domain, should show large age
            "github.com",      # old, established domain
        ]

        print("\n" + "=" * 58)
        print("  EVELYN — fetch_whois() test suite")
        print("=" * 58)
        print("  NOTE: WHOIS queries are slower than DNS — this may take a few seconds")

        for domain in TEST_DOMAINS:
            _print_result(fetch_whois(domain))