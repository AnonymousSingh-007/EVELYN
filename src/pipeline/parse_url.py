# src/pipeline/parse_url.py
#
# PURPOSE: Take a raw URL string and break it into clean parts.
# This is the entry point of the entire EVELYN pipeline.
# Every URL we ever process passes through this function first.

import tldextract
from urllib.parse import urlparse


def parse_url(url: str) -> dict:
    """
    Takes a raw URL and returns a dictionary of its parts.

    Example input:  "https://secure-bank-login.xyz/verify?id=123"
    Example output: {
        "raw_url":     "https://secure-bank-login.xyz/verify?id=123",
        "scheme":      "https",
        "subdomain":   "",
        "domain":      "secure-bank-login",
        "suffix":      "xyz",
        "full_domain": "secure-bank-login.xyz",
        "path":        "/verify",
        "query":       "id=123",
        "is_ip":       False
    }
    """

    parsed    = urlparse(url)
    extracted = tldextract.extract(url)

    # tldextract 5.x renamed registered_domain → top_domain_under_public_suffix
    # We try the new name first; fall back to the old name for older installs.
    # This way the code works regardless of which version someone has.
    if hasattr(extracted, "top_domain_under_public_suffix"):
        full_domain = extracted.top_domain_under_public_suffix
    else:
        full_domain = extracted.registered_domain   # pre-5.x

    host  = parsed.netloc
    is_ip = _is_ip_address(host)

    return {
        "raw_url":     url,
        "scheme":      parsed.scheme,
        "subdomain":   extracted.subdomain,
        "domain":      extracted.domain,
        "suffix":      extracted.suffix,
        "full_domain": full_domain,
        "path":        parsed.path,
        "query":       parsed.query,
        "is_ip":       is_ip,
    }


def _is_ip_address(host: str) -> bool:
    """
    Returns True if host is a raw IPv4 address like "185.220.101.47".
    Strips port number first if present: "185.220.101.47:8080" → "185.220.101.47"
    """
    host  = host.split(":")[0]
    parts = host.split(".")

    if len(parts) != 4:
        return False

    for part in parts:
        if not part.isdigit():
            return False
        if not (0 <= int(part) <= 255):
            return False

    return True


def _print_result(result: dict) -> None:
    """
    Pretty-prints one parse result to the terminal.
    Flags suspicious TLDs and raw IPs inline.
    Extracted into its own function so both the user-URL path
    and the test path can call it without duplicating code.
    """
    # TLDs that appear disproportionately in phishing campaigns.
    # Source: APWG eCrime reports 2022-2024.
    SUSPICIOUS_TLDS = {
        "xyz", "cc", "tk", "ml", "ga", "cf", "gq",
        "top", "pw", "work", "click", "link", "online"
    }

    ip_flag  = "  ⚠  RAW IP — no domain name"   if result["is_ip"] else ""
    tld_flag = "  ⚠  SUSPICIOUS TLD"             if result["suffix"] in SUSPICIOUS_TLDS else ""

    print(f"\n  URL:         {result['raw_url']}")
    print(f"  scheme:      {result['scheme']}")
    print(f"  subdomain:   '{result['subdomain']}'")
    print(f"  domain:      '{result['domain']}'")
    print(f"  suffix:      '{result['suffix']}'{tld_flag}")
    print(f"  full_domain: {result['full_domain']}")
    print(f"  path:        {result['path']}")
    print(f"  query:       '{result['query']}'")
    print(f"  is_ip:       {result['is_ip']}{ip_flag}")
    print(f"  {'─'*52}")


# ── SELF-TEST ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # If user supplied a URL as a command-line argument, analyse ONLY that URL.
    # This keeps the output clean and focused when running from the terminal.
    # Usage: python src/pipeline/parse_url.py https://evil-phish.xyz/login
    if len(sys.argv) > 1:
        user_url = sys.argv[1]
        print("\n" + "=" * 58)
        print("  EVELYN — parse_url()")
        print("=" * 58)
        _print_result(parse_url(user_url))

    # No argument supplied → run the built-in test suite so we can verify
    # that the function handles all edge cases correctly.
    else:
        TEST_URLS = [
            "https://secure-bank-login.xyz/verify?id=123",   # suspicious TLD
            "http://185.220.101.47/login",                   # raw IP
            "https://www.google.com/search?q=test",          # benign with subdomain
            "http://paypal-secure.update.co.uk/account",     # multi-level TLD
            "https://login.microsoftonline.com.phish.cc/",   # subdomain abuse
        ]

        print("\n" + "=" * 58)
        print("  EVELYN — parse_url() test suite")
        print("=" * 58)

        for url in TEST_URLS:
            _print_result(parse_url(url))