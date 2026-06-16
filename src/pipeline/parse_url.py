# src/pipeline/parse_url.py
#
# PURPOSE: Take a raw URL string and break it into clean parts.
# This is the entry point of the entire EVELYN pipeline.
# Every URL we ever process passes through this function first.

import tldextract  # This library knows about every TLD (.com, .xyz, .co.in, etc.)
from urllib.parse import urlparse  # Built into Python — parses URL structure


def parse_url(url: str) -> dict:
    """
    Takes a raw URL and returns a dictionary of its parts.

    Example input:  "https://secure-bank-login.xyz/verify?id=123"
    Example output: {
        "raw_url":   "https://secure-bank-login.xyz/verify?id=123",
        "scheme":    "https",
        "subdomain": "secure-bank-login",   <- wait, explained below
        "domain":    "xyz",                 <- the registered domain
        "suffix":    "xyz",                 <- the TLD
        "full_domain": "secure-bank-login.xyz",
        "path":      "/verify",
        "query":     "id=123",
        "is_ip":     False
    }
    """

    # urlparse splits the URL by its structural parts
    # "https://secure-bank-login.xyz/verify?id=123"
    #  scheme   netloc (host)         path   query
    parsed = urlparse(url)

    # tldextract is smarter than urlparse for domain splitting
    # It knows that "secure-bank-login.xyz" has:
    #   subdomain = "" (nothing before the domain)
    #   domain    = "secure-bank-login"
    #   suffix    = "xyz"
    # And that "login.secure.bank.co.uk" has:
    #   subdomain = "login.secure"
    #   domain    = "bank"
    #   suffix    = "co.uk"   <- tldextract knows this is ONE suffix
    extracted = tldextract.extract(url)

    # Check if the host is a raw IP address like "http://185.220.101.47/login"
    # Phishing domains hosted on raw IPs with no domain name is a signal
    # We detect this by checking: does urlparse give us a netloc that
    # looks like numbers-and-dots only?
    host = parsed.netloc  # e.g. "secure-bank-login.xyz" or "185.220.101.47"
    is_ip = _is_ip_address(host)

    # Build and return the result as a dictionary
    # A dictionary is the right structure here because each URL
    # produces exactly one set of named parts — no ordering needed
    return {
        "raw_url":     url,
        "scheme":      parsed.scheme,           # "http" or "https"
        "subdomain":   extracted.subdomain,     # "" or "www" or "login.secure"
        "domain":      extracted.domain,        # "secure-bank-login"
        "suffix":      extracted.suffix,        # "xyz" or "co.uk"
        "full_domain": extracted.registered_domain,  # "secure-bank-login.xyz"
        "path":        parsed.path,             # "/verify"
        "query":       parsed.query,            # "id=123"
        "is_ip":       is_ip,
    }


def _is_ip_address(host: str) -> bool:
    """
    Returns True if host looks like a raw IP address (e.g. "185.220.101.47").
    The underscore prefix _ means: this is a helper, not meant to be
    called from outside this file. It's only for parse_url() to use.
    """

    # Remove port number if present: "185.220.101.47:8080" → "185.220.101.47"
    host = host.split(":")[0]

    # Split by dots: ["185", "220", "101", "47"]
    parts = host.split(".")

    # An IPv4 address always has exactly 4 parts
    if len(parts) != 4:
        return False

    # Every part must be a number between 0 and 255
    for part in parts:
        if not part.isdigit():        # isdigit() → True if all characters are 0-9
            return False
        if not (0 <= int(part) <= 255):
            return False

    return True


# ─── SELF-TEST ────────────────────────────────────────────────────────────────
# This block only runs when you execute THIS file directly:
#   python src/pipeline/parse_url.py
# It does NOT run when you import parse_url from another file.
# This is the standard Python pattern for "run me to test me".

if __name__ == "__main__":

    # A list of test URLs — mix of phishing patterns and edge cases
    test_urls = [
        "https://secure-bank-login.xyz/verify?id=123",   # classic phishing
        "http://185.220.101.47/login",                   # raw IP, no domain
        "https://www.google.com/search?q=test",          # benign, subdomain
        "http://paypal-secure.update.co.uk/account",     # multi-level TLD
        "https://login.microsoftonline.com.phish.cc/",   # subdomain abuse
    ]

    print("=" * 60)
    print("EVELYN — parse_url() self-test")
    print("=" * 60)

    for url in test_urls:
        result = parse_url(url)
        print(f"\nURL:        {result['raw_url']}")
        print(f"  scheme:     {result['scheme']}")
        print(f"  subdomain:  '{result['subdomain']}'")
        print(f"  domain:     '{result['domain']}'")
        print(f"  suffix:     '{result['suffix']}'")
        print(f"  full_domain:{result['full_domain']}")
        print(f"  path:       {result['path']}")
        print(f"  query:      '{result['query']}'")
        print(f"  is_ip:      {result['is_ip']}")