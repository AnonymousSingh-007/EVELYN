# src/pipeline/fetch_page_meta.py
#
# PURPOSE: Fetch a domain's actual web page ONCE and extract three
# independent signals from that single request:
#   1. HTTP response headers  — server software, security headers,
#      redirect chain (where did we actually end up?)
#   2. Favicon hash            — phishing KITS (pre-packaged attack
#      toolkits sold/shared among attackers) very often reuse the
#      exact same favicon.ico across hundreds of "different" cloned
#      sites, because operators rarely bother customizing it. A
#      matching favicon hash across "unrelated" domains is a strong,
#      free, purely passive campaign-linking signal.
#   3. Page title + meta description — a cheap text fingerprint of
#      what the page CLAIMS to be (e.g. title says "PayPal — Log In"
#      but the domain has nothing to do with PayPal).
#
# WHY ONE REQUEST FOR ALL THREE: each of these signals could be its
# own file calling requests.get() independently, but that means 3x
# the network calls, 3x the chance of timeout, and 3x the load on
# whatever server we're investigating (which may itself be malicious
# infrastructure — minimizing our footprint there is good practice).
# One fetch, three derived signals.
#
# SECURITY NOTE: we are visiting a potentially malicious URL directly.
# We do NOT execute any JavaScript (we only fetch raw HTML via requests,
# never render in a browser), we set a strict timeout, and we never
# follow more than a bounded number of redirects. This is read-only,
# passive reconnaissance — the same exposure level as a normal browser
# visit, which is what a real victim would experience anyway.

import requests
import hashlib
import re
from urllib.parse import urljoin


# Security-relevant headers we check for PRESENCE/ABSENCE — legitimate
# modern sites (especially banks, the most-impersonated targets) almost
# always set these. Their absence is a weak-but-free signal.
SECURITY_HEADERS_OF_INTEREST = [
    "strict-transport-security",
    "content-security-policy",
    "x-frame-options",
    "x-content-type-options",
]

MAX_REDIRECTS = 5
REQUEST_TIMEOUT = 8.0
USER_AGENT = "Mozilla/5.0 (EVELYN-research-tool; academic project; passive analysis only)"


def fetch_page_meta(domain: str, use_https: bool = True) -> dict:
    """
    Fetches a domain's homepage once and extracts headers, favicon
    hash, and page metadata in a single pass.

    Example output:
    {
        "domain":            "secure-bank-login.xyz",
        "final_url":         "https://secure-bank-login.xyz/login",
        "redirect_count":    1,
        "redirect_chain":    ["https://secure-bank-login.xyz/", "https://.../login"],
        "status_code":       200,
        "server_header":     "nginx",
        "security_headers_present": ["x-content-type-options"],
        "security_headers_missing": ["strict-transport-security", ...],
        "favicon_hash":      "a3f9b2c1..." or None,
        "page_title":        "Secure Login - Verify Your Account",
        "meta_description":  "...",
        "brand_keywords_in_title": ["paypal"],   # see _detect_brand_keywords
        "resolved":          True,
        "error":             None
    }
    """
    if "://" in domain or "/" in domain:
        return _failure(domain, "InvalidInput: pass a bare domain, not a full URL")

    scheme = "https" if use_https else "http"
    url = f"{scheme}://{domain}/"

    try:
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
            allow_redirects=True,
            verify=True,   # don't silently accept broken/self-signed certs here;
                            # fetch_ssl_meta.py already characterizes cert validity
        )

        redirect_chain = [r.url for r in response.history] + [response.url]
        redirect_count = len(response.history)

        headers_lower = {k.lower(): v for k, v in response.headers.items()}
        server_header = headers_lower.get("server")

        present = [h for h in SECURITY_HEADERS_OF_INTEREST if h in headers_lower]
        missing = [h for h in SECURITY_HEADERS_OF_INTEREST if h not in headers_lower]

        html = response.text

        favicon_hash = _fetch_favicon_hash(response.url, html)
        page_title = _extract_title(html)
        meta_description = _extract_meta_description(html)
        brand_keywords = _detect_brand_keywords(page_title, meta_description)

        return {
            "domain":                   domain,
            "final_url":                response.url,
            "redirect_count":           redirect_count,
            "redirect_chain":           redirect_chain,
            "status_code":              response.status_code,
            "server_header":            server_header,
            "security_headers_present": present,
            "security_headers_missing": missing,
            "favicon_hash":             favicon_hash,
            "page_title":               page_title,
            "meta_description":         meta_description,
            "brand_keywords_in_title":  brand_keywords,
            "resolved":                 True,
            "error":                    None,
        }

    except requests.exceptions.TooManyRedirects:
        return _failure(domain, "TooManyRedirects")
    except requests.exceptions.SSLError as e:
        return _failure(domain, f"SSLError: {e}")
    except requests.exceptions.Timeout:
        return _failure(domain, "Timeout")
    except requests.exceptions.RequestException as e:
        return _failure(domain, f"RequestError: {e}")
    except Exception as e:
        return _failure(domain, f"UnknownError: {e}")


def _fetch_favicon_hash(page_url: str, html: str) -> str | None:
    """
    Finds the favicon URL referenced in the page HTML (or falls back
    to the standard /favicon.ico path), downloads it, and returns an
    MD5 hash of its raw bytes. MD5 is fine here — we're not doing
    anything security-critical with this hash, just using it as a
    cheap fingerprint to compare "do these two sites use the identical
    icon file," which is exactly what phishing-kit reuse looks like.
    """
    favicon_url = None

    match = re.search(
        r'<link[^>]+rel=["\'](?:shortcut )?icon["\'][^>]+href=["\']([^"\']+)["\']',
        html, re.IGNORECASE,
    )
    if match:
        favicon_url = urljoin(page_url, match.group(1))
    else:
        favicon_url = urljoin(page_url, "/favicon.ico")

    try:
        favicon_response = requests.get(
            favicon_url, timeout=5.0,
            headers={"User-Agent": USER_AGENT},
        )
        if favicon_response.status_code == 200 and len(favicon_response.content) > 0:
            return hashlib.md5(favicon_response.content).hexdigest()
    except requests.exceptions.RequestException:
        pass   # favicon fetch failing is common and not worth a hard error

    return None


def _extract_title(html: str) -> str | None:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()[:200]   # cap length defensively
    return None


def _extract_meta_description(html: str) -> str | None:
    match = re.search(
        r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']*)["\']',
        html, re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()[:300]
    return None


# Brand names most commonly impersonated in phishing campaigns —
# per APWG/PhishTank target-tracking reports. This is a SMALL, cheap,
# illustrative list, not an exhaustive brand database — its purpose
# is to catch the most obvious "title says X but domain has nothing
# to do with X" mismatch cases.
COMMON_PHISHING_TARGETS = [
    "paypal", "amazon", "microsoft", "apple", "netflix", "bank",
    "chase", "wellsfargo", "americanexpress", "visa", "mastercard",
    "facebook", "instagram", "google", "outlook", "office365",
    "docusign", "dhl", "fedex", "ups",
]


def _detect_brand_keywords(title: str | None, description: str | None) -> list:
    """
    Checks page title/description for well-known brand names. On its
    own this is a weak signal (legitimate brand-related pages exist),
    but COMBINED with "this brand's name appears, but the domain name
    doesn't match the brand at all" it becomes a classic phishing tell —
    that cross-check happens in build_graph.py, not here; this function
    just surfaces the raw keyword hits.
    """
    text = f"{title or ''} {description or ''}".lower()
    return [brand for brand in COMMON_PHISHING_TARGETS if brand in text]


def _failure(domain: str, error_msg: str) -> dict:
    return {
        "domain": domain, "final_url": None, "redirect_count": 0,
        "redirect_chain": [], "status_code": None, "server_header": None,
        "security_headers_present": [], "security_headers_missing": [],
        "favicon_hash": None, "page_title": None, "meta_description": None,
        "brand_keywords_in_title": [], "resolved": False, "error": error_msg,
    }


def _print_result(result: dict) -> None:
    print(f"\n  Domain:            {result['domain']}")
    if result["resolved"]:
        print(f"  Status:            ✓ RESOLVED  (HTTP {result['status_code']})")
        print(f"  Final URL:         {result['final_url']}")
        if result["redirect_count"] > 0:
            print(f"  Redirects:         {result['redirect_count']}  →  {result['redirect_chain']}")
        print(f"  Server:            {result['server_header']}")
        print(f"  Security headers present: {result['security_headers_present']}")
        print(f"  Security headers MISSING: {result['security_headers_missing']}")
        print(f"  Favicon hash:      {result['favicon_hash']}")
        print(f"  Page title:        {result['page_title']}")
        if result["brand_keywords_in_title"]:
            print(f"  ⚠ Brand keywords found: {result['brand_keywords_in_title']}")
    else:
        print(f"  Status:            ✗ FAILED  ({result['error']})")
    print(f"  {'─'*52}")


# ── SELF-TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        print("\n" + "=" * 58)
        print("  EVELYN — fetch_page_meta()")
        print("=" * 58)
        _print_result(fetch_page_meta(sys.argv[1]))
    else:
        TEST_DOMAINS = ["github.com", "google.com"]
        print("\n" + "=" * 58)
        print("  EVELYN — fetch_page_meta() test suite")
        print("=" * 58)
        for domain in TEST_DOMAINS:
            _print_result(fetch_page_meta(domain))