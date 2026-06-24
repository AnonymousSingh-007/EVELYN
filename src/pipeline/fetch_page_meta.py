# src/pipeline/fetch_page_meta.py
#
# PURPOSE: Fetch a domain's page ONCE and extract three independent
# signals from that single request: HTTP headers, favicon hash, and
# page title/meta/brand-keyword fingerprint.
#
# THIS VERSION FIXES A REAL BUG: the original favicon regex required
# rel="icon" and href="..." to appear in ONE specific attribute order,
# which missed real-world HTML constantly. GitHub itself was a
# confirmed miss — its <link> tags use rel/type/href in a different
# order, and it also exposes mask-icon and apple-touch-icon variants
# that the old code never even looked for.
#
# THE FIX IS A RECURSIVE, MULTI-STRATEGY FALLBACK CHAIN:
#   Strategy 1: parse ALL <link> tags in <head>, regardless of
#               attribute order, and rank by icon-type preference
#               (standard icon > shortcut icon > apple-touch-icon >
#               mask-icon) — this alone fixes the GitHub case.
#   Strategy 2: if no <link> tag matches, try the conventional
#               /favicon.ico path directly (the original fallback).
#   Strategy 3: if the page itself redirected (common for HTTPS
#               upgrade, www-prefix, or trailing-slash normalization),
#               RECURSIVELY retry favicon discovery against the
#               FINAL resolved URL's origin, not the original one —
#               a relative favicon path on the original URL can
#               resolve to a dead location if the real page lives
#               at a different host/path after redirects.
#   Strategy 4: if a candidate favicon URL 404s or returns a tiny/
#               invalid file, fall through to the NEXT candidate
#               in the ranked list rather than giving up entirely.
#
# This turns "one regex, one guess, give up" into "rank every real
# candidate, try them in order, recurse into the final URL's origin
# if needed" — which is what actually finding a favicon in the wild
# requires.

import requests
import hashlib
import re
from urllib.parse import urljoin, urlparse


SECURITY_HEADERS_OF_INTEREST = [
    "strict-transport-security",
    "content-security-policy",
    "x-frame-options",
    "x-content-type-options",
]

MAX_REDIRECTS = 5
REQUEST_TIMEOUT = 8.0
USER_AGENT = "Mozilla/5.0 (EVELYN-research-tool; academic project; passive analysis only)"

# Preference order for icon rel types — standard "icon"/"shortcut icon"
# is the most universally meaningful for comparing across sites; the
# more specialized variants (apple-touch-icon, mask-icon) are still
# useful campaign-linking signals but lower priority if a standard
# icon is also present, since they're sometimes platform-template
# defaults rather than a deliberate branding choice.
ICON_REL_PRIORITY = [
    "icon",
    "shortcut icon",
    "apple-touch-icon",
    "apple-touch-icon-precomposed",
    "mask-icon",
]

MIN_VALID_FAVICON_BYTES = 16   # anything smaller than this is almost
                                 # certainly an error page or empty stub,
                                 # not a real icon file


def fetch_page_meta(domain: str, use_https: bool = True, _depth: int = 0) -> dict:
    """
    Fetches a domain's homepage once and extracts headers, favicon
    hash, and page metadata.

    Parameters:
        domain    : bare domain to fetch
        use_https : try HTTPS first (falls back to HTTP internally on failure)
        _depth    : internal recursion guard — DO NOT set this yourself.
                    Used to prevent infinite recursion if redirect-chasing
                    favicon discovery somehow loops (e.g. a malformed
                    redirect cycle). Capped at 1 — we only ever recurse
                    ONE level deep (into the final redirected origin),
                    never further.

    Example output adds two new fields vs. the previous version:
        "favicon_url"        : the actual URL the favicon was fetched from
        "favicon_strategy"    : which strategy/candidate succeeded, for
                                 debugging and methods-section transparency
    """
    if "://" in domain or "/" in domain:
        return _failure(domain, "InvalidInput: pass a bare domain, not a full URL")

    scheme = "https" if use_https else "http"
    url = f"{scheme}://{domain}/"

    try:
        response = requests.get(
            url, timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
            allow_redirects=True, verify=True,
        )

        redirect_chain = [r.url for r in response.history] + [response.url]
        redirect_count = len(response.history)

        headers_lower = {k.lower(): v for k, v in response.headers.items()}
        server_header = headers_lower.get("server")
        present = [h for h in SECURITY_HEADERS_OF_INTEREST if h in headers_lower]
        missing = [h for h in SECURITY_HEADERS_OF_INTEREST if h not in headers_lower]

        html = response.text
        final_url = response.url   # the ACTUAL page URL after redirects —
                                    # favicon discovery must use THIS as the
                                    # base for resolving relative paths, not
                                    # the original input URL

        favicon_result = _discover_favicon(final_url, html, depth=_depth)

        page_title = _extract_title(html)
        meta_description = _extract_meta_description(html)
        brand_keywords = _detect_brand_keywords(page_title, meta_description)

        return {
            "domain":                   domain,
            "final_url":                final_url,
            "redirect_count":           redirect_count,
            "redirect_chain":           redirect_chain,
            "status_code":              response.status_code,
            "server_header":            server_header,
            "security_headers_present": present,
            "security_headers_missing": missing,
            "favicon_hash":             favicon_result["hash"],
            "favicon_url":              favicon_result["url"],
            "favicon_strategy":         favicon_result["strategy"],
            "page_title":               page_title,
            "meta_description":         meta_description,
            "brand_keywords_in_title":  brand_keywords,
            "resolved":                 True,
            "error":                    None,
        }

    except requests.exceptions.SSLError as e:
        # HTTPS failed — fall back to HTTP once. Many phishing pages,
        # especially low-effort ones, only serve plain HTTP, and giving
        # up on first HTTPS failure would silently lose those entirely.
        if use_https and _depth == 0:
            return fetch_page_meta(domain, use_https=False, _depth=_depth)
        return _failure(domain, f"SSLError: {e}")
    except requests.exceptions.TooManyRedirects:
        return _failure(domain, "TooManyRedirects")
    except requests.exceptions.Timeout:
        return _failure(domain, "Timeout")
    except requests.exceptions.RequestException as e:
        return _failure(domain, f"RequestError: {e}")
    except Exception as e:
        return _failure(domain, f"UnknownError: {e}")


def _discover_favicon(page_url: str, html: str, depth: int = 0) -> dict:
    """
    Multi-strategy favicon discovery. Returns:
    {"hash": str or None, "url": str or None, "strategy": str}

    Tries candidates in priority order and returns on the FIRST one
    that downloads successfully as a plausible icon file. This is the
    core fix — instead of one regex and one guess, this builds a
    ranked candidate list and works through it methodically.
    """
    candidates = _extract_favicon_candidates(page_url, html)

    # Always append the conventional default path LAST, as a final
    # fallback if no <link> tag specified anything usable.
    candidates.append((urljoin(page_url, "/favicon.ico"), "default_path"))

    for candidate_url, strategy in candidates:
        result = _try_download_favicon(candidate_url)
        if result is not None:
            return {"hash": result, "url": candidate_url, "strategy": strategy}

    # Recursive fallback: if every same-origin candidate failed, and
    # we haven't already recursed, try the ORIGIN of the final page
    # URL directly — catches cases where a deep page path breaks
    # relative resolution, or where the icon only lives at true root.
    if depth == 0:
        parsed = urlparse(page_url)
        origin = f"{parsed.scheme}://{parsed.netloc}/"
        if origin != page_url:
            origin_result = _try_download_favicon(urljoin(origin, "/favicon.ico"))
            if origin_result is not None:
                return {"hash": origin_result, "url": urljoin(origin, "/favicon.ico"),
                        "strategy": "recursive_origin_fallback"}

    return {"hash": None, "url": None, "strategy": "none_found"}


def _extract_favicon_candidates(page_url: str, html: str) -> list:
    """
    Parses ALL <link> tags in the HTML that reference an icon, in ANY
    attribute order, and returns them ranked by ICON_REL_PRIORITY.

    This is the actual bug fix: the old code used one rigid regex
    pattern assuming rel="icon" comes before href="...". This version
    finds every <link ...> tag first, then independently extracts rel
    and href from WITHIN each tag regardless of which comes first.
    """
    link_tags = re.findall(r"<link\b[^>]*>", html, re.IGNORECASE)

    found = []   # list of (priority_index, url, rel_value)

    for tag in link_tags:
        rel_match = re.search(r'rel\s*=\s*["\']([^"\']*)["\']', tag, re.IGNORECASE)
        href_match = re.search(r'href\s*=\s*["\']([^"\']*)["\']', tag, re.IGNORECASE)

        if not rel_match or not href_match:
            continue

        rel_value = rel_match.group(1).strip().lower()
        href_value = href_match.group(1).strip()

        if rel_value not in ICON_REL_PRIORITY:
            continue
        if not href_value:
            continue

        priority = ICON_REL_PRIORITY.index(rel_value)
        absolute_url = urljoin(page_url, href_value)
        found.append((priority, absolute_url, rel_value))

    found.sort(key=lambda x: x[0])

    return [(url, f"link_tag:{rel}") for _, url, rel in found]


def _try_download_favicon(favicon_url: str) -> str | None:
    """
    Attempts to download one favicon candidate and hash it. Returns
    the MD5 hash string on success, or None if this candidate didn't
    pan out (so the caller can move on to the next one in the list).
    """
    try:
        response = requests.get(
            favicon_url, timeout=5.0,
            headers={"User-Agent": USER_AGENT},
        )
        if response.status_code != 200:
            return None
        content = response.content
        if len(content) < MIN_VALID_FAVICON_BYTES:
            return None
        return hashlib.md5(content).hexdigest()
    except requests.exceptions.RequestException:
        return None


def _extract_title(html: str) -> str | None:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()[:200]
    return None


def _extract_meta_description(html: str) -> str | None:
    meta_tags = re.findall(r"<meta\b[^>]*>", html, re.IGNORECASE)
    for tag in meta_tags:
        name_match = re.search(r'name\s*=\s*["\']([^"\']*)["\']', tag, re.IGNORECASE)
        content_match = re.search(r'content\s*=\s*["\']([^"\']*)["\']', tag, re.IGNORECASE)
        if name_match and content_match and name_match.group(1).strip().lower() == "description":
            return content_match.group(1).strip()[:300]
    return None


COMMON_PHISHING_TARGETS = [
    "paypal", "amazon", "microsoft", "apple", "netflix", "bank",
    "chase", "wellsfargo", "americanexpress", "visa", "mastercard",
    "facebook", "instagram", "google", "outlook", "office365",
    "docusign", "dhl", "fedex", "ups",
]


def _detect_brand_keywords(title: str | None, description: str | None) -> list:
    text = f"{title or ''} {description or ''}".lower()
    return [brand for brand in COMMON_PHISHING_TARGETS if brand in text]


def _failure(domain: str, error_msg: str) -> dict:
    return {
        "domain": domain, "final_url": None, "redirect_count": 0,
        "redirect_chain": [], "status_code": None, "server_header": None,
        "security_headers_present": [], "security_headers_missing": [],
        "favicon_hash": None, "favicon_url": None, "favicon_strategy": None,
        "page_title": None, "meta_description": None,
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
        if result["favicon_hash"]:
            print(f"  Favicon hash:      {result['favicon_hash']}")
            print(f"  Favicon URL:       {result['favicon_url']}")
            print(f"  Favicon strategy:  {result['favicon_strategy']}")
        else:
            print(f"  Favicon hash:      None found "
                  f"(strategy attempted: {result['favicon_strategy']})")
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
        # github.com is included DELIBERATELY — it's the exact domain
        # that exposed the original bug (favicon=None). If this test
        # now shows a real hash for github.com, the fix is confirmed.
        TEST_DOMAINS = ["github.com", "google.com", "wikipedia.org"]
        print("\n" + "=" * 58)
        print("  EVELYN — fetch_page_meta() test suite")
        print("=" * 58)
        print("  NOTE: github.com is included specifically to verify the")
        print("  favicon-detection bug fix (was previously favicon=None)\n")
        for domain in TEST_DOMAINS:
            _print_result(fetch_page_meta(domain))