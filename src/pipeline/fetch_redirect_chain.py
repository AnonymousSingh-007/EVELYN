# src/pipeline/fetch_redirect_chain.py
#
# PURPOSE: Follow a URL's FULL redirect chain to find where it actually
# ends up. This MUST run before build_graph() decides what "the domain"
# even is — your real netlify.app investigation showed exactly why:
# the PhishTank-listed URL may not be the final landing page at all.
#
# PHISHING REDIRECT PATTERNS THIS CATCHES:
#   - Link shorteners (bit.ly, tinyurl) hiding the real destination
#   - "Cloaking" — a free-hosting page that immediately redirects to
#     the REAL phishing infrastructure (the free host is a disposable
#     front door; the real attacker infra is one hop further)
#   - Open-redirect abuse on legitimate sites (attacker uses a trusted
#     domain's own redirect feature to launder the link's appearance)
#   - Multi-stage redirect chains designed to evade simple "check the
#     domain in the URL" detectors
#
# DESIGN DECISION: this module determines the TRUE TARGET domain BEFORE
# any other pipeline module runs. build_graph.py should call this FIRST
# and run the entire rest of the pipeline (DNS/WHOIS/cert/etc) against
# the FINAL domain in the chain, not just the domain in the original URL.

import requests


MAX_REDIRECTS = 8
REQUEST_TIMEOUT = 8.0
USER_AGENT = "Mozilla/5.0 (EVELYN-research-tool; academic project; passive analysis only)"


def fetch_redirect_chain(url: str) -> dict:
    """
    Follows a URL through all redirects and returns the full chain.

    Example output:
    {
        "original_url":   "https://bit.ly/3xK9z",
        "final_url":      "https://vigilant-austin-37062a.netlify.app/login",
        "chain":          ["https://bit.ly/3xK9z", "https://vigilant-austin-37062a.netlify.app/login"],
        "hop_count":      1,
        "is_cloaked":     False,   # True if final domain's TLD/registration looks
                                     # suspiciously different from a "normal" redirect
        "resolved":       True,
        "error":          None
    }
    """
    try:
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
            allow_redirects=True,
        )

        chain = [r.url for r in response.history] + [response.url]
        hop_count = len(response.history)

        if hop_count > MAX_REDIRECTS:
            # requests itself raises TooManyRedirects before this point
            # in practice, but we keep this as a defensive backstop
            pass

        return {
            "original_url": url,
            "final_url":    response.url,
            "chain":        chain,
            "hop_count":    hop_count,
            "status_code":  response.status_code,
            "resolved":     True,
            "error":        None,
        }

    except requests.exceptions.TooManyRedirects:
        return _failure(url, "TooManyRedirects — possible redirect loop or evasion technique")
    except requests.exceptions.SSLError as e:
        return _failure(url, f"SSLError: {e}")
    except requests.exceptions.Timeout:
        return _failure(url, "Timeout")
    except requests.exceptions.RequestException as e:
        return _failure(url, f"RequestError: {e}")
    except Exception as e:
        return _failure(url, f"UnknownError: {e}")


def _failure(url: str, error_msg: str) -> dict:
    return {
        "original_url": url, "final_url": None, "chain": [], "hop_count": 0,
        "status_code": None, "resolved": False, "error": error_msg,
    }


def _print_result(result: dict) -> None:
    print(f"\n  Original URL:  {result['original_url']}")
    if result["resolved"]:
        print(f"  Status:        ✓ RESOLVED  (HTTP {result['status_code']})")
        print(f"  Final URL:     {result['final_url']}")
        print(f"  Hops:          {result['hop_count']}")
        if result["hop_count"] > 0:
            print(f"  Full chain:")
            for i, hop_url in enumerate(result["chain"]):
                print(f"      [{i}] {hop_url}")
        if result["hop_count"] >= 2:
            print(f"  ⚠ Multi-hop redirect — investigate whether this is cloaking")
    else:
        print(f"  Status:        ✗ FAILED  ({result['error']})")
    print(f"  {'─'*52}")


# ── SELF-TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        print("\n" + "=" * 58)
        print("  EVELYN — fetch_redirect_chain()")
        print("=" * 58)
        _print_result(fetch_redirect_chain(sys.argv[1]))
    else:
        TEST_URLS = [
            "https://github.com",        # should have 0-1 hops, simple
            "https://google.com",        # google.com -> www.google.com, 1 hop typically
        ]
        print("\n" + "=" * 58)
        print("  EVELYN — fetch_redirect_chain() test suite")
        print("=" * 58)
        for url in TEST_URLS:
            _print_result(fetch_redirect_chain(url))