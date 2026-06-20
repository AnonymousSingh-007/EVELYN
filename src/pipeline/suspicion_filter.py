# src/pipeline/suspicion_filter.py
#
# PURPOSE: Decide whether a co-hosted or cert-peer domain is worth
# keeping in the graph / expanding recursively, based on INDEPENDENT
# suspicion signals — not just "it happens to share infrastructure."
#
# WHY THIS MATTERS (from your real recursive run):
#   bravenew.agency, qimmah.ai, inmofiscal.academy all got pulled into
#   your phishing campaign graph purely because they share an IP with
#   your seed domain. But shared cheap hosting is used by THOUSANDS of
#   unrelated legitimate small businesses. Without filtering, your
#   "campaign infrastructure map" is mostly innocent bystanders, which
#   would make your eigenvalue spectrum and clustering results MEANINGLESS
#   — you'd be measuring "what does shared hosting look like" not
#   "what does a phishing campaign look like."
#
# THIS IS A CHEAP, ZERO-COST GATE: it reuses signals you ALREADY collect
# (fetch_whois, parse_url) rather than calling any new paid API.

from src.pipeline.fetch_whois import fetch_whois
from src.pipeline.parse_url import parse_url


# Same suspicious-TLD list used in parse_url.py's display logic —
# centralizing it here so both modules can import from one source
# of truth instead of maintaining two separate copies.
SUSPICIOUS_TLDS = {
    "xyz", "cc", "tk", "ml", "ga", "cf", "gq",
    "top", "pw", "work", "click", "link", "online"
}

# A domain registered more recently than this is considered "new
# enough to be suspicious" on its own. Chosen from the phishing
# literature's common threshold (most studies use 30-90 days; we
# use 90 to be slightly more permissive given WHOIS data quality
# issues you've already seen — privacy-protected/redacted WHOIS
# records sometimes report inaccurate creation dates).
SUSPICIOUS_AGE_DAYS = 90


def suspicion_score(domain: str, skip_whois: bool = False) -> dict:
    """
    Computes a suspicion score for a domain based on cheap, free signals.
    Used to decide whether a co_host/cert_peer node is worth keeping
    and recursively expanding, vs. discarding as likely-innocent noise.

    Parameters:
        domain     : bare domain to evaluate
        skip_whois : if True, skips the WHOIS call (faster, but loses
                     the domain-age signal — use this when you're
                     filtering MANY candidates quickly and want to
                     avoid hammering WHOIS servers; do a second pass
                     with skip_whois=False on survivors only)

    Returns:
    {
        "domain":          str,
        "score":           int (0-3, higher = more suspicious),
        "signals":         list of str (which signals fired),
        "is_suspicious":   bool (score >= 1),
        "domain_age_days": int or None,
    }
    """
    signals = []

    # Signal 1: suspicious TLD (free — just string parsing, no network call)
    parsed = parse_url(f"https://{domain}")
    if parsed["suffix"] in SUSPICIOUS_TLDS:
        signals.append(f"suspicious_tld:{parsed['suffix']}")

    # Signal 2: very long domain name with many hyphens (common in
    # phishing domains trying to mimic a brand: "paypal-secure-verify-update.com")
    # Free check, no network call.
    hyphen_count = parsed["domain"].count("-")
    if hyphen_count >= 2:
        signals.append(f"hyphen_heavy:{hyphen_count}_hyphens")

    domain_age_days = None
    if not skip_whois:
        # Signal 3: very new domain registration — this is the strongest
        # signal but costs a WHOIS network call, so it's skippable for
        # fast first-pass filtering of large candidate lists.
        whois_result = fetch_whois(domain)
        if whois_result["resolved"]:
            domain_age_days = whois_result["domain_age_days"]
            if domain_age_days is not None and domain_age_days < SUSPICIOUS_AGE_DAYS:
                signals.append(f"new_domain:{domain_age_days}_days_old")

    score = len(signals)

    return {
        "domain":          domain,
        "score":           score,
        "signals":         signals,
        "is_suspicious":   score >= 1,
        "domain_age_days": domain_age_days,
    }


def filter_suspicious_domains(domains: list, skip_whois: bool = False, verbose: bool = True) -> list:
    """
    Filters a list of candidate domains (e.g. co_host or cert_peer
    discoveries) down to only the ones showing independent suspicion
    signals. Returns the filtered list AND prints a summary of what
    got kept vs discarded — this filtering decision is itself worth
    logging for your methods/limitations section.
    """
    kept = []
    discarded = []

    for domain in domains:
        result = suspicion_score(domain, skip_whois=skip_whois)
        if result["is_suspicious"]:
            kept.append(domain)
            if verbose:
                print(f"    ✓ KEEP  {domain}  (signals: {result['signals']})")
        else:
            discarded.append(domain)
            if verbose:
                print(f"    ✗ DROP  {domain}  (no suspicion signals — likely innocent co-tenant)")

    if verbose:
        print(f"\n    Filtered {len(domains)} candidates → "
              f"{len(kept)} kept, {len(discarded)} discarded as noise")

    return kept


# ── SELF-TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("\n" + "=" * 58)
    print("  EVELYN — suspicion_filter()")
    print("=" * 58)

    if len(sys.argv) > 1:
        domain = sys.argv[1]
        result = suspicion_score(domain, skip_whois=False)
        print(f"\n  Domain:          {result['domain']}")
        print(f"  Score:           {result['score']}")
        print(f"  Signals:         {result['signals']}")
        print(f"  Is suspicious:   {result['is_suspicious']}")
        print(f"  Domain age:      {result['domain_age_days']} days")
    else:
        # Test against the EXACT real domains from your recursive run —
        # this directly validates whether the filter would have correctly
        # discarded the likely-innocent co-tenants you discovered.
        print("\n  Testing against real domains from your recursive expansion run:")
        print("  (using skip_whois=False to get the full real signal)\n")
        test_domains = [
            "bravenew.agency",
            "artificialintellegence.ai",
            "qimmah.ai",
            "inmofiscal.academy",
            "aelva.ai",
            "conceptshq.ai",
        ]
        filter_suspicious_domains(test_domains, skip_whois=False, verbose=True)