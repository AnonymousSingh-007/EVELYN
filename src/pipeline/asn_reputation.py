# src/pipeline/asn_reputation.py
#
# PURPOSE: Score an ASN (hosting provider) by its known association
# with bulletproof hosting, VPN/proxy services, or documented abuse
# patterns. This turns the ASN node from a NEUTRAL LABEL into a real
# weighted signal — "AS44477" by itself means nothing to a human
# reader, but "AS44477 (Stark Industries — sanctioned bulletproof
# hosting, rebranded as WorkTitans/THE Hosting in 2025)" is immediately
# actionable threat intelligence.
#
# DATA SOURCE: a curated, explicitly-tagged subset of the bountyyfi/
# bad-asn-list community blocklist (github.com/bountyyfi/bad-asn-list),
# itself informed by CISA/NSA/FBI's 2025 joint "Bulletproof Defense"
# guidance and Spamhaus DROP lists. We use ONLY entries that are
# EXPLICITLY labeled bulletproof/abuse/sanctioned in that source —
# we deliberately do NOT flag generic cloud/VPS providers (Hetzner,
# DigitalOcean, AWS, OVH, etc) as suspicious on ASN alone, because
# those host millions of completely legitimate sites and doing so
# would make this signal worthless through false positives.
#
# HONEST LIMITATION (state this in your methods/limitations section):
# Censys's own 2026 research on this exact topic notes there is "no
# single ground-truth dataset" for bulletproof hosting, and that
# operators increasingly rotate ASNs and use reseller ecosystems to
# blend into mainstream providers specifically to evade lists like
# this one. This signal has real value but WILL miss freshly-rotated
# or well-laundered bulletproof infrastructure — it should never be
# the sole basis for a phishing/benign decision, only one weighted
# input among many (exactly the same principle as the suspicion filter).

# Tier 1: EXPLICITLY labeled "bulletproof hosting" in the source list,
# or directly named in CISA/Treasury sanctions actions. Highest confidence.
BULLETPROOF_ASNS = {
    "AS213702": "Qwins Ltd",
    "AS198953": "Proton66 OOO",
    "AS401116": "NYBULA",
    "AS401120": "CHEAPY-HOST",
    "AS200593": "PROSPERO OOO",
    "AS214943": "Railnet LLC",
    "AS61432":  "VAIZ-AS (Ukraine)",
    "AS210950": "E-RISHENNYA-ASN (Ukraine)",
    "AS211736": "FDN3 (Ukraine)",
    "AS210848": "TK-NET (Seychelles)",
    "AS212283": "ROZA-AS (Bulgaria)",
    "AS204428": "SS-Net (Bulgaria)",
    "AS44559":  "MIRhosting (Stark Industries)",
    "AS216246": "Aeza Group Ltd",
    "AS57523":  "Chang Way Technologies Co. Limited",
    "AS207566": "Chang Way Technologies Co. Limited",
    "AS44477":  "Stark Industries Solutions Ltd (PQ Hosting) — EU-sanctioned May 2025",
    "AS209847": "WorkTitans B.V. — Stark Industries rebrand post-sanctions",
    "AS211252": "Delis LLC (CrazyRDP)",
    "AS401115": "EKABI (CrazyRDP downstream)",
}

# Tier 2: associated with specific named threat actor groups or
# documented ransomware/cybercrime infrastructure in threat-intel
# reporting (Censys, Intel471, etc), even if not formally labeled
# "bulletproof" by name.
THREAT_ACTOR_LINKED_ASNS = {
    "AS62212":  "SmartApe OU — linked to FIN7",
    "AS42474":  "SmartApe OU (Estonia)",
    "AS56694":  "LLC Smart Ape (Russia)",
    "AS209588": "Flyservers S.A. — linked to ShadowSyndicate",
    "AS209132": "Alviva Holding Limited — ransomware infrastructure",
    "AS51381":  "1337TEAM LIMITED (ELITETEAM, Seychelles)",
    "AS60424":  "1337TEAM LIMITED (ELITETEAM)",
    "AS56873":  "1337TEAM LIMITED (ELITETEAM)",
    "AS39770":  "1337TEAM LIMITED (ELITETEAM)",
    "AS200373": "Drei-K-Tech-GmbH — DDoS proxy source",
}

# Tier 3: VPN, residential-proxy, or Tor-exit infrastructure. Not
# inherently malicious — but a phishing page's ADMIN PANEL or C2
# infrastructure connecting FROM one of these is a meaningful signal
# in a different way than the public-facing phishing domain itself
# being hosted here (which is common and less informative).
ANONYMIZATION_ASNS = {
    "AS212238": "Datacamp Limited", "AS60068": "Datacamp Limited",
    "AS211612": "Datacamp Limited", "AS206092": "Datacamp Limited",
    "AS9009":   "M247 Europe SRL", "AS16247": "M247 Ltd", "AS42973": "M247 Ltd",
    "AS33837":  "Fredrik Holmqvist (Tor exit node)",
    "AS201814": "MEVSPACE sp. z o.o. (CroxyProxy)",
    "AS202425": "IP Volume inc (Speedify VPN)",
    "AS17561":  "Plainproxies (3xK Tech)",
    "AS132817": "Webshare Proxy",
    "AS64267":  "Rayobyte Proxy",
    "AS131642": "Rapidseedbox Proxy",
}


def score_asn(asn: str) -> dict:
    """
    Scores an ASN by reputation tier.

    Example output:
    {
        "asn":         "AS44477",
        "tier":        "bulletproof",
        "label":       "Stark Industries Solutions Ltd (PQ Hosting) — EU-sanctioned May 2025",
        "risk_score":  3,
        "is_flagged":  True
    }

    risk_score: 0 = no known association, 1 = anonymization/proxy infra,
    2 = threat-actor-linked, 3 = explicitly bulletproof/sanctioned.
    """
    # Normalize: ASN strings can arrive as "AS4766" or "4766" depending
    # on the source (your fetch_asn.py sometimes returns RDAP handle
    # strings rather than clean AS numbers — we handle both forms here)
    asn_clean = asn.strip().upper()
    if not asn_clean.startswith("AS") and asn_clean.replace("AS", "").isdigit():
        asn_clean = f"AS{asn_clean.lstrip('AS')}"

    if asn_clean in BULLETPROOF_ASNS:
        return {"asn": asn, "tier": "bulletproof", "label": BULLETPROOF_ASNS[asn_clean],
                "risk_score": 3, "is_flagged": True}
    if asn_clean in THREAT_ACTOR_LINKED_ASNS:
        return {"asn": asn, "tier": "threat_actor_linked", "label": THREAT_ACTOR_LINKED_ASNS[asn_clean],
                "risk_score": 2, "is_flagged": True}
    if asn_clean in ANONYMIZATION_ASNS:
        return {"asn": asn, "tier": "anonymization", "label": ANONYMIZATION_ASNS[asn_clean],
                "risk_score": 1, "is_flagged": True}

    return {"asn": asn, "tier": "unknown", "label": None, "risk_score": 0, "is_flagged": False}


def _print_result(result: dict) -> None:
    print(f"\n  ASN:           {result['asn']}")
    print(f"  Tier:          {result['tier']}")
    print(f"  Risk score:    {result['risk_score']}/3")
    if result["label"]:
        print(f"  ⚠ {result['label']}")
    print(f"  {'─'*52}")


# ── SELF-TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        print("\n" + "=" * 58)
        print("  EVELYN — score_asn()")
        print("=" * 58)
        _print_result(score_asn(sys.argv[1]))
    else:
        TEST_ASNS = [
            "AS44477",   # Stark Industries — should hit Tier 1 (bulletproof)
            "AS62212",   # SmartApe — should hit Tier 2 (threat-actor-linked)
            "AS9009",    # M247 — should hit Tier 3 (anonymization)
            "AS15169",   # Google — should hit Tier 0 (unknown / clean)
        ]
        print("\n" + "=" * 58)
        print("  EVELYN — score_asn() test suite")
        print("=" * 58)
        for asn in TEST_ASNS:
            _print_result(score_asn(asn))