# EVELYN
### Email Verification & Exploit Localization Yield Network

> Quantum graph-theoretic phishing infrastructure detection and campaign attribution.  
> **Research project** — not production software.

---

## What this is

EVELYN detects phishing campaigns by analysing the **topology** of attacker
infrastructure — how domains, IPs, TLS certificates, registrars, and ASNs
connect to each other — rather than the surface features of URLs or page
content.

The core insight: attackers change domain names cheaply and constantly, but
reorganising their hosting infrastructure is expensive. The *shape* of how
they connect their assets is a stable fingerprint. EVELYN computes that
fingerprint using a **continuous-time quantum walk** on a hypergraph, producing
a permutation-invariant feature vector φ(G) that is blind to node labels and
sensitive only to topology.

---

## Research goal

Given a previously unseen phishing domain, attribute it to a known attacker
campaign — or flag it as a novel campaign — using only passively-observable
DNS, WHOIS, certificate transparency, and BGP data.

---

## Architecture