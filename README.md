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
Raw URL

└─► parse_url()          extract domain / TLD / subdomain
└─► resolve_dns()  DNS A record → IP node
└─► fetch_whois()   WHOIS → registrar node
└─► fetch_cert()    crt.sh → TLS cert node
└─► build_graph()   assemble G_i (NetworkX)
└─► hamiltonian()   H = −A
└─► walk()    U(t) = expm(−iHt)
└─► fingerprint()   φ(G)
└─► dbscan_cluster()
└─► Campaign attribution

---

## Repository layout

EVELYN/

├── src/

│   ├── pipeline/       URL parsing → DNS → WHOIS → cert → graph construction

│   ├── quantum/        Hamiltonian, quantum walk, fingerprint extraction

│   └── clustering/     DBSCAN campaign clustering and evaluation

├── data/

│   ├── raw/            PhishTank CSVs and raw DNS dumps  [gitignored]

│   ├── processed/      Cleaned labelled URL lists        [gitignored]

│   └── graphs/         Serialised NetworkX graphs        [gitignored]

├── notebooks/          Exploratory Jupyter notebooks

├── experiments/        Saved run configs and result logs

├── results/

│   ├── figures/        UMAP plots and paper figures

│   └── metrics/        ARI / NMI / F1 score CSVs

├── docs/               Running methods notes → Section 3 of paper

└── tests/              Unit tests for every src/ function

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/EVELYN.git
cd EVELYN
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

---

## Usage

```bash
# Parse and analyse a single URL
python src/pipeline/parse_url.py https://suspicious-domain.xyz/login

# Run the built-in test suite for any module
python src/pipeline/parse_url.py
python src/pipeline/resolve_dns.py
```

---

## Stage progress

| Stage | Description | Status |
|-------|-------------|--------|
| 0 | Conceptual foundation — quantum walks, graph theory, GNN comparison | ✅ Done |
| 1a | `parse_url.py` — URL decomposition with TLD awareness | ✅ Done |
| 1b | `resolve_dns.py` — DNS A record resolution | 🔄 In progress |
| 1c | `fetch_whois.py` + `fetch_cert.py` | ⏳ Pending |
| 1d | `build_graph.py` — NetworkX graph assembly | ⏳ Pending |
| 2 | `hamiltonian.py` + `walk.py` — quantum walk implementation | ⏳ Pending |
| 3 | `fingerprint.py` — φ(G) feature vector extraction | ⏳ Pending |
| 4 | `dbscan_cluster.py` — campaign clustering | ⏳ Pending |
| 5 | `evaluate.py` + UMAP visualisation | ⏳ Pending |

---

## Academic context

Target venues: IEEE S&P, USENIX Security, CCS, NDSS.  
Method: Continuous-time quantum walk fingerprinting on infrastructure hypergraphs.  
Baseline comparisons: GraphSAGE, GAT, lexical Random Forest.

---

*This repository is private during active research. Do not distribute.*