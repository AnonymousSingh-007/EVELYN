<div align="center">

```
███████╗██╗   ██╗███████╗██╗  ██╗   ██╗███╗   ██╗
██╔════╝██║   ██║██╔════╝██║  ╚██╗ ██╔╝████╗  ██║
█████╗  ██║   ██║█████╗  ██║   ╚████╔╝ ██╔██╗ ██║
██╔══╝  ╚██╗ ██╔╝██╔══╝  ██║    ╚██╔╝  ██║╚██╗██║
███████╗ ╚████╔╝ ███████╗███████╗██║   ██║ ╚████║
╚══════╝  ╚═══╝  ╚══════╝╚══════╝╚═╝   ╚═╝  ╚═══╝
```

### `Email Verification & Exploit Localization Yield Network`

[![Status](https://img.shields.io/badge/status-active_research-brightgreen?style=flat-square&logo=electron)]()
[![Python](https://img.shields.io/badge/python-3.11+-blue?style=flat-square&logo=python&logoColor=white)]()
[![Stage](https://img.shields.io/badge/stage-1b_resolve__dns-orange?style=flat-square)]()
[![Venue](https://img.shields.io/badge/target-IEEE_S%26P_%2F_USENIX-red?style=flat-square)]()
[![License](https://img.shields.io/badge/license-private_research-lightgrey?style=flat-square)]()

*Quantum graph-theoretic phishing infrastructure detection and campaign attribution.*

---

**"They can change their names. They cannot change their shape."**

</div>

---

## ◈ What is EVELYN

Phishing campaigns are not random. Every attacker leaves behind a structural signature in the internet's fabric — a pattern of how their domains, IPs, certificates, registrars, and hosting providers connect to each other. They change domain names constantly. They cannot afford to rebuild their entire infrastructure.

**EVELYN hunts the shape, not the name.**

It models attacker infrastructure as a hypergraph and computes a **continuous-time quantum walk fingerprint** φ(G) — a permutation-invariant feature vector that encodes topology while remaining completely blind to node labels. Two phishing domains with completely different names, if they share infrastructure shape, will produce identical fingerprints. That is the attribution signal.

```
ATTACKER CHANGES:          ATTACKER CANNOT CHANGE:
─────────────────          ───────────────────────
✗ Domain name              ✓ How many IPs they use
✗ URL path                 ✓ Which registrar they bulk-buy from  
✗ Page content             ✓ Which ASN hosts their servers
✗ TLS certificate          ✓ Whether they share certs across domains
✗ Lure theme               ✓ The topology of their infrastructure
```

---

## ◈ The Core Idea

```
G_i  =  infrastructure subgraph of phishing domain i
         (nodes: domain, IP, registrar, TLS cert, ASN)
         (edges: resolves-to, registered-by, shares-cert, hosted-in)

H    =  −A                    ← Hamiltonian = negative adjacency matrix

U(t) =  e^(−iHt)              ← quantum evolution operator

φ(G) =  { |⟨j|U(t)|k⟩|² }   ← the fingerprint: topology encoded as
                                  probability amplitudes after interference
```

> Same infrastructure shape → same φ(G) → same attacker.  
> Every time. Provably.

---

## ◈ Architecture

```
                        ┌─────────────────────────────────────┐
                        │           EVELYN PIPELINE            │
                        └─────────────────────────────────────┘

  RAW URL
    │
    ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ parse_url() │───▶│resolve_dns()│───▶│fetch_whois()│───▶│fetch_cert() │
│             │    │             │    │             │    │             │
│ domain      │    │ IP node     │    │ registrar   │    │ TLS cert    │
│ TLD         │    │ A record    │    │ node        │    │ fingerprint │
│ subdomain   │    │ MX record   │    │ org name    │    │ shared cert │
│ is_ip flag  │    │ PTR record  │    │ created_dt  │    │ detection   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                │
                        ┌───────────────────────────────────────┘
                        │
                        ▼
               ┌──────────────────┐
               │  build_graph()   │  ←── assemble G_i as NetworkX hypergraph
               └──────────────────┘
                        │
              ┌─────────┴──────────┐
              │                    │
              ▼                    ▼
    ┌──────────────────┐  ┌──────────────────┐
    │  hamiltonian()   │  │   (baselines)    │
    │  H = −A          │  │   GraphSAGE      │
    └──────────────────┘  │   Lexical RF     │
              │            └──────────────────┘
              ▼
    ┌──────────────────┐
    │     walk()       │  U(t) = expm(−iHt)
    │  t ∈ {t₁…tₘ}    │  scipy.linalg.expm
    └──────────────────┘
              │
              ▼
    ┌──────────────────┐
    │  fingerprint()   │  φ(G) = |⟨j|U(t)|k⟩|²  for all j,k,t
    │  feature vector  │  permutation-invariant, fixed-dimension
    └──────────────────┘
              │
              ▼
    ┌──────────────────┐
    │ dbscan_cluster() │  DBSCAN on φ(G) embeddings
    │   + evaluate()   │  ARI / NMI / silhouette
    └──────────────────┘
              │
              ▼
    ┌──────────────────┐
    │    UMAP plot     │  campaign clusters → paper figure
    │  campaign attr.  │  known / novel / benign
    └──────────────────┘
```

---

## ◈ Repository Layout

```
EVELYN/
│
├── src/                         ← all production source code
│   ├── pipeline/                ← data collection (Stage 1)
│   │   ├── parse_url.py         ✅ URL → domain/TLD/IP-flag
│   │   ├── resolve_dns.py       🔄 domain → IP node (in progress)
│   │   ├── fetch_whois.py       ⏳ domain → registrar node
│   │   ├── fetch_cert.py        ⏳ domain → TLS cert node
│   │   └── build_graph.py       ⏳ all above → NetworkX G_i
│   │
│   ├── quantum/                 ← the core method (Stage 2–3)
│   │   ├── hamiltonian.py       ⏳ H = −A
│   │   ├── walk.py              ⏳ U(t) = expm(−iHt)
│   │   └── fingerprint.py       ⏳ φ(G) feature extraction
│   │
│   └── clustering/              ← attribution engine (Stage 4–5)
│       ├── dbscan_cluster.py    ⏳ DBSCAN campaign clustering
│       └── evaluate.py          ⏳ ARI, NMI, silhouette, UMAP
│
├── data/                        ← [gitignored — never commit phishing data]
│   ├── raw/                     PhishTank CSVs, raw DNS outputs
│   ├── processed/               Cleaned labelled URL lists
│   └── graphs/                  Serialised NetworkX graphs per domain
│
├── notebooks/                   ← Jupyter exploration (not production)
├── experiments/                 ← saved run configs and result logs
├── results/
│   ├── figures/                 UMAP plots → paper figures
│   └── metrics/                 ARI / NMI / F1 CSVs per experiment
│
├── docs/                        ← running methods notes → Section 3 of paper
├── tests/                       ← unit tests for every src/ function
├── requirements.txt
└── .gitignore
```

---

## ◈ Setup

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/EVELYN.git
cd EVELYN

# 2. Create isolated environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Install all dependencies
pip install -r requirements.txt
```

---

## ◈ Usage

Every module is self-testing. Run any file directly to verify it works.

```bash
# ── Stage 1a: URL parsing ─────────────────────────────────────
# Analyse a single URL (your input only, no test clutter)
python src/pipeline/parse_url.py https://hdfc-secure-login.xyz/verify

# Run the built-in test suite
python src/pipeline/parse_url.py

# ── Stage 1b: DNS resolution ──────────────────────────────────
python src/pipeline/resolve_dns.py https://suspicious-domain.xyz
python src/pipeline/resolve_dns.py                    # test suite

# ── Future stages (coming soon) ───────────────────────────────
python src/pipeline/fetch_whois.py <domain>
python src/pipeline/fetch_cert.py  <domain>
python src/pipeline/build_graph.py <url>
python src/quantum/fingerprint.py  <url>
```

---

## ◈ Stage Progress

| # | Stage | Description | Status |
|---|-------|-------------|--------|
| 0 | Foundation | Quantum walks, graph theory, GNN comparison | ✅ Complete |
| 1a | `parse_url.py` | URL decomposition — TLD, subdomain, IP detection | ✅ Complete |
| 1b | `resolve_dns.py` | DNS A / MX / PTR resolution → IP node | 🔄 In progress |
| 1c | `fetch_whois.py` | WHOIS → registrar, org, creation date | ⏳ Pending |
| 1c | `fetch_cert.py` | crt.sh → TLS cert fingerprint, shared-cert detection | ⏳ Pending |
| 1d | `build_graph.py` | Assemble full NetworkX hypergraph G_i | ⏳ Pending |
| 2 | `hamiltonian.py` | H = −A on any NetworkX graph | ⏳ Pending |
| 2 | `walk.py` | U(t) = expm(−iHt) with multiple t values | ⏳ Pending |
| 3 | `fingerprint.py` | φ(G) feature vector extraction + verification | ⏳ Pending |
| 4 | `dbscan_cluster.py` | DBSCAN campaign clustering on φ(G) | ⏳ Pending |
| 5 | `evaluate.py` | ARI, NMI, silhouette, UMAP paper figure | ⏳ Pending |

---

## ◈ Why Quantum Walk, Not GraphSAGE

| Property | GraphSAGE / GAT | EVELYN φ(G) |
|----------|----------------|-------------|
| Captures local structure (1–2 hop) | ✅ Yes | ✅ Yes |
| Captures global topology | ⚠️ Requires deep networks | ✅ By construction |
| Permutation-invariant | ⚠️ Not guaranteed | ✅ Provably |
| Works on unseen topology patterns | ❌ Needs labelled examples | ✅ Zero-shot |
| Fixed-dim output for variable-size graphs | ❌ Requires pooling heuristics | ✅ Native |
| Label-blind (domain name irrelevant) | ❌ Node features matter | ✅ Topology only |

The honest tradeoff: φ(G) costs O(n³) vs O(n·d) for GNNs. For campaign subgraphs where n < 100, this is milliseconds. The structural invariance guarantee is worth it.

---

## ◈ Academic Context

```
Target venues  :  IEEE S&P  |  USENIX Security  |  CCS  |  NDSS
Method         :  Continuous-time quantum walk on infrastructure hypergraphs
Baseline comps :  GraphSAGE, GAT, lexical Random Forest, classical random walk
Dataset        :  PhishTank (phishing) + Tranco top-1M (benign) — 1,000 URLs min
Evaluation     :  ARI, NMI, silhouette score, zero-day attribution F1
```

---

## ◈ The Rule

> Every function in `src/` — you write the first draft.  
> It will be wrong. We debug it together. You understand the fix.  
> No black boxes. No copy-paste without comprehension.  
> A reviewer will ask why. You will have the answer.

---

<div align="center">

*This repository is private during active research. Do not distribute.*

`EVELYN` · built line by line · understood end to end

</div>