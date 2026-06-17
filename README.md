<div align="center">

```
███████╗██╗   ██╗███████╗██╗  ██╗   ██╗███╗   ██╗
██╔════╝██║   ██║██╔════╝██║  ╚██╗ ██╔╝████╗  ██║
█████╗  ██║   ██║█████╗  ██║   ╚████╔╝ ██╔██╗ ██║
██╔══╝  ╚██╗ ██╔╝██╔══╝  ██║    ╚██╔╝  ██║╚██╗██║
███████╗ ╚████╔╝ ███████╗███████╗██║   ██║ ╚████║
╚══════╝  ╚═══╝  ╚══════╝╚══════╝╚═╝   ╚═╝  ╚═══╝
```

### **E**mail **V**erification & **E**xploit **L**ocalization **Y**ield **N**etwork

*Quantum graph-theoretic phishing infrastructure detection.*  
*They change their names. They can't change their shape.*

---

![Status](https://img.shields.io/badge/STATUS-ACTIVE_RESEARCH-brightgreen?style=for-the-badge&logo=github)
![Stage](https://img.shields.io/badge/STAGE-1b_DNS_RESOLUTION-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/PYTHON-3.14-yellow?style=for-the-badge&logo=python)
![Venue Target](https://img.shields.io/badge/TARGET-IEEE_S%26P_%7C_USENIX-red?style=for-the-badge)
![License](https://img.shields.io/badge/ACCESS-PRIVATE_RESEARCH-black?style=for-the-badge)

</div>

---

## ◈ THE PROBLEM

Attackers spin up hundreds of phishing domains a day.  
Classical detectors look at the **name** — the string, the words, the TLD.  
So attackers change the name. Constantly. Cheaply. $0.88 per domain.

**EVELYN doesn't look at the name.**

It looks at the **shape** — how 50 domains share 3 IPs, 1 registrar, and a single TLS certificate fingerprint. That shape is the attacker's true signature. Rebuilding it costs time, money, and exposure. They can't change it fast enough.

---

## ◈ THE WEAPON

At the core of EVELYN is a **continuous-time quantum walk** on a phishing infrastructure hypergraph.

```
φ(G)  =  |⟨j| e^{−iHt} |k⟩|²   for all j,k ∈ V,  t ∈ {t₁,...,tₘ}
```

Where:
- `H = −A` — the Hamiltonian (negative adjacency matrix of the infrastructure graph)
- `e^{−iHt}` — wave evolution through the graph over time `t` (6 lines of Python)
- `|⟨j|...|k⟩|²` — interference amplitude: the probability the wave travels from node `k` to node `j`
- `φ(G)` — the topology fingerprint, **invariant to node labels, invariant to domain names**

Two campaigns with identical infrastructure patterns produce **identical** `φ(G)`.  
A zero-day domain from a known attacker **clusters with its campaign** before anyone has seen it.

---

## ◈ ARCHITECTURE

```
                    ┌──────────────────────────────────────────┐
                    │             EVELYN PIPELINE               │
                    └──────────────────────────────────────────┘

  RAW URL
    │
    ▼
┌────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ parse_url()│──►│resolve_dns()│──►│fetch_whois()│──►│ fetch_cert()│
│            │   │             │   │             │   │             │
│ domain     │   │ IP node     │   │ registrar   │   │ TLS cert    │
│ TLD        │   │ A record    │   │ node        │   │ fingerprint │
│ subdomain  │   │ TTL         │   │ WHOIS data  │   │ crt.sh      │
└────────────┘   └─────────────┘   └─────────────┘   └──────┬──────┘
                                                              │
                                                              ▼
                                                     ┌─────────────┐
                                                     │build_graph()│
                                                     │  NetworkX   │
                                                     │  G_i ego-   │
                                                     │  graph      │
                                                     └──────┬──────┘
                                                            │
                      ┌─────────────────────────────────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │   hamiltonian()     │
           │   H = −A            │
           └──────────┬──────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │      walk()         │
           │  U(t) = expm(−iHt) │
           └──────────┬──────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │   fingerprint()     │  ◄── permutation-invariant
           │   φ(G) vector       │      topology fingerprint
           └──────────┬──────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │  dbscan_cluster()   │
           │  campaign clusters  │
           └──────────┬──────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │    ATTRIBUTION      │
           │  ✓ Known campaign   │
           │  ⚠ Novel campaign   │
           │  ○ Benign           │
           └─────────────────────┘
```

---

## ◈ WHY QUANTUM WALKS BEAT GNNs

| Property | GraphSAGE / GAT | EVELYN φ(G) |
|---|---|---|
| Captures global structure | ✗ 1–2 hop only | ✓ Full spectral encoding |
| Permutation invariant | ✗ Training artifact | ✓ Provably, by construction |
| Zero-shot on new topologies | ✗ Needs labeled examples | ✓ Unsupervised fingerprint |
| Fixed-dim output for variable graphs | ✗ | ✓ |
| Detects isomorphic campaigns | ✗ | ✓ |
| Cost | O(n·d) per layer | O(n³) bounded by ego-graph |

**One-sentence reviewer answer:**
> *"For campaign attribution where node labels change but topology is stable, quantum walk fingerprints are provably superior because they are topology-invariant by construction, not by training."*

---

## ◈ REPOSITORY

```
EVELYN/
│
├── src/
│   ├── pipeline/
│   │   ├── parse_url.py        ✅  URL → domain / TLD / subdomain / IP flag
│   │   ├── resolve_dns.py      🔄  domain → IP node (A record + TTL)
│   │   ├── fetch_whois.py      ⏳  domain → registrar node
│   │   ├── fetch_cert.py       ⏳  domain → TLS cert fingerprint node
│   │   └── build_graph.py      ⏳  all nodes → NetworkX graph G_i
│   │
│   ├── quantum/
│   │   ├── hamiltonian.py      ⏳  G_i → H = −A
│   │   ├── walk.py             ⏳  H → U(t) = expm(−iHt)
│   │   └── fingerprint.py      ⏳  U(t) → φ(G) feature vector
│   │
│   └── clustering/
│       ├── dbscan_cluster.py   ⏳  φ(G) vectors → campaign clusters
│       └── evaluate.py         ⏳  ARI / NMI / F1 + UMAP visualisation
│
├── data/
│   ├── raw/            ← PhishTank CSVs, raw DNS output      [gitignored]
│   ├── processed/      ← Cleaned labelled URL dataset        [gitignored]
│   └── graphs/         ← Serialised NetworkX graph objects   [gitignored]
│
├── notebooks/          ← Jupyter exploration (not production)
├── experiments/        ← Saved run configs, hyperparameter logs
├── results/
│   ├── figures/        ← UMAP plots, paper figures
│   └── metrics/        ← ARI / NMI / F1 CSVs per experiment
├── docs/               ← Running methods notes → Section 3 of paper
├── tests/              ← Unit tests for every src/ function
├── requirements.txt
└── README.md
```

---

## ◈ SETUP

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/EVELYN.git
cd EVELYN

# Environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# Dependencies
pip install -r requirements.txt
```

---

## ◈ USAGE

```bash
# Analyse a single URL
python src/pipeline/parse_url.py https://hdfc-secure-login.xyz/verify

# Run the built-in test suite for any module
python src/pipeline/parse_url.py
python src/pipeline/resolve_dns.py

# (coming) Full pipeline on a URL
python src/pipeline/build_graph.py https://target-domain.xyz

# (coming) Compute quantum fingerprint
python src/quantum/fingerprint.py --graph data/graphs/target.gpickle

# (coming) Cluster and attribute
python src/clustering/dbscan_cluster.py --embeddings results/phi_vectors.npy
```

---

## ◈ MISSION PROGRESS

```
  STAGE 0  ████████████████████  COMPLETE  Quantum walk theory + graph foundations
  STAGE 1a ████████████████████  COMPLETE  parse_url() — TLD-aware URL decomposition
  STAGE 1b ████░░░░░░░░░░░░░░░░  ACTIVE    resolve_dns() — DNS A record resolution
  STAGE 1c ░░░░░░░░░░░░░░░░░░░░  PENDING   fetch_whois() + fetch_cert()
  STAGE 1d ░░░░░░░░░░░░░░░░░░░░  PENDING   build_graph() — NetworkX assembly
  STAGE 2  ░░░░░░░░░░░░░░░░░░░░  PENDING   hamiltonian() + walk() — U(t)=expm(−iHt)
  STAGE 3  ░░░░░░░░░░░░░░░░░░░░  PENDING   fingerprint() — φ(G) extraction
  STAGE 4  ░░░░░░░░░░░░░░░░░░░░  PENDING   dbscan_cluster() — campaign attribution
  STAGE 5  ░░░░░░░░░░░░░░░░░░░░  PENDING   evaluate() — ARI/NMI/F1 + UMAP figure
```

---

## ◈ ACADEMIC CONTEXT

| | |
|---|---|
| **Method** | Continuous-time quantum walk on phishing infrastructure hypergraphs |
| **Fingerprint** | φ(G) = \|⟨j\|e^{−iHt}\|k⟩\|² — permutation-invariant topology vector |
| **Clustering** | DBSCAN on φ(G) embeddings (density-based, no fixed k required) |
| **Evaluation** | ARI, NMI, silhouette score vs GraphSAGE / GAT / lexical RF baselines |
| **Data** | PhishTank + passive DNS / WHOIS / crt.sh collection |
| **Target venues** | IEEE S&P · USENIX Security · CCS · NDSS |

---

## ◈ THE RULE

> Every function in `src/` — **you write the first draft.**  
> It will be wrong. We debug it together. You learn from the bug.  
> No copy-paste without comprehension. No moving forward until you can  
> delete and rewrite every line from understanding.

---

<div align="center">

*This repository is private during active research.*  
*Do not distribute. Do not share. Build first.*

```
[ EVELYN IS WATCHING ]
```

</div>