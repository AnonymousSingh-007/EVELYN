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
![Stage](https://img.shields.io/badge/STAGE-2_QUANTUM_WALK_VALIDATED-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/PYTHON-3.14-yellow?style=for-the-badge&logo=python)
![Venue Target](https://img.shields.io/badge/TARGET-IEEE_S%26P_%7C_USENIX-red?style=for-the-badge)
![License](https://img.shields.io/badge/ACCESS-PRIVATE_RESEARCH-black?style=for-the-badge)

</div>

---

## ◈ THE PROBLEM

Attackers spin up hundreds of phishing domains a day.  
Classical detectors look at the **name** — the string, the words, the TLD.  
So attackers change the name. Constantly. Cheaply. $0.88 per domain.

**EVELYN doesn't just look at the name.**

It looks at the **shape** — how domains share IPs, registrars, certificates,
TLS server fingerprints, and favicons. That shape is the attacker's true
signature. Rebuilding it costs time, money, and exposure.

**Novelty check (June 2026):** GNN-based infrastructure graph analysis
exists in industry (Unit42). Quantum-walk graph matching exists in theory
(Farhi & Guttmann 1998; Wang & Douglas). Quantum ML for phishing exists via
feature encoding (QSVM/QRAC). The intersection — quantum walk topology
fingerprinting for phishing campaign attribution, evaluated head-to-head
against GNN baselines on real data — appears open. See `docs/related_work.md`.

---

## ◈ THE WEAPON

At the core of EVELYN is a **continuous-time quantum walk** on a phishing
infrastructure hypergraph.

```
φ(G)  =  |⟨j| e^{−iHt} |k⟩|²   for all j,k ∈ V,  t ∈ {t₁,...,tₘ}
```

Where:
- `H = −A` — the Hamiltonian (negative adjacency matrix)
- `e^{−iHt}` — wave evolution through the graph over time `t`
- `|⟨j|...|k⟩|²` — interference amplitude
- `φ(G)` — the topology fingerprint, **invariant to node labels**

**Verified, not just claimed (Stage 2 complete):**
- Hamiltonian eigenvalues match analytical solution on K3 exactly (`-2, 1, 1`)
- Eigendecomposition method agrees with `scipy.linalg.expm` to 1e-15
- Unitarity (`U·U† = I`) holds even at t=1000 (numerical stability confirmed)
- Fingerprint is permutation-invariant to machine precision (`~1e-8` diff after
  random node relabeling) — the literal computational proof of "the wave
  doesn't know domain names, only structure"

**First real evidence beyond the concept:** a minimal GraphSAGE baseline
(`src/quantum/gnn_baseline.py`) reproduces the textbook 1-2 hop GNN blind
spot (C6 ring vs. two disjoint triangles → GraphSAGE distance = 0.000000).
The quantum walk fingerprint correctly distinguishes this pair (distance =
0.191) and a hub-topology pair relevant to real campaign structure
(single-star vs. double-star, distance = 0.105). See
`src/quantum/compare_blind_spots.py`.

---

## ◈ ARCHITECTURE

```
RAW URL
  │
  ▼
fetch_redirect_chain()  ◄── resolves TRUE final domain first
  │                          (catches cloaking, shorteners, free-host fronts)
  ▼
parse_url() ──► resolve_dns() ──► fetch_whois() ──► fetch_cert()
                                                       (crt.sh + CertSpotter
                                                        fallback, 7-day cache)
  │
  ├──► fetch_ssl_meta()        live TLS cert validity/self-signed check
  ├──► fetch_page_meta()       HTTP headers, favicon hash, title/brand mismatch
  ├──► fetch_subdomains()      CT logs + targeted DNS wordlist
  ├──► fetch_jarm()             TLS server software fingerprint (kit reuse)
  │
  ▼ (per discovered IP)
  fetch_asn() + score_asn()    ASN reputation vs. curated bulletproof-hosting list
  fetch_geo()
  check_shared_hosting() ──► suspicion_filter()   (gates co-hosts: must show
                                                    independent suspicion signal,
                                                    not just shared infrastructure)
  │
  ▼
build_graph()  ──►  G_i  (NetworkX, node-budget capped at 50)
  │
  ▼ (optional)
build_graph_recursive()  ──► multi-hop campaign expansion, suspicion-gated
  │
  ▼
hamiltonian()  H = −A  (eigendecomposition + Hermitian validation)
  │
  ▼
walk()  U(t) = expm(−iHt)  (multi-t, cross-validated vs scipy)
  │
  ▼
fingerprint()  φ(G)  ◄── permutation-invariant, fixed-length regardless of n
  │
  ├──► gnn_baseline()  GraphSAGE comparison ──► compare_blind_spots()
  │
  ▼
dbscan_cluster()  [Stage 4 — pending]
  │
  ▼
ATTRIBUTION:  ✓ Known campaign  /  ⚠ Novel campaign  /  ○ Benign
```

---

## ◈ WHY QUANTUM WALKS BEAT GNNs

| Property | GraphSAGE / GAT | EVELYN φ(G) |
|---|---|---|
| Captures global structure | ✗ 1–2 hop only | ✓ Full spectral encoding |
| Permutation invariant | ✗ Training artifact | ✓ Provably, verified to 1e-8 |
| Zero-shot on new topologies | ✗ Needs labeled examples | ✓ Unsupervised fingerprint |
| Fixed-dim output for variable graphs | ✓ (via pooling) | ✓ |
| C6 vs. 2×C3 (known GNN blind spot) | ✗ distance = 0.000 | ✓ distance = 0.191 |
| Cost | O(n·d) per layer | O(n³) bounded by ego-graph (n≤50) |

**One-sentence reviewer answer:**
> *"For campaign attribution where node labels change but topology is stable,
> quantum walk fingerprints are provably superior because they are
> topology-invariant by construction, not by training — and we demonstrate
> this concretely on structures where GraphSAGE-style aggregation collapses
> to identical representations."*

---

## ◈ REPOSITORY

```
EVELYN/
│
├── src/
│   ├── pipeline/
│   │   ├── parse_url.py             ✅  URL → domain / TLD / subdomain / IP flag
│   │   ├── fetch_redirect_chain.py  ✅  resolve TRUE final URL (runs first)
│   │   ├── resolve_dns.py           ✅  domain → IP node (A record + TTL)
│   │   ├── fetch_whois.py           ✅  domain → registrar node + domain age
│   │   ├── fetch_cert.py            ✅  crt.sh + CertSpotter fallback + cache
│   │   ├── cert_cache.py            ✅  7-day on-disk cache (zero-cost rate-limit fix)
│   │   ├── fetch_ssl_meta.py        ✅  live TLS validity / self-signed check
│   │   ├── fetch_page_meta.py       ✅  headers, favicon hash, title/brand mismatch
│   │   ├── fetch_subdomains.py      ✅  CT logs + targeted DNS wordlist
│   │   ├── fetch_jarm.py            ✅  TLS server software fingerprint
│   │   ├── fetch_asn.py             ✅  IP → ASN / hosting provider
│   │   ├── asn_reputation.py        ✅  ASN vs. curated bulletproof-hosting list
│   │   ├── fetch_geo.py             ✅  IP → geolocation
│   │   ├── check_shared_hosting.py  ✅  IP → co-hosted domains
│   │   ├── suspicion_filter.py      ✅  gates co-hosts by independent signal
│   │   ├── fetch_phishtank.py       ✅  PhishTank dataset loader
│   │   ├── fetch_benign.py          ✅  Tranco benign dataset loader
│   │   ├── build_graph.py           ✅  assembles G_i (11 modules, budget-capped)
│   │   ├── build_graph_recursive.py ✅  multi-hop campaign expansion
│   │   └── batch_pipeline.py        ✅  concurrent batch runner, checkpointed
│   │
│   ├── quantum/
│   │   ├── hamiltonian.py           ✅  G_i → H = −A (verified vs. K3 analytically)
│   │   ├── walk.py                  ✅  H → U(t)=expm(−iHt) (cross-validated)
│   │   ├── fingerprint.py           ✅  U(t) → φ(G) (permutation-invariance proven)
│   │   ├── gnn_baseline.py          ✅  minimal GraphSAGE for comparison
│   │   └── compare_blind_spots.py   ✅  quantum vs. GNN structural blind-spot detector
│   │
│   ├── clustering/
│   │   ├── dbscan_cluster.py        ⏳  φ(G) vectors → campaign clusters
│   │   └── evaluate.py              ⏳  ARI / NMI / F1 + UMAP visualisation
│   │
│   └── viz/
│       └── paper_figures.py         ✅  Q1-style figures, PNG+PDF, per-stage
│
├── data/
│   ├── raw/            ← PhishTank + Tranco CSVs              [gitignored]
│   ├── processed/      ← batch_results.csv, checkpoints       [gitignored]
│   └── graphs/         ← Serialised NetworkX graph objects    [gitignored]
│
├── docs/
│   └── related_work.md ← Novelty-gap tracking vs. Unit42/QML literature
├── notebooks/          ← Exploratory Jupyter notebooks
├── experiments/        ← Saved run configs, hyperparameter logs
├── results/figures/    ← UMAP plots, paper figures (PNG+PDF)
├── results/metrics/    ← ARI / NMI / F1 CSVs per experiment
├── tests/               ← Unit tests for every src/ function
├── requirements.txt
└── README.md
```

---

## ◈ SETUP

```bash
git clone https://github.com/YOUR_USERNAME/EVELYN.git
cd EVELYN
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

---

## ◈ USAGE

All pipeline modules run as Python modules from the project root
(`python -m src.pipeline.X`, not `python src/pipeline/X.py`) — this is
required for the internal `from src...` imports to resolve correctly.

```bash
# Single-domain full pipeline (11 modules)
python -m src.pipeline.build_graph https://suspicious-domain.xyz 1

# Multi-hop campaign expansion (suspicion-gated)
python -m src.pipeline.build_graph_recursive https://suspicious-domain.xyz 1

# Batch process a labeled dataset
python -m src.pipeline.batch_pipeline

# Quantum walk validation suite
python -m src.quantum.hamiltonian
python -m src.quantum.walk
python -m src.quantum.fingerprint
python -m src.quantum.gnn_baseline
python -m src.quantum.compare_blind_spots --real   # against your saved graphs

# Generate paper figures
python -m src.viz.paper_figures --demo
```

---

## ◈ MISSION PROGRESS

```
  STAGE 0  ████████████████████  COMPLETE  Quantum walk theory + graph foundations
  STAGE 1a ████████████████████  COMPLETE  parse_url() — TLD-aware URL decomposition
  STAGE 1b ████████████████████  COMPLETE  resolve_dns(), fetch_whois(), fetch_cert()
  STAGE 1c ████████████████████  COMPLETE  ASN/geo/shared-hosting + suspicion filter
  STAGE 1d ████████████████████  COMPLETE  build_graph() — 11-module ego-graph assembly
  STAGE 1e ████████████████████  COMPLETE  JARM, subdomains, page-meta, redirect chain,
                                            ASN reputation, recursive expansion
  STAGE 2  ████████████████████  COMPLETE  hamiltonian()+walk()+fingerprint() — numerically
                                            validated, permutation-invariance proven
  STAGE 2b ████████████░░░░░░░░  ACTIVE    GNN baseline + blind-spot comparison
                                            (synthetic suite passing 2/3; real-graph
                                            run pending larger dataset)
  STAGE 3  ░░░░░░░░░░░░░░░░░░░░  PENDING   Scale dataset (500+ phishing, 500+ benign)
  STAGE 4  ░░░░░░░░░░░░░░░░░░░░  PENDING   dbscan_cluster() — campaign attribution
  STAGE 5  ░░░░░░░░░░░░░░░░░░░░  PENDING   evaluate() — ARI/NMI/F1 + UMAP figure,
                                            full GNN baseline accuracy comparison
```

---

## ◈ ACADEMIC CONTEXT

| | |
|---|---|
| **Method** | Continuous-time quantum walk on phishing infrastructure hypergraphs |
| **Fingerprint** | φ(G) = \|⟨j\|e^{−iHt}\|k⟩\|² — permutation-invariant, eigenvalue-histogram + return-probability summary |
| **Baseline** | Minimal untrained GraphSAGE (structural comparison) → full PyTorch Geometric GraphSAGE at Stage 5 scale |
| **Clustering** | DBSCAN on φ(G) embeddings (density-based, no fixed k required) |
| **Evaluation** | ARI, NMI, silhouette score; blind-spot detection vs. GNN baseline |
| **Data** | PhishTank + Tranco (benign) + passive DNS/WHOIS/crt.sh/CertSpotter/RDAP/JARM |
| **Reputation data** | Curated bulletproof-hosting ASN list (bountyyfi/bad-asn-list, CISA/NSA/FBI 2025 guidance) |
| **Target venues** | IEEE S&P · USENIX Security · CCS · NDSS |
| **Known limitations** | crt.sh free-tier unreliability (mitigated via cache + CertSpotter fallback); free-hosting-platform phishing (Netlify/Vercel) weakens WHOIS/cert signals at the platform-domain level; bulletproof-hosting ASN lists are inherently incomplete (operators rotate ASNs) |

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