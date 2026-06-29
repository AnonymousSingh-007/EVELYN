<div align="center">

```
███████╗██╗   ██╗███████╗██╗  ██╗   ██╗███╗   ██╗
██╔════╝██║   ██║██╔════╝██║  ╚██╗ ██╔╝████╗  ██║
█████╗  ██║   ██║█████╗  ██║   ╚████╔╝ ██╔██╗ ██║
██╔══╝  ╚██╗ ██╔╝██╔══╝  ██║    ╚██╔╝  ██║╚██╗██║
███████╗ ╚████╔╝ ███████╗███████╗██║   ██║ ╚████║
╚══════╝  ╚═══╝  ╚══════╝╚══════╝╚═╝   ╚═╝  ╚═══╝
```

### **E**xploit **V**ector **E**stimation & **L**atent **Y**ield **N**etwork

*Quantum graph-theoretic estimation of hidden phishing infrastructure.*
*You found 80% of the network. EVELYN estimates the rest.*

![Status](https://img.shields.io/badge/STATUS-ACTIVE_RESEARCH-brightgreen?style=for-the-badge)
![Stage](https://img.shields.io/badge/STAGE-PIVOT_TO_COMPLETION-orange?style=for-the-badge)
![Python](https://img.shields.io/badge/PYTHON-3.14-yellow?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/LICENSE-RESEARCH_USE-black?style=for-the-badge)

</div>

---

## What this is (updated direction)

EVELYN started as a phishing-detection project comparing a quantum-walk
graph fingerprint against a GNN baseline. That comparison produced a
real, useful, but ultimately **inconclusive** result: on real phishing
infrastructure graphs, the two methods mostly agreed with each other,
and where they disagreed, it wasn't reliably in either method's favor.
Full history of that work is preserved in
[`docs/findings.md`](docs/findings.md) — nothing is hidden, including
the parts that didn't pan out as hoped.

**The project has pivoted to a sharper, more useful, and more novel
question**, prompted directly by that inconclusive result:

> When a threat hunter has discovered *part* of a phishing campaign's
> infrastructure — say, one domain and its IP — but not the rest of it,
> can a quantum-walk-based model **estimate what the missing
> infrastructure looks like**, before it's been found?

This is **graph completion / missing-link estimation**, not
classification. It asks a forward-looking question — "what does the
unseen 20% of this network probably contain" — rather than a
backward-looking one — "is this known thing phishing or not."

> **Why this is novel.** Quantum-walk-based missing-link prediction is
> an active, real research area (Liang et al. 2022; Goldsmith et al.
> 2023; Moutinho et al. 2023, *Phys. Rev. A*) — but every paper found
> in our literature search applies it where the **node set is already
> fully known** (e.g., "will these two existing users connect?").
> None apply it to **discovering unknown infrastructure nodes that
> haven't been observed yet** — predicting that a *currently unknown*
> IP or domain probably exists and roughly what role it plays, from a
> partial observation of a phishing campaign. That specific
> combination — quantum walk estimation applied to *undiscovered*
> attacker infrastructure — does not appear in the literature as of
> this writing. See [`docs/related_work.md`](docs/related_work.md).

---

## The experiment this pivot enables

Because EVELYN can already build real multi-hop campaign graphs
(`build_graph_recursive.py`), testing this hypothesis doesn't require
new data collection — it requires a new evaluation protocol:

```
1. Take a real, fully-expanded campaign graph (e.g. 40 nodes)
2. Deliberately hide 20% of it (e.g. remove 8 nodes + their edges)
3. Ask the quantum-walk-based estimator: "what's missing?"
4. Check: did it correctly estimate where/what the hidden nodes were?
```

This is directly analogous to how link-prediction papers validate
their models (hide real edges, see if the model recovers them) — we
extend it to hidden *nodes*, which is the harder and less-explored
version of the problem.

---

## What's preserved from the original detection-comparison work

The full quantum walk implementation, the GraphSAGE baseline, and the
11-module threat-hunting pipeline are unchanged and still the
foundation everything above is built on:

```
RAW URL
  │
  ▼
fetch_redirect_chain()   ◄── resolves the TRUE final domain first
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│  11 THREAT-HUNTING SIGNAL MODULES                            │
│  DNS · WHOIS · Certificate transparency (crt.sh + CertSpotter)│
│  SSL metadata · Page metadata (favicon / headers / brand)     │
│  Subdomain enumeration · JARM TLS server fingerprint           │
│  ASN reputation (bulletproof-hosting list) · Geolocation       │
│  Shared hosting → gated by an independent suspicion filter     │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
build_graph() / build_graph_recursive()  ──►  G_i  (NetworkX hypergraph)
  │
  ▼
hamiltonian()   H = −A          (eigendecomposition, Hermitian-validated)
  │
  ▼
walk()          U(t) = e^{−iHt}  (cross-validated against scipy to 1e-15)
  │
  ▼
fingerprint()   φ(G) = |⟨j|U(t)|k⟩|²   ◄── permutation-invariant, proven
  │                                         to 1e-8 by direct relabeling test
  ▼
[NEW] estimator()   predicts hidden nodes/edges from a partial graph
```

---

## The math, briefly

```
φ(G)  =  |⟨j| e^{−iHt} |k⟩|²    for all j,k ∈ V,  t ∈ {t₁,...,tₘ}
```

- `H = −A` — the Hamiltonian, the negative adjacency matrix of the graph
- `e^{−iHt}` — the wave's evolution through the graph over time `t`
- `|⟨j|...|k⟩|²` — the probability the wave travels from node `k` to node `j`
- `φ(G)` — the resulting fingerprint, fixed-length regardless of graph size

For missing-link/node estimation, the same `|⟨j|U(t)|k⟩|²` quantity
becomes a **score**: a high value for a `(j,k)` pair that ISN'T yet
a known edge is a signal that a connection probably exists there but
hasn't been discovered — this is precisely the mechanism used in the
2022/2023 link-prediction papers cited above, adapted here to score
*candidate undiscovered infrastructure* rather than known social pairs.

**This is not asserted — it is computationally verified in this repo**:

| Property | Test | Result |
|---|---|---|
| Hamiltonian correctness | Eigenvalues vs. analytical solution on K3 | Exact match (`-2, 1, 1`) |
| Numerical stability | Eigendecomposition vs. `scipy.linalg.expm` | Agree to `8.4e-16` |
| Unitarity (`U·U† = I`) | Tested at t up to 1000 | Holds at every value |
| **Permutation invariance** | Random node relabeling, fingerprint recomputed | Identical to `~1e-8` |

```bash
python -m src.quantum.hamiltonian
python -m src.quantum.walk
python -m src.quantum.fingerprint
```

---

## What we learned from the detection-comparison phase (preserved, not deleted)

| Property | GraphSAGE / GAT | EVELYN φ(G) |
|---|---|---|
| Sees beyond 1–2 hops | ✗ | ✓ full-graph spectral view |
| Permutation invariant | ✗ (training artifact) | ✓ provable, verified to 1e-8 |
| C6 ring vs. two separate triangles (synthetic, textbook case) | ✗ distance = 0.000 | ✓ distance = 0.191 |
| **28 real phishing/benign pairs, head-to-head** | agreed with quantum walk on 23/28 | agreed with GNN on 23/28 |

The synthetic advantage is real and reproducible. It did **not**
reliably transfer to single-domain real phishing graphs, where both
methods mostly agreed, and where they disagreed, results were mixed
rather than favoring one method. Full numbers in `docs/findings.md`.
This negative-leaning result is what motivated the pivot above —
treating the two methods as a classification horse race wasn't
producing a clear or useful signal; reframing the problem as
estimation/completion gives the quantum walk's unique mathematical
property (global structural sensitivity) a job that node-feature
methods genuinely cannot do at all, rather than a job they merely do
differently.

---

## Repository layout

```
EVELYN/
├── src/
│   ├── pipeline/        11 signal-collection modules + graph assembly
│   ├── quantum/          Hamiltonian, walk, fingerprint, GNN baseline,
│   │                     [new] missing-node/edge estimator
│   ├── clustering/       (pending) campaign attribution
│   └── viz/               Stage figures, comparison figures, 3D explorer
├── data/
│   ├── raw/              PhishTank + Tranco source CSVs      [gitignored]
│   ├── processed/        Batch run results, checkpoints      [gitignored]
│   └── graphs/           Serialized graph objects            [gitignored]
├── docs/
│   ├── related_work.md   Literature positioning, novelty check
│   └── findings.md       Honest results log — the full history,
│                         including the pivot reasoning
├── results/figures/      Generated PNG/PDF figures            [committed]
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/EVELYN.git
cd EVELYN
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

## Usage

```bash
# Full 11-module pipeline on one domain
python -m src.pipeline.build_graph https://suspicious-domain.xyz 1

# Multi-hop campaign expansion (the input the estimator will use)
python -m src.pipeline.build_graph_recursive https://suspicious-domain.xyz 1

# Generate the four evidence figures for one domain
python -m src.viz.stage_figures https://suspicious-domain.xyz 1

# Run the quantum-vs-GNN historical comparison (preserved for reference)
python -m src.quantum.compare_blind_spots --real
```

---

## Limitations (stated plainly, not buried)

- **Free-hosting-platform phishing** (Netlify, Vercel, Weebly, Lovable)
  weakens domain-level WHOIS/cert signals, since most collected
  infrastructure describes the platform, not the attacker.
- **Free CT-log APIs (crt.sh, CertSpotter) are unreliable** under load;
  mitigated with a dual-source fallback and a 7-day local cache.
- **Bulletproof-hosting ASN lists are inherently incomplete** —
  operators rotate ASNs specifically to evade lists like the one used here.
- **The estimation/completion direction is new** as of this writing and
  has not yet been validated on real held-out infrastructure — the
  hide-and-recover protocol described above is the next concrete step.
- **Live investigation of currently-active malicious infrastructure**
  frequently triggers antivirus/firewall interception on a standard
  development machine — by design, since this is the correct security
  posture for most users.

---

## Academic context

| | |
|---|---|
| **Method** | Continuous-time quantum walk on phishing infrastructure hypergraphs |
| **Original framing** | Topology fingerprint for phishing classification (see `findings.md`) |
| **Current framing** | Quantum-walk-based estimation of undiscovered infrastructure nodes/edges |
| **Closest prior work** | Liang et al. 2022; Goldsmith et al. 2023; Moutinho et al. 2023 — all on known-node link prediction, not undiscovered-node estimation |
| **Data** | PhishTank (phishing) + Tranco (benign), passive DNS/WHOIS/CT/RDAP/JARM |
| **Target venues** | IEEE S&P · USENIX Security · CCS · NDSS |

---

<div align="center">

*Research project. Not production security software.*
*Findings are reported as they occur, including the ones that prompted a pivot.*

</div>