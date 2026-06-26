# EVELYN — Findings Log

This is an honest, dated log of what each experiment actually showed —
including results that complicate the core hypothesis. A research log
that only records wins is not a research log.

---

## 2026-06 — Quantum walk vs. GraphSAGE, real corpus (8 domains, 28 pairs)

**Setup:** 2 real phishing domains (PhishTank), 6 benign domains
(Tranco top sites), full 11-module pipeline, suspicion-filtered graphs.

**Result:** 5/28 pairs showed a quantum-walk-detects /
GraphSAGE-misses blind spot.

**Important caveat:** all 5 blind-spot pairs involved one specific
domain (`nytimes.com`) whose DNS lookup timed out during collection,
leaving it an unusually sparse 12-node graph compared to every other
domain in the corpus. This means the observed sensitivity is currently
demonstrated for *general structural sparsity*, not yet specifically
for *phishing campaign topology*. This is a real, reproducible result —
but it is a data-completeness finding, not (yet) a phishing-specific one.

## 2026-06 — A case where the GNN baseline won

**Setup:** `viaverde-seguranca.com`, a genuinely fresh (WHOIS age = 1
day), self-hosted phishing domain, compared against 4 benign domains.

**Result:** GraphSAGE consistently flagged this domain as structurally
different from every benign comparison (distances 0.32–0.40). The
quantum walk saw it as nearly identical to all of them (distances
0.02–0.03).

**Why this matters:** this is the opposite of the paper's core claim,
on real data, and it's reported here rather than omitted. The likely
cause: this domain's graph was missing 2 of 11 signal modules (SSL,
PageMeta — blocked by local antivirus during collection), leaving a
thin, sparse, mostly-co-host-filler graph. GraphSAGE's simple node-type
one-hot features may pick up *absence of node types* as a feature
difference more readily than the walk's spectral view does on a sparse
graph. This needs a controlled re-test with all 11 modules succeeding,
and ideally a multi-domain *campaign* graph (via `build_graph_recursive.py`)
rather than a single isolated domain, since the walk's theoretical
advantage (per the synthetic C6-vs-2xC3 test) is specifically about
seeing structure beyond 1-2 hops — which a single isolated domain's
graph may not have enough of to demonstrate.

## 2026-06 — Synthetic blind-spot suite (controlled, not real-world)

**Setup:** Three pairs of graphs with mathematically known structural
relationships:
1. C6 ring vs. two disjoint triangles (textbook GNN expressiveness limit)
2. Single star (1 hub, 8 leaves) vs. double star (2 hubs, 4 leaves each)
3. Synthetic "shared-IP campaign" vs. "shared-cert campaign" pattern

**Result:** 2/3 showed the predicted blind spot. The third pair (#3)
showed BOTH methods agreeing the graphs were similar — on inspection,
this was because the two synthetic graphs were accidentally isomorphic
in structure (same shape, different edge labels), not a meaningful
test case. This was a flaw in the test design, not a finding, and is
recorded here for transparency rather than quietly fixed and forgotten.

---

## Open questions this raises, in priority order

1. Does the quantum walk's advantage hold specifically on multi-domain
   campaign graphs (shared infrastructure across several domains), as
   opposed to single isolated domains? Untested as of this writing.
2. How much of the `nytimes.com` result is "sparse graph" vs.
   "genuinely different structure"? Needs a controlled comparison where
   graph size/completeness is held constant across the pair.
3. Does the result change once all 11 modules succeed for every domain
   in the corpus (i.e., once antivirus/network issues are eliminated
   via an isolated research environment)?