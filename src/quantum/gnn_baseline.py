# src/quantum/gnn_baseline.py
#
# PURPOSE: A minimal GraphSAGE-style embedding baseline, built to run
# on the EXACT SAME graphs (same nodes, same edges, same node-type
# features) that fingerprint.py consumes. This is your competitor —
# the thing your quantum walk method needs to beat or find blind
# spots in, per the Unit42-style GNN approach identified in our
# literature search.
#
# WHY "MINIMAL" IS THE RIGHT CHOICE RIGHT NOW:
#   A full GraphSAGE implementation (PyTorch Geometric, trained over
#   epochs with backprop) needs LABELED TRAINING DATA AT SCALE to be
#   a fair comparison — and Stage 1 only gave you ~20-30 real graphs
#   so far. Training a real GNN on 20 examples would just memorize
#   noise and tell you nothing.
#
#   This file implements the CORE MATHEMATICAL OPERATION of GraphSAGE
#   — neighbourhood feature aggregation — using RANDOM (untrained)
#   projection weights, exactly as the "blind spot" experiment needs:
#   we are not trying to prove our walk beats a FULLY TRAINED GNN on
#   accuracy (that needs Stage 5's full dataset). We ARE testing
#   whether GraphSAGE-style 1-2 hop aggregation, even with PERFECT
#   weights, can structurally distinguish graphs that our quantum
#   walk distinguishes — because if it CAN'T distinguish them even in
#   principle (i.e., the aggregated representations collapse to the
#   same vector regardless of weights), that's the structural blind
#   spot, not a training/data issue.
#
#   Once you have 500+ real labeled graphs (Stage 5), upgrade this to
#   a fully trained PyTorch Geometric GraphSAGE for the accuracy
#   comparison. This file gives you the STRUCTURAL comparison NOW.

import numpy as np
import networkx as nx


# Node-type one-hot encoding — must match the "type" attribute values
# set in build_graph.py exactly.
NODE_TYPES = ["domain", "ip", "registrar", "cert_peer", "asn", "geo", "co_host", "unknown"]


def _node_type_feature(node_data: dict) -> np.ndarray:
    """
    Converts a node's "type" attribute into a one-hot feature vector.
    This is the SIMPLEST possible node feature a GNN could use — real
    GraphSAGE deployments (like Unit42's) would also include domain
    age, TTL, cert validity, etc. as additional continuous features.
    We keep it to type-only here to isolate the STRUCTURAL aggregation
    behavior we're testing, not feature-engineering differences.
    """
    node_type = node_data.get("type", "unknown")
    vec = np.zeros(len(NODE_TYPES))
    if node_type in NODE_TYPES:
        vec[NODE_TYPES.index(node_type)] = 1.0
    else:
        vec[NODE_TYPES.index("unknown")] = 1.0
    return vec


def graphsage_embed(G: nx.Graph, n_layers: int = 2, embed_dim: int = 16,
                    seed: int = 0, aggregation: str = "mean") -> dict:
    """
    Computes a GraphSAGE-style graph-level embedding via repeated
    neighbourhood aggregation, using RANDOM (untrained) projection
    weights — see module docstring for why this is the right choice
    for the blind-spot experiment specifically.

    Algorithm (standard GraphSAGE forward pass, mean aggregator):
      h_v^0 = node_type_feature(v)                       for all v
      h_v^k = sigma(W_k @ [h_v^{k-1} ; mean_{u in N(v)} h_u^{k-1}])
      graph_embedding = mean_v h_v^{n_layers}

    Parameters:
        G           : NetworkX graph
        n_layers    : number of aggregation hops (2 = standard GraphSAGE depth,
                      meaning each node "sees" up to 2 hops away — this is
                      EXACTLY the known limitation we're probing)
        embed_dim   : output embedding dimension per node, per layer
        seed        : random seed for the (untrained) projection weights
        aggregation : "mean" or "max" — which neighbour-aggregation function

    Returns:
    {
        "graph_embedding": np.ndarray (embed_dim,) — fixed length regardless of n_nodes,
        "node_embeddings": dict {node: np.ndarray} — per-node embeddings, for inspection,
        "n_layers":         int,
        "n_nodes":          int,
    }
    """
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    n_nodes = len(nodes)

    if n_nodes == 0:
        return {"graph_embedding": np.zeros(embed_dim), "node_embeddings": {},
                "n_layers": n_layers, "n_nodes": 0}

    # Initial node features: one-hot type vectors
    h = {node: _node_type_feature(G.nodes[node]) for node in nodes}
    current_dim = len(NODE_TYPES)

    # Precompute neighbour lists once
    neighbors = {node: list(G.neighbors(node)) for node in nodes}

    for layer in range(n_layers):
        # A fresh random projection matrix per layer — this is the
        # UNTRAINED weight matrix. Input dim = current_dim (self) +
        # current_dim (aggregated neighbour features) = 2 * current_dim.
        W = rng.standard_normal((embed_dim, current_dim * 2)) * 0.1

        h_next = {}
        for node in nodes:
            self_feat = h[node]

            neighbor_feats = [h[n] for n in neighbors[node]]
            if neighbor_feats:
                if aggregation == "mean":
                    agg_feat = np.mean(neighbor_feats, axis=0)
                elif aggregation == "max":
                    agg_feat = np.max(neighbor_feats, axis=0)
                else:
                    raise ValueError(f"Unknown aggregation: {aggregation}")
            else:
                # Isolated node (no neighbours) — aggregate is zero vector
                agg_feat = np.zeros(current_dim)

            combined = np.concatenate([self_feat, agg_feat])
            # ReLU nonlinearity — standard in GraphSAGE
            h_next[node] = np.maximum(0, W @ combined)

        h = h_next
        current_dim = embed_dim

    # Graph-level embedding: mean-pool over all final node embeddings.
    # This pooling step is itself worth noting in your methods/limitations
    # section — mean pooling is a KNOWN additional source of information
    # loss on top of the 1-2 hop aggregation limit, and is exactly the
    # kind of detail a hostile reviewer will ask you to justify.
    graph_embedding = np.mean([h[node] for node in nodes], axis=0)

    return {
        "graph_embedding": graph_embedding,
        "node_embeddings": h,
        "n_layers":         n_layers,
        "n_nodes":          n_nodes,
    }


def embedding_distance(emb_a: np.ndarray, emb_b: np.ndarray, metric: str = "cosine") -> float:
    """Same interface as fingerprint_distance() in fingerprint.py, for fair comparison."""
    if metric == "cosine":
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)
        if norm_a == 0 or norm_b == 0:
            return 1.0
        return 1.0 - float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
    elif metric == "euclidean":
        return float(np.linalg.norm(emb_a - emb_b))
    else:
        raise ValueError(f"Unknown metric: {metric}")


# ── SELF-TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 58)
    print("  EVELYN — gnn_baseline()")
    print("=" * 58)

    # Test 1: basic sanity — does it run and produce fixed-length output
    # regardless of graph size?
    G_small = nx.complete_graph(3)
    G_large = nx.complete_graph(10)

    emb_small = graphsage_embed(G_small, seed=1)["graph_embedding"]
    emb_large = graphsage_embed(G_large, seed=1)["graph_embedding"]

    print(f"\n  Fixed-length output check:")
    print(f"    K3  embedding shape: {emb_small.shape}")
    print(f"    K10 embedding shape: {emb_large.shape}")
    print(f"    {'✓ PASS' if emb_small.shape == emb_large.shape else '✗ FAIL'} — same shape regardless of graph size")

    # Test 2: the KNOWN GNN limitation — two non-isomorphic graphs that
    # 1-2 hop mean-aggregation famously CANNOT distinguish: a single
    # 6-cycle (C6) vs two disjoint triangles (C3 + C3). Both are
    # 2-regular graphs (every node has exactly 2 neighbours), so EVERY
    # node's local neighbourhood looks IDENTICAL in both graphs — this
    # is the textbook example of a GNN expressiveness limitation
    # (related to the Weisfeiler-Leman test failure case).
    print(f"\n  Known GNN blind spot test: C6 (one 6-cycle) vs 2xC3 (two triangles)")
    G_c6 = nx.cycle_graph(6)
    G_2c3 = nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(3))

    emb_c6  = graphsage_embed(G_c6,  seed=1, n_layers=2)["graph_embedding"]
    emb_2c3 = graphsage_embed(G_2c3, seed=1, n_layers=2)["graph_embedding"]
    gnn_dist = embedding_distance(emb_c6, emb_2c3, metric="cosine")

    print(f"    GraphSAGE (2-layer) cosine distance: {gnn_dist:.6f}")
    print(f"    Expected: near-ZERO — this is the textbook 1-2 hop GNN blind spot")
    print(f"    (every node in both graphs has identical 2-hop neighbourhood structure)")