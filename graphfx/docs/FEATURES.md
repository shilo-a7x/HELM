# graphfx features

graphfx computes per-vertex and per-edge topological features of a NetworkX graph.
Most are plain NetworkX functions, accelerated on a GPU through `nx-cugraph` when it is
installed. The motif counts (`motif3`, `motif4`) come from the upstream `graph-measures`
package, which graphfx calls as a subprocess (see `motif_client.py`); `hierarchy_energy` is
SpringRank, computed with the `springrank` package (GPL-3.0).

```python
import graphfx

node_df = graphfx.compute_node_features(G)   # gpu=True by default; features=[...] selects a subset
edge_df = graphfx.compute_edge_features(G)
```

- `G` is any NetworkX graph; vertex labels need not be contiguous integers.
- The node table has a `node` column and the edge table `source` and `target` columns holding
  the original labels.
- A feature that needs an undirected graph is computed on the undirected view of a directed
  input; a feature that needs a directed graph is skipped for an undirected input.

Install with `scripts/setup.sh` at the repository root. Select the GPU with `CUDA_VISIBLE_DEVICES`.

## Default features

Node (15; the directed-only ones apply to directed graphs only):
`degree`, `in_degree`\*, `out_degree`\*, `average_neighbor_degree`, `louvain`, `leiden`, `k_core`,
`page_rank`, `eigenvector_centrality`, `katz_centrality`, `betweenness_centrality`,
`clustering_coefficient`, `hierarchy_energy`\* (SpringRank), `motif3`, `motif4`.

`louvain` and `leiden` are the size of the vertex's community, not its identifier.
`clustering_coefficient` is computed on the undirected view of the graph.

Edge (3): `edge_betweenness`, `jaccard_coefficient`, `preferential_attachment`.

\* directed graphs only.

## CPU and GPU differ in two features

On the GPU, `louvain` and `leiden` use the cuGraph implementations; on the CPU, NetworkX's.
They are different algorithms and give different partitions on larger graphs (on the wiki graphs
nearly every community-size value differs). Everything else agrees to about 1e-7.
Train and apply a model with features from the same backend. On the bundled datasets the
effect on the final metrics is small (mean directed recall changes by about 0.002 on wiki
and 0.01 on microbiome) but not zero.

## Opt-in features

These are too slow for the default set or have no use on the bundled graphs. Request them with
`features=[...]`.

- Node: `closeness_centrality`, `eccentricity`, `load_centrality`, `fiedler_vector`,
  `square_clustering_coefficient`, `triangle_count`, `weakly_connected_components`,
  `generalized_degree`, `communicability_betweenness_centrality` (O(n^4)), `trophic_incoherence`
  (NaN on graphs without in-degree-0 vertices, as here), `attractor_basin`, `flow`, `bfs_moments`.
- Edge: `edge_current_flow_betweenness`, `min_edge_cut`, `adamic_adar_index`,
  `resource_allocation_index`.

`edge_motif3` and `edge_motif4` are registered but not implemented.

## Determinism

Louvain and Leiden are seeded (`seed=42`). Katz centrality fixes the start vector of its
eigenvalue estimate. On the CPU repeated runs are bit-identical; on the GPU `betweenness_centrality`
varies in the last digits (about 1e-8) between runs.
