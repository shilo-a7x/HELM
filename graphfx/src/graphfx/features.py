"""Feature registry: name -> metadata describing applicability, cost, and backend.

See docs/FEATURES.md for the feature list. This module is pure
metadata (no computation) so pipeline.py/api.py can plan work without importing
every backend.
"""

from dataclasses import dataclass
from enum import Enum


class Directedness(Enum):
    DIRECTED = "directed"
    UNDIRECTED = "undirected"
    EITHER = "either"


class CostTier(Enum):
    CHEAP = "cheap"  # sub-second at our current graph sizes (<10K nodes)
    MODERATE = "moderate"  # low single-digit seconds
    EXPENSIVE = "expensive"  # superlinear / opt-in only


class Backend(Enum):
    NETWORKX = "networkx"  # plain nx.*, transparently GPU-accelerated via nx-cugraph when configured
    HIERARCHY = "hierarchy"  # graphfx.hierarchy (springrank / trophic levels)
    MOTIF = "motif"  # graphfx.motif_client, subprocess boundary to graph-measures


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    directedness: Directedness
    backend: Backend
    cost: CostTier
    opt_in: bool = False  # excluded from the default feature set unless explicitly requested


NODE_FEATURES: dict[str, FeatureSpec] = {
    "degree": FeatureSpec("degree", Directedness.EITHER, Backend.NETWORKX, CostTier.CHEAP),
    "in_degree": FeatureSpec("in_degree", Directedness.DIRECTED, Backend.NETWORKX, CostTier.CHEAP),
    "out_degree": FeatureSpec("out_degree", Directedness.DIRECTED, Backend.NETWORKX, CostTier.CHEAP),
    "average_neighbor_degree": FeatureSpec(
        "average_neighbor_degree", Directedness.EITHER, Backend.NETWORKX, CostTier.CHEAP
    ),
    "louvain": FeatureSpec("louvain", Directedness.UNDIRECTED, Backend.NETWORKX, CostTier.CHEAP),
    "leiden": FeatureSpec("leiden", Directedness.UNDIRECTED, Backend.NETWORKX, CostTier.MODERATE),
    # Undirected only: cugraph.leiden() (26.8) raises on directed input despite its documentation,
    # and nx.community.leiden_communities rejects directed graphs outright. See nx_backend.leiden().
    "k_core": FeatureSpec("k_core", Directedness.EITHER, Backend.NETWORKX, CostTier.MODERATE),
    "page_rank": FeatureSpec("page_rank", Directedness.EITHER, Backend.NETWORKX, CostTier.CHEAP),
    "eigenvector_centrality": FeatureSpec(
        "eigenvector_centrality", Directedness.EITHER, Backend.NETWORKX, CostTier.CHEAP
    ),
    "katz_centrality": FeatureSpec("katz_centrality", Directedness.EITHER, Backend.NETWORKX, CostTier.CHEAP),
    # opt_in, EXPENSIVE: nx-cugraph has no closeness_centrality dispatch, so it runs on the CPU only
    # (18 s on a 3031-vertex, 83K-edge graph).
    "closeness_centrality": FeatureSpec(
        "closeness_centrality", Directedness.EITHER, Backend.NETWORKX, CostTier.EXPENSIVE, opt_in=True
    ),
    "betweenness_centrality": FeatureSpec(
        "betweenness_centrality", Directedness.EITHER, Backend.NETWORKX, CostTier.MODERATE
    ),
    "eccentricity": FeatureSpec(
        "eccentricity", Directedness.EITHER, Backend.NETWORKX, CostTier.EXPENSIVE, opt_in=True
    ),
    "load_centrality": FeatureSpec(
        "load_centrality", Directedness.EITHER, Backend.NETWORKX, CostTier.EXPENSIVE, opt_in=True
    ),
    # opt_in, EXPENSIVE: no nx-cugraph dispatch for fiedler_vector -- CPU-only
    # eigsh solve, 37.5s on the real reference graph. Slowest node feature besides
    # motif4 -- too slow for the default set.
    "fiedler_vector": FeatureSpec(
        "fiedler_vector", Directedness.UNDIRECTED, Backend.NETWORKX, CostTier.EXPENSIVE, opt_in=True
    ),
    # UNDIRECTED (forced internally, always applicable -- see applicable()):
    # nx.clustering's directed (Fagiolo) formula measured slope 1.71 across
    # 60 real graphs (up to 7.3s), vs. 0.48 (<0.1s) for the undirected
    # formula. Unlike square_clustering_coefficient below, forcing undirected
    # here is a clean win, not a tradeoff -- see nx_backend.clustering_coefficient.
    "clustering_coefficient": FeatureSpec(
        "clustering_coefficient", Directedness.UNDIRECTED, Backend.NETWORKX, CostTier.CHEAP
    ),
    # opt_in, EXPENSIVE: slope 1.29 (directed) / 1.44 (undirected) across 60
    # real graphs -- steep either way (forcing undirected doesn't help, it's
    # actually the worse case here, unlike clustering_coefficient above).
    "square_clustering_coefficient": FeatureSpec(
        "square_clustering_coefficient", Directedness.EITHER, Backend.NETWORKX, CostTier.EXPENSIVE, opt_in=True
    ),
    "triangle_count": FeatureSpec(
        "triangle_count", Directedness.UNDIRECTED, Backend.NETWORKX, CostTier.CHEAP, opt_in=True
    ),
    # opt_in: equal to motifs_node_3_2 on undirected graphs, and retraining without it showed no
    # reliable difference downstream.
    "weakly_connected_components": FeatureSpec(
        "weakly_connected_components", Directedness.DIRECTED, Backend.NETWORKX, CostTier.CHEAP, opt_in=True
    ),
    # (weakly_connected_components above is constant on the bundled graphs, which are connected.)
    # opt_in, EXPENSIVE: running time grows roughly as E^1.4, the steepest of the node features.
    "generalized_degree": FeatureSpec(
        "generalized_degree", Directedness.UNDIRECTED, Backend.NETWORKX, CostTier.EXPENSIVE, opt_in=True
    ),
    "communicability_betweenness_centrality": FeatureSpec(
        "communicability_betweenness_centrality",
        Directedness.UNDIRECTED,
        Backend.NETWORKX,
        CostTier.EXPENSIVE,
        opt_in=True,
    ),
    "hierarchy_energy": FeatureSpec(
        "hierarchy_energy", Directedness.DIRECTED, Backend.HIERARCHY, CostTier.CHEAP
    ),
    "trophic_incoherence": FeatureSpec(
        "trophic_incoherence", Directedness.DIRECTED, Backend.HIERARCHY, CostTier.CHEAP, opt_in=True
    ),  # opt_in: needs at least one basal (in-degree 0) vertex; all-NaN on the bundled graphs
    "attractor_basin": FeatureSpec(
        "attractor_basin", Directedness.DIRECTED, Backend.NETWORKX, CostTier.EXPENSIVE, opt_in=True
    ),
    "flow": FeatureSpec("flow", Directedness.DIRECTED, Backend.NETWORKX, CostTier.EXPENSIVE, opt_in=True),
    # opt_in, EXPENSIVE: calls single_source_shortest_path_length once per vertex in a Python loop,
    # so per-call overhead dominates and the GPU brings no speed-up.
    "bfs_moments": FeatureSpec(
        "bfs_moments", Directedness.EITHER, Backend.NETWORKX, CostTier.EXPENSIVE, opt_in=True
    ),
    "motif3": FeatureSpec("motif3", Directedness.EITHER, Backend.MOTIF, CostTier.CHEAP),
    "motif4": FeatureSpec("motif4", Directedness.EITHER, Backend.MOTIF, CostTier.MODERATE),
}

EDGE_FEATURES: dict[str, FeatureSpec] = {
    "edge_betweenness": FeatureSpec(
        "edge_betweenness", Directedness.EITHER, Backend.NETWORKX, CostTier.MODERATE
    ),
    "edge_current_flow_betweenness": FeatureSpec(
        "edge_current_flow_betweenness",
        Directedness.UNDIRECTED,
        Backend.NETWORKX,
        CostTier.EXPENSIVE,
        opt_in=True,
    ),
    "min_edge_cut": FeatureSpec(
        "min_edge_cut", Directedness.EITHER, Backend.NETWORKX, CostTier.EXPENSIVE, opt_in=True
    ),
    # Link-prediction-style similarity scores, computed only over EXISTING edges
    # (ebunch=G.edges(), not the full non-edge candidate set these functions
    # default to) -- O(E) with degree/neighbor lookups, cheap. Undirected only:
    # networkx's own implementations are @not_implemented_for("directed").
    "jaccard_coefficient": FeatureSpec(
        "jaccard_coefficient", Directedness.UNDIRECTED, Backend.NETWORKX, CostTier.CHEAP
    ),
    # opt_in, EXPENSIVE: slope 1.29 across 60 real graphs (R^2=0.96).
    "adamic_adar_index": FeatureSpec(
        "adamic_adar_index", Directedness.UNDIRECTED, Backend.NETWORKX, CostTier.EXPENSIVE, opt_in=True
    ),
    "preferential_attachment": FeatureSpec(
        "preferential_attachment", Directedness.UNDIRECTED, Backend.NETWORKX, CostTier.CHEAP
    ),
    # opt_in, EXPENSIVE: slope 1.29 across 60 real graphs (R^2=0.96).
    "resource_allocation_index": FeatureSpec(
        "resource_allocation_index", Directedness.UNDIRECTED, Backend.NETWORKX, CostTier.EXPENSIVE, opt_in=True
    ),
    # edge_motif3/4 are registered but not implemented: compute_edge_features raises
    # NotImplementedError for them, so they stay out of the default set.
    "edge_motif3": FeatureSpec(
        "edge_motif3", Directedness.EITHER, Backend.MOTIF, CostTier.MODERATE, opt_in=True
    ),
    "edge_motif4": FeatureSpec(
        "edge_motif4", Directedness.EITHER, Backend.MOTIF, CostTier.MODERATE, opt_in=True
    ),
}

DEFAULT_NODE_FEATURES = [name for name, spec in NODE_FEATURES.items() if not spec.opt_in]
DEFAULT_EDGE_FEATURES = [name for name, spec in EDGE_FEATURES.items() if not spec.opt_in]


def applicable(spec: FeatureSpec, directed: bool) -> bool:
    """UNDIRECTED features apply to every graph: a directed input is reduced to its undirected view
    by the feature's own backend function, never dropped."""
    if spec.directedness is Directedness.EITHER:
        return True
    if spec.directedness is Directedness.DIRECTED:
        return directed
    return True


def split_by_directedness(features: list[str], directed: bool, registry: dict[str, FeatureSpec]) -> list[str]:
    """Filter a requested feature list down to those applicable to this graph's directedness."""
    return [f for f in features if applicable(registry[f], directed)]
