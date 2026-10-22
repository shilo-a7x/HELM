"""The simple wrapper: nx.Graph in, feature DataFrame out."""

import logging

import networkx as nx
import pandas as pd

from graphfx import hierarchy, motif_client, nx_backend
from graphfx.graph_utils import canonicalize
from graphfx.features import (
    DEFAULT_EDGE_FEATURES,
    DEFAULT_NODE_FEATURES,
    EDGE_FEATURES,
    NODE_FEATURES,
    Backend,
    split_by_directedness,
)

logger = logging.getLogger(__name__)

_NODE_FUNCS = {
    "degree": nx_backend.degree,
    "in_degree": nx_backend.in_degree,
    "out_degree": nx_backend.out_degree,
    "average_neighbor_degree": nx_backend.average_neighbor_degree,
    "louvain": nx_backend.louvain,
    "leiden": nx_backend.leiden,
    "k_core": nx_backend.k_core,
    "page_rank": nx_backend.page_rank,
    "eigenvector_centrality": nx_backend.eigenvector_centrality,
    "katz_centrality": nx_backend.katz_centrality,
    "closeness_centrality": nx_backend.closeness_centrality,
    "betweenness_centrality": nx_backend.betweenness_centrality,
    "clustering_coefficient": nx_backend.clustering_coefficient,
    "square_clustering_coefficient": nx_backend.square_clustering_coefficient,
    "triangle_count": nx_backend.triangle_count,
    "weakly_connected_components": nx_backend.weakly_connected_components,
    "generalized_degree": nx_backend.generalized_degree,
    "fiedler_vector": nx_backend.fiedler_vector,
    "bfs_moments": nx_backend.bfs_moments,
    "eccentricity": nx_backend.eccentricity,
    "load_centrality": nx_backend.load_centrality,
    "flow": nx_backend.flow,
    "attractor_basin": nx_backend.attractor_basin,
    "communicability_betweenness_centrality": nx_backend.communicability_betweenness_centrality,
}

_HIERARCHY_FUNCS = {
    "hierarchy_energy": hierarchy.hierarchy_energy,
    "trophic_incoherence": hierarchy.trophic_incoherence,
}

# Multi-column features: name -> column suffixes (order matches the tuple each
# backend function returns).
_MULTI_COLUMN = {
    "bfs_moments": ("mean", "std"),
}


def _expand(name: str, values: dict) -> dict[str, dict]:
    """Turn a dict[node] -> scalar-or-tuple into {column_name: dict[node]->scalar}."""
    sample = next(iter(values.values()), None)
    if isinstance(sample, tuple):
        suffixes = _MULTI_COLUMN.get(name)
        width = len(sample)
        if suffixes is None or len(suffixes) != width:
            suffixes = [str(i) for i in range(width)]
        columns = {f"{name}_{suffix}": {} for suffix in suffixes}
        for node, row in values.items():
            for suffix, v in zip(suffixes, row):
                columns[f"{name}_{suffix}"][node] = v
        return columns
    return {name: values}


def _expand_motif(level: int, values: dict) -> dict[str, dict]:
    """Column naming matches graph-measures' own established convention
    (motifs_node_{level}_{k}) exactly, not the generic f"{name}_{suffix}"
    pattern -- this is what the reference output CSVs already use."""
    width = len(next(iter(values.values()), ()))
    columns = {f"motifs_node_{level}_{k}": {} for k in range(1, width + 1)}
    for node, row in values.items():
        for k, v in enumerate(row, start=1):
            columns[f"motifs_node_{level}_{k}"][node] = v
    return columns


def compute_node_features(
    graph: nx.Graph,
    features: list[str] | None = None,
    *,
    seed: int = 42,
    betweenness_k: int | None = None,
    gpu: bool = True,
    device: int = 0,
) -> pd.DataFrame:
    """Compute node features for `graph`, returning a DataFrame with an explicit
    `node` column holding the caller's ORIGINAL node labels (not a derived index) --
    this survives `to_csv()` regardless of `index=True/False`, and stays correct
    for graphs with string/tuple/non-contiguous-integer node ids.

    Internally, `graph` is canonicalized to a contiguous [0, n-1] integer label
    space (see `graphfx.graph_utils.canonicalize`) before any backend runs --
    several backends (esp. the motif subprocess boundary) require this and would
    silently misbehave otherwise. Results are mapped back to the original labels
    before returning, so this is transparent to callers.

    `features=None` computes the default (non-opt-in) feature set applicable to
    this graph's directedness. Features backed by `motif_client`/`hierarchy`
    that aren't wired up yet raise a clear NotImplementedError when requested.
    """
    canonical_graph, label_map = canonicalize(graph)
    directed = canonical_graph.is_directed()
    requested = features if features is not None else DEFAULT_NODE_FEATURES
    applicable = split_by_directedness(requested, directed, NODE_FEATURES)
    nx_backend.configure_gpu_backend(gpu)
    if any(NODE_FEATURES[name].backend is Backend.MOTIF for name in applicable):
        logger.info("motif3/motif4 requested: dispatching to the graph-measures subprocess "
                     "with gpu=%s (device=%s).", gpu, device)

    columns: dict[str, dict] = {}
    for name in applicable:
        spec = NODE_FEATURES[name]
        if spec.backend is Backend.NETWORKX:
            func = _NODE_FUNCS.get(name)
            if func is None:
                raise NotImplementedError(
                    f"{name}: no implementation is registered for this feature "
                    "(see graphfx.features.NODE_FEATURES)"
                )
            kwargs = {"k": betweenness_k, "seed": seed} if name == "betweenness_centrality" else {}
            kwargs = {"seed": seed} if name in ("louvain", "leiden") else kwargs
            values = func(canonical_graph, **kwargs)
        elif spec.backend is Backend.HIERARCHY:
            values = _HIERARCHY_FUNCS[name](canonical_graph)
        elif spec.backend is Backend.MOTIF:
            level = 3 if name == "motif3" else 4
            values = motif_client.compute_motif(canonical_graph, level, gpu=gpu, device=device)
            columns.update(_expand_motif(level, values))
            continue
        else:
            raise AssertionError(f"unhandled backend {spec.backend}")
        columns.update(_expand(name, values))

    df = pd.DataFrame(columns)
    df.index.name = "_canonical_node"
    df = df.sort_index().reset_index()
    df["node"] = df["_canonical_node"].map(label_map)
    df = df.drop(columns=["_canonical_node"])
    return df[["node"] + [c for c in df.columns if c != "node"]]


def compute_edge_features(
    graph: nx.Graph,
    features: list[str] | None = None,
    *,
    gpu: bool = True,
) -> pd.DataFrame:
    """Returns a DataFrame with explicit `source`/`target` columns holding the
    caller's ORIGINAL node labels (see `compute_node_features` for why)."""
    canonical_graph, label_map = canonicalize(graph)
    directed = canonical_graph.is_directed()
    requested = features if features is not None else DEFAULT_EDGE_FEATURES
    applicable = split_by_directedness(requested, directed, EDGE_FEATURES)
    nx_backend.configure_gpu_backend(gpu)

    per_edge: dict[tuple, dict] = {}
    for name in applicable:
        spec = EDGE_FEATURES[name]
        if spec.backend is Backend.NETWORKX:
            values = _EDGE_FUNCS[name](canonical_graph)
        elif spec.backend is Backend.MOTIF:
            raise NotImplementedError(f"{name}: motif subprocess backend not wired up yet")
        else:
            raise AssertionError(f"unhandled backend {spec.backend}")
        for edge, v in values.items():
            per_edge.setdefault(edge, {})[name] = v

    records: list[dict] = []
    for (u, v), vals in per_edge.items():
        records.append({"source": label_map[u], "target": label_map[v], **vals})
    return pd.DataFrame(records)


def _edge_betweenness(G: nx.Graph) -> dict:
    return nx.edge_betweenness_centrality(G)


def _edge_current_flow_betweenness(G: nx.Graph) -> dict:
    # Keyed back onto G's own (possibly directed) edge tuples -- nx's result is
    # keyed by UG's edges, which for directed input won't align with the other
    # edge features' directed keys in compute_edge_features' per-edge merge.
    UG = G.to_undirected() if G.is_directed() else G
    scores = nx.edge_current_flow_betweenness_centrality(UG)
    return {(u, v): scores.get((u, v), scores.get((v, u))) for u, v in G.edges()}


def _min_edge_cut(G: nx.Graph) -> dict:
    return {(u, v): nx.edge_connectivity(G, u, v) for u, v in G.edges()}


def _jaccard_coefficient(G: nx.Graph) -> dict:
    # ebunch=G.edges() restricts the score to existing edges (O(E)); the default scores every
    # non-edge pair (O(n^2)), which is link prediction, not an edge feature.
    # Undirected only: a directed input is reduced to its undirected view. The (u, v) keys below
    # are the original graph's edges, so they merge with the other edge features.
    UG = G.to_undirected() if G.is_directed() else G
    scores = {(u, v): p for u, v, p in nx.jaccard_coefficient(UG, ebunch=UG.edges())}
    return {(u, v): scores.get((u, v), scores.get((v, u))) for u, v in G.edges()}


def _adamic_adar_index(G: nx.Graph) -> dict:
    UG = G.to_undirected() if G.is_directed() else G
    scores = {(u, v): p for u, v, p in nx.adamic_adar_index(UG, ebunch=UG.edges())}
    return {(u, v): scores.get((u, v), scores.get((v, u))) for u, v in G.edges()}


def _preferential_attachment(G: nx.Graph) -> dict:
    UG = G.to_undirected() if G.is_directed() else G
    scores = {(u, v): p for u, v, p in nx.preferential_attachment(UG, ebunch=UG.edges())}
    return {(u, v): scores.get((u, v), scores.get((v, u))) for u, v in G.edges()}


def _resource_allocation_index(G: nx.Graph) -> dict:
    UG = G.to_undirected() if G.is_directed() else G
    scores = {(u, v): p for u, v, p in nx.resource_allocation_index(UG, ebunch=UG.edges())}
    return {(u, v): scores.get((u, v), scores.get((v, u))) for u, v in G.edges()}


_EDGE_FUNCS = {
    "edge_betweenness": _edge_betweenness,
    "edge_current_flow_betweenness": _edge_current_flow_betweenness,
    "min_edge_cut": _min_edge_cut,
    "jaccard_coefficient": _jaccard_coefficient,
    "adamic_adar_index": _adamic_adar_index,
    "preferential_attachment": _preferential_attachment,
    "resource_allocation_index": _resource_allocation_index,
}
