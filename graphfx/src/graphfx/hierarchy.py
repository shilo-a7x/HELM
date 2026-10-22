"""Hierarchical-ranking features (SpringRank energy, trophic incoherence).

These replace graph-measures' `hierarchy_energy`, which is disabled upstream: its Laplacian is not
symmetric positive definite, so the conjugate-gradient solver it uses is not valid for it.
"""

import logging

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


def hierarchy_energy(G: nx.Graph) -> dict:
    """SpringRank (De Bacco, Larremore, Moore, Science Advances 2018) hierarchical
    ranking. Minimizes H(s) = 1/2 * sum_ij A_ij*(s_i - s_j - 1)^2 via a sparse
    linear solve on the Laplacian of A+A^T -- well-defined for any directed
    graph, including cyclic ones (unlike trophic_levels below).
    """
    from springrank import SpringRank

    nodes = sorted(G.nodes())
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, format="csr", dtype=float)
    model = SpringRank().fit(A)
    # springrank 0.0.10 documents the result as `ranks_` but the attribute is `ranks`.
    return dict(zip(nodes, model.ranks))


def trophic_incoherence(G: nx.Graph) -> dict:
    """NetworkX's trophic_levels/trophic_incoherence_parameter (Johnson et al.,
    PNAS 2014). Needs at least one "basal" vertex (in-degree 0), which graphs where every
    vertex has an incoming edge (such as the bundled ones) do not have; the feature is then
    NaN for all vertices and a warning is logged instead of raising.
    """
    try:
        levels = nx.trophic_levels(G)
    except nx.NetworkXError:
        logger.warning(
            "trophic_incoherence: graph has no basal nodes (in-degree 0) -- "
            "trophic levels are undefined for this graph; returning NaN for "
            "all vertices."
        )
        return {n: float("nan") for n in G.nodes()}
    diffs = nx.trophic_differences(G, levels=levels)
    per_node_diffs: dict[int, list[float]] = {n: [] for n in G.nodes()}
    for (u, v), d in diffs.items():
        per_node_diffs[v].append(d)
    return {n: float(np.std(vals)) if vals else float("nan") for n, vals in per_node_diffs.items()}
