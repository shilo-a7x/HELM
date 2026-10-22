"""Node-label canonicalization.

Several backends (most notably the motif_client subprocess boundary to
graph-measures, whose CSR wire format explicitly assumes node labels are
exactly the contiguous integers [0, ..., n-1] -- see graph_converter.py's own
docstring) silently misbehave or produce wrong results on arbitrary node
labels (strings, non-contiguous ints, tuples, ...).

To make this a non-issue everywhere rather than something every backend has
to remember to handle, `graphfx.api` canonicalizes on entry and maps results
back to the caller's original labels on exit -- callers can pass a graph with
any hashable node labels and never need to think about this.
"""

import logging

import networkx as nx

logger = logging.getLogger(__name__)


def is_canonical(G: nx.Graph) -> bool:
    """True iff G's nodes are already exactly the integers [0, ..., n-1], in
    that order matters not -- only the *set* has to match, since downstream
    consumers index by label value, not iteration order."""
    nodes = list(G.nodes())
    return all(isinstance(n, int) and not isinstance(n, bool) for n in nodes) and set(nodes) == set(
        range(len(nodes))
    )


def canonicalize(G: nx.Graph) -> tuple[nx.Graph, dict]:
    """Return (canonical_graph, label_map) where canonical_graph's nodes are
    exactly [0, ..., n-1] and label_map maps canonical id -> original label.

    If G is already canonical, returns G unchanged (no copy) and an identity
    map -- this is the common case for our current pipeline and should stay
    cheap. Logs a warning when relabeling actually happens, since it's a sign
    the input graph wasn't built the way this pipeline expects, even though
    it's handled correctly either way.
    """
    if is_canonical(G):
        nodes = list(G.nodes())
        return G, {n: n for n in nodes}

    logger.warning(
        "Input graph nodes are not a contiguous [0, n-1] integer range "
        "(found %d nodes with non-canonical labels, e.g. %r) -- relabeling "
        "internally for backend compatibility; results are mapped back to "
        "your original node labels automatically.",
        G.number_of_nodes(),
        next(iter(G.nodes()), None),
    )
    original_nodes = list(G.nodes())
    to_canonical = {n: i for i, n in enumerate(original_nodes)}
    label_map = {i: n for n, i in to_canonical.items()}
    canonical_graph = nx.relabel_nodes(G, to_canonical, copy=True)
    return canonical_graph, label_map
