"""Plain-NetworkX feature computation.

Every function here calls a standard `nx.*`/`nx.community.*` API and nothing
else. This is deliberate: when `nx-cugraph` is installed and configured (via
`nx.config.backend_priority` or a `backend=` kwarg), NetworkX transparently
dispatches supported calls to GPU and falls back to its own CPU implementation
otherwise -- so this module does not need any GPU-specific branching itself.

Exception: `leiden` falls back to `cugraph.leiden()` directly, or to `leidenalg` on the CPU, when
NetworkX cannot dispatch it (see `leiden()` below).

Each function takes the graph (already coerced to the correct directedness by
the caller) and returns `dict[node] -> value`, where `value` is a scalar for
single-column features or a fixed-order tuple for multi-column ones.
"""

import logging
from collections import Counter

import networkx as nx

logger = logging.getLogger(__name__)

_gpu_backend_active = False  # tracks the last configure_gpu_backend() outcome, for logging only


def gpu_backend_active() -> bool:
    """Whether the last configure_gpu_backend() call left GPU dispatch active."""
    return _gpu_backend_active


def configure_gpu_backend(enabled: bool) -> bool:
    """Turn nx-cugraph dispatch on/off for every subsequent plain `nx.*` call in
    this process. Call this once per `compute_node_features`/`compute_edge_features`
    invocation (api.py does this) rather than per-feature -- it's process-global
    state (`nx.config`), not something to toggle inside a single function.

    Returns whether GPU dispatch actually ended up active (False if `enabled` was
    True but nx-cugraph isn't installed) -- callers should log this so it's always
    explicit which path ran, not silently inferred.

    `fallback_to_nx=True` is set whenever GPU is enabled: nx.config's own default
    is False, which makes NetworkX *raise* instead of falling back to CPU for any
    algorithm nx-cugraph doesn't implement (e.g. eccentricity, square_clustering) --
    without this, requesting gpu=True would break every non-GPU-covered feature.
    """
    global _gpu_backend_active
    if not enabled:
        nx.config.backend_priority.algos = []
        _gpu_backend_active = False
        logger.info("GPU backend disabled: nx.* calls run on CPU.")
        return False
    try:
        import nx_cugraph  # noqa: F401
    except ImportError:
        nx.config.backend_priority.algos = []
        _gpu_backend_active = False
        logger.warning("gpu=True requested but nx-cugraph is not installed -- falling back to CPU. "
                        "Install the 'gpu' extra to enable GPU dispatch.")
        return False
    nx.config.backend_priority.algos = ["cugraph"]
    nx.config.fallback_to_nx = True
    # graphfx always hands each backend a fresh canonical graph and never mutates
    # it in place mid-pipeline, so the cross-call GPU-conversion cache nx-cugraph
    # warns about is safe here -- silence the warning rather than have it fire
    # on every single feature call.
    nx.config.warnings_to_ignore.add("cache")
    _gpu_backend_active = True
    logger.info("GPU backend enabled: nx.* calls dispatch to nx-cugraph where supported, "
                "CPU fallback otherwise.")
    return True


def _edges_to_cudf(G: nx.Graph):
    """Build a cudf DataFrame with 'src'/'dst' columns from G's edges, for direct
    cugraph.* calls that need a cugraph.Graph rather than going through
    NetworkX's dispatch layer."""
    import cudf

    src, dst = zip(*G.edges()) if G.number_of_edges() else ((), ())
    return cudf.DataFrame({"src": list(src), "dst": list(dst)})


def degree(G: nx.Graph) -> dict:
    return dict(G.degree())


def in_degree(G: nx.Graph) -> dict:
    return dict(G.in_degree())


def out_degree(G: nx.Graph) -> dict:
    return dict(G.out_degree())


def average_neighbor_degree(G: nx.Graph) -> dict:
    return nx.average_neighbor_degree(G)


def louvain(G: nx.Graph, *, seed: int = 42) -> dict:
    """Community SIZE per vertex (not the community id). Undirected only: a directed input is
    reduced to its undirected view."""
    UG = G.to_undirected() if G.is_directed() else G
    partition = nx.community.louvain_communities(UG, seed=seed)
    sizes = {}
    for community in partition:
        n = len(community)
        for node in community:
            sizes[node] = n
    return sizes


def leiden(G: nx.Graph, *, seed: int = 42) -> dict:
    """Community SIZE per vertex via Leiden, always on the undirected view of the graph:
    cugraph.leiden() (26.8) raises on directed input despite its documentation, and
    nx.community.leiden_communities rejects directed graphs outright.

    Backend order: nx dispatch (accelerates via nx-cugraph if configured) ->
    raw `cugraph.leiden()` if nx-cugraph isn't configured but cugraph is
    installed -> `leidenalg`+`python-igraph` CPU fallback.
    """
    UG = G.to_undirected() if G.is_directed() else G
    partition = None
    try:
        partition = nx.community.leiden_communities(UG, seed=seed)
        logger.info("leiden: ran via nx dispatch (%s).",
                     "GPU/nx-cugraph" if _gpu_backend_active else "CPU")
    except NotImplementedError:
        try:
            import cugraph

            cu_G = cugraph.Graph(directed=False)
            edges_df = _edges_to_cudf(UG)
            cu_G.from_cudf_edgelist(edges_df, source="src", destination="dst")
            parts_df, _modularity = cugraph.leiden(cu_G, random_state=seed)
            grouped = {}
            for row in parts_df.to_pandas().itertuples():
                grouped.setdefault(row.partition, set()).add(row.vertex)
            partition = list(grouped.values())
            logger.info("leiden: ran via direct cugraph.leiden() call (GPU).")
        except ImportError:
            try:
                import igraph as ig
                import leidenalg as la
            except ImportError as exc:
                raise RuntimeError(
                    "leiden: no GPU path (nx-cugraph/cugraph not installed) and "
                    "no CPU fallback (leidenalg/python-igraph not installed). "
                    "Install the 'gpu' extra or the 'leiden-cpu' extra."
                ) from exc
            logger.info("leiden: ran via leidenalg (CPU fallback, no GPU library installed).")
            nodes = sorted(UG.nodes())
            node_to_idx = {n: i for i, n in enumerate(nodes)}
            ig_graph = ig.Graph(
                n=len(nodes), edges=[(node_to_idx[u], node_to_idx[v]) for u, v in UG.edges()]
            )
            ig_partition = la.find_partition(ig_graph, la.ModularityVertexPartition, seed=seed)
            partition = [{nodes[i] for i in community} for community in ig_partition]
    sizes = {}
    for community in partition:
        n = len(community)
        for node in community:
            sizes[node] = n
    return sizes


def k_core(G: nx.Graph) -> dict:
    """K-core of the undirected view of the graph, as in graph-measures (its C++ code assumes an
    undirected graph and gives an incoherent peeling for directed input)."""
    return nx.core_number(G.to_undirected() if G.is_directed() else G)


def page_rank(G: nx.Graph, *, alpha: float = 0.9) -> dict:
    """PageRank with alpha=0.9, graph-measures' default, instead of networkx's 0.85. Dangling-vertex
    handling differs slightly between the implementations, so values are close to, not equal to,
    graph-measures' output."""
    return nx.pagerank(G, alpha=alpha)


def eigenvector_centrality(G: nx.Graph, *, max_iter: int = 1000) -> dict:
    return nx.eigenvector_centrality(G, max_iter=max_iter)


def katz_centrality(G: nx.Graph, *, max_iter: int = 1000) -> dict:
    """Katz centrality's power-iteration converges only for alpha < 1/lambda_max
    (lambda_max = the adjacency matrix's spectral radius). The library default
    (alpha=0.1) is just a guess and fails to converge on graphs with a larger
    spectral radius than that implies -- confirmed on our own reference graphs.
    Compute a safe alpha directly from the graph instead of guessing/raising.
    """
    import numpy as np
    import scipy.sparse.linalg as sla

    A = nx.adjacency_matrix(G).astype(float)
    try:
        # Fixed start vector: ARPACK's default is random, which makes repeated runs differ in the last bits.
        lambda_max = abs(sla.eigs(A, k=1, which="LM", return_eigenvectors=False, v0=np.ones(A.shape[0]))[0])
    except sla.ArpackNoConvergence as exc:
        lambda_max = abs(exc.eigenvalues[0]) if len(exc.eigenvalues) else None
    if not lambda_max or lambda_max == 0:
        lambda_max = max(dict(G.degree()).values(), default=1)  # crude fallback bound
    alpha = 0.9 / lambda_max
    return nx.katz_centrality(G, alpha=alpha, max_iter=max_iter)


def closeness_centrality(G: nx.Graph) -> dict:
    return nx.closeness_centrality(G)


def betweenness_centrality(G: nx.Graph, *, k: int | None = None, seed: int = 42) -> dict:
    """Exact when k=None; approximate (sampled) when k is set."""
    return nx.betweenness_centrality(G, k=k, seed=seed if k is not None else None)


def clustering_coefficient(G: nx.Graph) -> dict:
    """Forces the undirected view regardless of input directedness. Measured
    across 60 real graphs: directed input via nx.clustering's own directed
    (Fagiolo) formula has log-log slope 1.71 (up to 7.3s); the undirected
    formula measured slope 0.48 (<0.1s, trivially cheap) -- same pattern as
    triangle_count/generalized_degree/fiedler_vector/louvain/leiden, which
    already force undirected for the same reason. Directed callers get the
    undirected clustering value, not the Fagiolo directed-triad one.
    """
    return nx.clustering(G.to_undirected() if G.is_directed() else G)


def square_clustering_coefficient(G: nx.Graph) -> dict:
    return nx.square_clustering(G)


def triangle_count(G: nx.Graph) -> dict:
    """Undirected only -- nx.triangles raises NetworkXNotImplemented for directed graphs."""
    return nx.triangles(G.to_undirected() if G.is_directed() else G)


def weakly_connected_components(G: nx.Graph) -> dict:
    """Component SIZE per node (directed graphs only, by definition)."""
    sizes = {}
    for component in nx.weakly_connected_components(G):
        n = len(component)
        for node in component:
            sizes[node] = n
    return sizes


def generalized_degree(G: nx.Graph) -> dict:
    """Undirected only. Returns dict[node] -> fixed-width tuple, index j = number of
    edges incident to that node participating in exactly j triangles (Orsini et al.,
    Nat. Commun. 2015). Reimplemented directly (graph-measures wraps the same
    nx call but had a reshape bug noted upstream) -- we do the reshape ourselves.
    """
    UG = G.to_undirected() if G.is_directed() else G
    raw: dict[int, Counter] = nx.algorithms.cluster.generalized_degree(UG)
    max_j = max((k for counter in raw.values() for k in counter), default=-1) + 1
    out = {}
    for node, counter in raw.items():
        row = [0] * max_j
        for j, count in counter.items():
            row[j] = count
        out[node] = tuple(row)
    return out


def fiedler_vector(G: nx.Graph) -> dict:
    """Undirected only, requires a connected graph -- caller should size-gate/guard
    disconnected graphs (nx.fiedler_vector raises on those)."""
    UG = G.to_undirected() if G.is_directed() else G
    values = nx.fiedler_vector(UG)
    return dict(zip(sorted(UG.nodes()), values))


def eccentricity(G: nx.Graph) -> dict:
    """Max shortest-path distance from each node to every node REACHABLE from
    it (not `nx.eccentricity`, which raises on disconnected/non-strongly-
    connected graphs -- matches graph-measures' own EccentricityCalculator,
    which switched to this exact reachable-set-only definition for the same
    reason, per its own comment). Directed and undirected both supported --
    for directed graphs this is out-eccentricity (distances via outgoing
    edges only), consistent with how `dists`/`all_pairs_shortest_path_length`
    behaves on a DiGraph.
    """
    return {node: max(lengths.values()) if lengths else 0
            for node, lengths in nx.all_pairs_shortest_path_length(G)}


def load_centrality(G: nx.Graph) -> dict:
    return nx.load_centrality(G)


def flow(G: nx.Graph) -> dict:
    """Bespoke metric (graph-measures' FlowCalculator, no external
    equivalent) -- directed only. For each node n with b(n) = |ancestors(n)
    union descendants(n)| > 0:

        flow(n) = (1 / b(n)) * sum_{m reachable from n, dist_dir(n,m) > 0}
                      dist_undir(n, m) / dist_dir(n, m)

    i.e. the average ratio of "shortest path ignoring edge direction" to
    "shortest path respecting edge direction" over every node n can reach --
    values close to 1 mean n's directed reachability is about as efficient
    as the underlying undirected structure allows; values >> 1 mean directed
    edges force long detours relative to the undirected shortcut. Isolated
    nodes (b(n) == 0) get 0.

    Reimplemented with BFS (`single_source_shortest_path_length`) rather than
    graph-measures' `all_pairs_dijkstra_path_length(weight='weight')` --
    equivalent on unweighted graphs (the graphs here carry no edge weights) and cheaper (BFS is
    O(V+E) per source, Dijkstra O(E log V)). Edge weights are ignored.
    """
    UG = G.to_undirected()
    out = {}
    for node in G.nodes():
        reachable = set(nx.ancestors(G, node)) | set(nx.descendants(G, node))
        b = len(reachable)
        if b == 0:
            out[node] = 0.0
            continue
        dir_dist = nx.single_source_shortest_path_length(G, node)
        undir_dist = nx.single_source_shortest_path_length(UG, node)
        total = 0.0
        for m in reachable:
            d_dir = dir_dist.get(m, 0)
            if d_dir == 0:
                continue
            total += undir_dist.get(m, 0) / d_dir
        out[node] = total / b
    return out


def attractor_basin(G: nx.Graph, *, alpha: float = 2) -> dict:
    """Bespoke metric (graph-measures' AttractorBasinCalculator, no external
    equivalent) -- directed only. For node n, let out_dist[n]/in_dist[n] be
    the histograms of shortest-path distances to n's descendants / from n's
    ancestors, and avg_out[d]/avg_in[d] be those histograms' per-distance
    averages across ALL nodes (a global normalization, not per-node). Then:

        attractor_basin(n) = numerator(n) / denominator(n)
        numerator(n)   = sum_d (in_dist[n][d]  / avg_in[d])  * alpha^(-d)
        denominator(n) = sum_d (out_dist[n][d] / avg_out[d]) * alpha^(-d)

    Intuition: weights each reachable node by how much closer it is than
    "typical" (relative to avg_in/avg_out) and by an exponentially decaying
    alpha^(-d) in distance, then takes the in/out ratio -- high values mean
    n is an "attractor" (pulled toward disproportionately more than it
    pushes out), matching the name. NaN when denominator is 0 (e.g. n has no
    descendants), same as graph-measures' own default.

    Reimplemented with two BFS passes (forward + reverse-graph) per node
    instead of graph-measures' `all_pairs_dijkstra_path_length` PLUS separate
    `nx.ancestors`/`nx.descendants` calls -- the BFS distance dict already
    tells you who's reachable, so this avoids computing reachability twice.
    Still fundamentally O(V*(V+E)): expect the same cost class as `flow`.
    """
    from collections import Counter

    n_nodes = G.number_of_nodes()
    RG = G.reverse(copy=False)
    out_dist: dict = {}
    in_dist: dict = {}
    for node in G.nodes():
        fwd = nx.single_source_shortest_path_length(G, node)
        fwd.pop(node, None)
        out_dist[node] = Counter(fwd.values())
        bwd = nx.single_source_shortest_path_length(RG, node)
        bwd.pop(node, None)
        in_dist[node] = Counter(bwd.values())

    def avg_per_dist(count_dist):
        totals: dict = {}
        for counter in count_dist.values():
            for dist, occurrences in counter.items():
                totals[dist] = totals.get(dist, 0) + occurrences
        return {dist: count / n_nodes for dist, count in totals.items()}

    avg_out = avg_per_dist(out_dist)
    avg_in = avg_per_dist(in_dist)

    out = {}
    for node in G.nodes():
        denominator = sum((count / avg_out[d]) * (alpha ** (-d)) for d, count in out_dist[node].items())
        if denominator == 0:
            out[node] = float("nan")
            continue
        numerator = sum((count / avg_in[d]) * (alpha ** (-d)) for d, count in in_dist[node].items())
        out[node] = numerator / denominator
    return out


def communicability_betweenness_centrality(G: nx.Graph) -> dict:
    """Direct `nx.communicability_betweenness_centrality` wrapper (Estrada/
    Higham/Hatano) -- undirected only (like graph-measures' own wrapper).
    O(n^4): computes a dense matrix exponential of the (n-1)x(n-1) adjacency
    submatrix once per vertex. There is no GPU implementation; it is unusable on large graphs.
    """
    UG = G.to_undirected() if G.is_directed() else G
    return nx.communicability_betweenness_centrality(UG)


def bfs_moments(G: nx.Graph) -> dict:
    """Per node: (mean, std) of unweighted BFS-distance-plus-one to every reachable
    node. Reimplemented from the mathematical definition (not copied from
    graph-measures) -- cheap, single BFS per node, no Dijkstra/weights involved.
    """
    import numpy as np

    out = {}
    for node in G.nodes():
        dist_counts = Counter(nx.single_source_shortest_path_length(G, node).values())
        dists = np.array([d + 1 for d in dist_counts.keys()], dtype=float)
        weights = np.array(list(dist_counts.values()), dtype=float)
        mean = float(np.average(dists, weights=weights))
        variance = float(np.average((dists - mean) ** 2, weights=weights))
        out[node] = (mean, variance**0.5)
    return out
