"""Minimum spanning arborescence (directed MST) in O(E log V).

A faster drop-in for networkx.minimum_spanning_arborescence, which runs
Edmonds' algorithm without a priority queue (~O(V*E) in practice) and keeps
every contraction step in memory.

Algorithm: Tarjan's (1977) O(E log V) formulation of Chu-Liu/Edmonds, with
mergeable skew heaps carrying lazy weight offsets and a union-find with
rollback, so contracted cycles can be expanded back into an explicit edge set
(the reconstruction issue noted by Camerini, Fratta & Maffioli, 1979). This is
a direct port of KACTL's DirectedMST (authors chilli, Takanori Maehara, Benq, Simon Lindholm;
KTH Royal Institute of Technology, github.com/kth-competitive-programming/kactl, license CC0), written
iteratively (no recursion) with array-based nodes.

Weights are real numbers (no integer quantization). The root is free, as in
networkx: a virtual super-root gets an edge of weight M to every vertex, with
M larger than the total weight of any spanning arborescence, so the optimum
uses exactly one super-root edge, which marks the chosen root. If no spanning
arborescence exists, ValueError is raised (networkx raises an exception too).

Ties may be broken differently from networkx; the total weight of the
returned arborescence is optimal either way.
"""
import math

import numpy as np


def minimum_spanning_arborescence(nodes, src, dst, weight):
    """Minimum-weight spanning arborescence of a directed graph.

    Parameters
    ----------
    nodes : sequence of hashable vertex ids
    src, dst : sequences of vertex ids, one entry per directed edge src -> dst
    weight : sequence of finite float edge weights (any sign)

    Returns
    -------
    edges : list of (u, v, w) of the arborescence, one per non-root vertex
    root  : the root vertex
    """
    nodes = list(nodes)
    n = len(nodes)
    if n == 0:
        raise ValueError("empty graph")
    index = {v: i for i, v in enumerate(nodes)}
    a = [index[u] for u in src]
    b = [index[v] for v in dst]
    w = [float(x) for x in weight]
    # M below is built from the weights; inf/NaN would make it inf/NaN.
    if not all(math.isfinite(x) for x in w):
        raise ValueError("edge weights must be finite (no inf or NaN)")
    if n == 1:
        return [], nodes[0]

    # Super-root r = n, with an edge of weight M to every vertex.
    span = sum(abs(x) for x in w)
    M = 2.0 * span + 1.0
    r = n
    N = n + 1
    a += [r] * n
    b += list(range(n))
    w += [M] * n
    m = len(w)

    # ---- skew heap over edge ids (arrays: key weight, children, lazy delta) ----
    W = list(w)
    L = [-1] * m
    R = [-1] * m
    D = [0.0] * m

    def prop(x):
        d = D[x]
        if d:
            W[x] += d
            if L[x] >= 0:
                D[L[x]] += d
            if R[x] >= 0:
                D[R[x]] += d
            D[x] = 0.0

    def merge(x, y):
        if x < 0:
            return y
        if y < 0:
            return x
        spine = []
        while x >= 0 and y >= 0:
            prop(x)
            prop(y)
            if W[x] > W[y]:
                x, y = y, x
            spine.append(x)
            x = R[x]
        rest = x if x >= 0 else y
        for node in reversed(spine):
            R[node] = rest
            L[node], R[node] = R[node], L[node]
            rest = node
        return rest

    heap = [-1] * N
    for e in range(m):
        heap[b[e]] = merge(heap[b[e]], e)

    # ---- union-find with rollback (union by size, no path compression) ----
    par = [-1] * N
    history = []

    def find(x):
        while par[x] >= 0:
            x = par[x]
        return x

    def join(x, y):
        x, y = find(x), find(y)
        if x == y:
            return False
        if par[x] > par[y]:
            x, y = y, x
        history.append((x, par[x]))
        history.append((y, par[y]))
        par[x] += par[y]
        par[y] = x
        return True

    def rollback(t):
        while len(history) > t:
            i, v = history.pop()
            par[i] = v

    # ---- main loop (KACTL dmst) ----
    seen = [-1] * N
    seen[r] = r
    path = [0] * N
    Q = [-1] * N           # edge ids chosen along the current path
    in_edge = [-1] * N     # chosen incoming edge per vertex / component
    cycles = []            # (component, uf time, edge ids of the cycle)

    for s in range(N):
        u = s
        qi = 0
        while seen[u] < 0:
            if heap[u] < 0:
                raise ValueError("no spanning arborescence exists")
            top = heap[u]
            prop(top)
            ew = W[top]
            D[top] -= ew
            prop(top)
            heap[u] = merge(L[top], R[top])
            Q[qi] = top
            path[qi] = u
            qi += 1
            seen[u] = s
            u = find(a[top])
            if seen[u] == s:  # cycle: contract
                cyc = -1
                end = qi
                t = len(history)
                while True:
                    qi -= 1
                    x = path[qi]
                    cyc = merge(cyc, heap[x])
                    if not join(u, x):
                        break
                u = find(u)
                heap[u] = cyc
                seen[u] = -1
                cycles.append((u, t, Q[qi:end]))
        for i in range(qi):
            in_edge[find(b[Q[i]])] = Q[i]

    # ---- expand contracted cycles, most recent first ----
    for u, t, comp in reversed(cycles):
        rollback(t)
        keep = in_edge[u]
        for e in comp:
            in_edge[find(b[e])] = e
        in_edge[find(b[keep])] = keep

    roots = [v for v in range(n) if a[in_edge[v]] == r]
    if len(roots) != 1:
        raise ValueError("no spanning arborescence exists")
    root = roots[0]
    edges = [(nodes[a[in_edge[v]]], nodes[v], w[in_edge[v]]) for v in range(n) if v != root]
    return edges, nodes[root]


def minimum_spanning_arborescence_nx(D, attr="weight"):
    """Same as networkx.minimum_spanning_arborescence(D, attr) for a weighted
    nx.DiGraph: returns an nx.DiGraph holding the arborescence."""
    import networkx as nx

    nodes = list(D.nodes())
    src, dst, wt = [], [], []
    for u, v, data in D.edges(data=True):
        src.append(u)
        dst.append(v)
        wt.append(data.get(attr, 1.0))
    edges, _ = minimum_spanning_arborescence(nodes, src, dst, np.asarray(wt, dtype=float))
    A = nx.DiGraph()
    A.add_nodes_from(nodes)
    A.add_weighted_edges_from(edges, weight=attr)
    return A
