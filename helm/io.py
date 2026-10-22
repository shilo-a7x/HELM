"""Graph input/output for HELM.

Graphs are stored either as pickled networkx graphs (``.pkl``) or as plain edge
lists (``.edgelist``). An edge list has one ``source target`` pair per line
(whitespace or comma separated, integer vertex IDs) and states whether the
graph is directed in a header line::

    # directed
    0 1
    0 2

``# undirected`` is the other option. If the header is missing, the caller
must say so with ``directed=``; the directedness is never guessed.

An optional ``# vertices 0 1 2 ...`` line fixes the vertex order (and lists
isolated vertices). Files written by ``save_graph`` always include it, so a
graph reloads with the same vertex and edge order it was saved with.

Pickles execute code when loaded. Only load pickles you trust.
"""
import json
import os
import pickle

import networkx as nx

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DIRECTED_HEADER = "# directed"
UNDIRECTED_HEADER = "# undirected"
VERTICES_HEADER = "# vertices"


# Output layout, relative to the run's output directory:
#   <output>/<collection>/<graph_id>/features/{node,edge}_features.csv
#   <output>/<collection>/<graph_id>/scores/edge_scores.csv
#   <output>/<collection>/model/
FEATURE_DIR = "features"
SCORE_DIR = "scores"
MODEL_DIR = "model"
NODE_FEATURES = "node_features.csv"
EDGE_FEATURES = "edge_features.csv"
EDGE_SCORES = "edge_scores.csv"


class GraphFormatError(ValueError):
    """The input graph file is malformed."""


def read_edgelist(path, directed=None):
    """Read an integer edge list into a ``nx.Graph`` or ``nx.DiGraph``.

    ``directed`` must be given when the file has no header line; if both are
    present they must agree.
    """
    header = None
    vertices = []
    edges = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                tag = " ".join(line.lower().split())
                if tag in (DIRECTED_HEADER, UNDIRECTED_HEADER):
                    header = tag == DIRECTED_HEADER
                elif tag.startswith(VERTICES_HEADER + " "):
                    try:
                        vertices = [int(x) for x in tag.split()[2:]]
                    except ValueError:
                        raise GraphFormatError(f"{path}:{lineno}: vertex IDs must be integers") from None
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                raise GraphFormatError(f"{path}:{lineno}: expected 'source target', got {line!r}")
            try:
                edges.append((int(parts[0]), int(parts[1])))
            except ValueError:
                raise GraphFormatError(
                    f"{path}:{lineno}: vertex IDs must be integers, got {parts[0]!r} {parts[1]!r}"
                ) from None
    if header is None and directed is None:
        raise GraphFormatError(
            f"{path}: no '# directed' / '# undirected' header line and no directed= "
            "argument; cannot tell whether the graph is directed"
        )
    if header is not None and directed is not None and header != directed:
        raise GraphFormatError(
            f"{path}: header says {'directed' if header else 'undirected'} but "
            f"directed={directed} was requested"
        )
    graph = nx.DiGraph() if (header if header is not None else directed) else nx.Graph()
    graph.add_nodes_from(vertices)
    graph.add_edges_from(edges)
    return graph


def write_edgelist(graph, path):
    """Write ``graph`` as an edge list with a directedness header line."""
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "w") as f:
        f.write((DIRECTED_HEADER if graph.is_directed() else UNDIRECTED_HEADER) + "\n")
        f.write(VERTICES_HEADER + " " + " ".join(str(n) for n in graph.nodes) + "\n")
        for u, v in graph.edges():
            f.write(f"{u} {v}\n")


def load_graph(path, directed=None):
    """Load a graph from a ``.pkl`` or ``.edgelist`` file."""
    if path.endswith(".edgelist"):
        return read_edgelist(path, directed=directed)
    with open(path, "rb") as f:
        return pickle.load(f)


def save_graph(graph, file_path):
    """Save a graph as ``.pkl`` or ``.edgelist``, creating parent directories."""
    if file_path.endswith(".edgelist"):
        write_edgelist(graph, file_path)
        return
    dirpath = os.path.dirname(file_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(graph, f)


def validate_graph(graph, name="graph"):
    """Check a graph can go through the pipeline; raise ``GraphFormatError`` if not."""
    if graph.number_of_edges() == 0:
        raise GraphFormatError(f"{name}: no edges")
    loops = [n for n in graph if graph.has_edge(n, n)]
    if loops:
        raise GraphFormatError(f"{name}: {len(loops)} self-loop(s), e.g. vertex {loops[0]}")
    ug = graph.to_undirected(as_view=True)
    if not nx.is_connected(ug):
        n = nx.number_connected_components(ug)
        raise GraphFormatError(f"{name}: graph is disconnected ({n} components)")


def _resolve(graph_path, manifest, key):
    """Absolute path of a graph file named in a manifest: relative paths are tried against the
    current directory first and then against the repository root (where ``data/`` lives)."""
    for candidate in (graph_path, os.path.join(REPO_ROOT, graph_path)):
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    raise GraphFormatError(f"{manifest}: {key} {graph_path!r} does not exist")


def load_manifest(path, output_dir, name=None, features_dir=None):
    """Read a manifest of ``{graph_id, collection, G_path[, T_path]}`` entries,
    fill in the per-graph feature and score paths under ``output_dir``, and
    write the result to ``<output_dir>/manifests/<name>.json``.

    The stages read and update manifest files as they run, so they always
    work on this per-run copy and never on the input manifest. Returns the
    path of the copy.

    With ``features_dir``, feature paths point at precomputed CSVs laid out as
    ``<features_dir>/<collection>/<graph_id>/{node,edge}_features.csv`` instead.
    """
    with open(path) as f:
        entries = json.load(f)
    if not isinstance(entries, list) or not entries:
        raise GraphFormatError(f"{path}: manifest must be a non-empty JSON list")
    filled = []
    for i, entry in enumerate(entries):
        for key in ("graph_id", "collection", "G_path"):
            if key not in entry:
                raise GraphFormatError(f"{path}: entry {i} is missing {key!r}")
        entry = dict(entry)
        for key in ("G_path", "T_path"):
            if key in entry:
                entry[key] = _resolve(entry[key], path, key)
        gdir = os.path.join(output_dir, entry["collection"], entry["graph_id"])
        fdir = (
            os.path.join(features_dir, entry["collection"], entry["graph_id"])
            if features_dir
            else os.path.join(gdir, FEATURE_DIR)
        )
        filled.append({
            **entry,
            "node_features_path": os.path.join(fdir, NODE_FEATURES),
            "edge_features_path": os.path.join(fdir, EDGE_FEATURES),
            "score_path": os.path.join(gdir, SCORE_DIR, EDGE_SCORES),
        })
    name = name or os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(output_dir, "manifests", name + ".json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(filled, f, indent=2)
    return out_path
