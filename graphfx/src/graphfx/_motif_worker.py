"""Runs inside the Python 3.7 motif environment (built by scripts/setup.sh), not in graphfx's own
environment: motif_client starts it as a standalone script. It uses only the standard library,
networkx and graphMeasures, since nothing else is guaranteed to exist in that old environment.

Wire format (JSON rather than a pickle, which would not load across the two networkx versions):

Input file: {"indices": [...], "neighbors": [...], "directed": bool,
             "level": 3|4, "gpu": bool, "device": int}
  -- indices/neighbors are the CSR representation of an ALREADY-CANONICAL
  graph (nodes exactly [0, n-1]; see graphfx.graph_utils.canonicalize on the
  caller side). indices has length n+1 (standard CSR offsets); neighbors[
  indices[i]:indices[i+1]] are node i's neighbors, sorted ascending -- this
  matches graph-measures' own `graph_converter.py` format exactly, and is
  built directly rather than round-tripped through an nx.Graph pickle.

Output file (CSV): header `node,motifs_node_{level}_1,...,motifs_node_{level}_K`
  -- one row per vertex, in ascending vertex-id order.
"""

import csv
import json
import sys
import tempfile

import networkx as nx
from graphMeasures import FeatureCalculator


def main():
    input_path, output_path = sys.argv[1], sys.argv[2]
    with open(input_path) as f:
        spec = json.load(f)

    indices = spec["indices"]
    neighbors = spec["neighbors"]
    directed = spec["directed"]
    level = spec["level"]
    gpu = spec["gpu"]
    device = spec["device"]
    n = len(indices) - 1

    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(n))
    for node in range(n):
        for nbr in neighbors[indices[node] : indices[node + 1]]:
            G.add_edge(node, nbr)

    feature_name = f"motif{level}"
    with tempfile.TemporaryDirectory() as scratch_dir:
        fc = FeatureCalculator(
            G,
            [feature_name],
            dir_path=scratch_dir,
            acc=True,
            gpu=gpu,
            device=device,
            directed=directed,
            should_zscore=False,
        )
        fc.calculate_features(should_dump=False)
        df = fc.get_features()

    motif_cols = [c for c in df.columns if c.startswith(f"motifs_node_{level}_")]
    motif_cols.sort(key=lambda c: int(c.rsplit("_", 1)[1]))

    df = df.sort_index()
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node"] + motif_cols)
        # itertuples(index=True) yields (node, *motif_cols) directly off the
        # DataFrame's internal buffers -- avoids one .loc[node, c] scalar
        # lookup per (node, column) pair, which dominates runtime on large
        # graphs (e.g. up to ~2M lookups for a 10K-node directed motif4).
        writer.writerows(df[motif_cols].itertuples(index=True, name=None))


if __name__ == "__main__":
    main()
