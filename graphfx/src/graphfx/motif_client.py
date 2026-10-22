"""Subprocess boundary to the upstream `graph-measures` package (GPL-3.0), used for the motif3 and
motif4 features, which no other library computes on directed graphs.

It is a subprocess rather than an import because graph-measures' accelerated build needs
Boost.Python and Python 3.7, which cannot share an interpreter with graphfx (Python 3.13).
`scripts/setup.sh` builds that environment.

Configuration (environment variables; the defaults are what `scripts/setup.sh` creates):
  GRAPHFX_MOTIF_ENV_PYTHON       python of the Python 3.7 conda environment [<repo>/.motif-env/bin/python]
  GRAPHFX_GRAPH_MEASURES_PATH    the graph-measures checkout, added to PYTHONPATH for the subprocess
                                 [<repo>/graph-measures]
  CUDA_VISIBLE_DEVICES           passed to the subprocess unchanged; select the GPU with it
"""

import csv
import json
import logging
import os
import subprocess
import tempfile

import networkx as nx

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# scripts/setup.sh builds both of these inside the repository directory.
_DEFAULT_MOTIF_ENV_PYTHON = os.path.join(_REPO_ROOT, ".motif-env", "bin", "python")
_DEFAULT_GRAPH_MEASURES_PATH = os.path.join(_REPO_ROOT, "graph-measures")
_WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_motif_worker.py")


def _to_csr(G: nx.Graph) -> tuple[list[int], list[int]]:
    """Build the CSR (indices, neighbors) representation graph-measures itself
    uses (graph_converter.py), directly from an already-canonical graph (nodes
    exactly [0, n-1] -- caller's responsibility, see graphfx.graph_utils)."""
    n = G.number_of_nodes()
    indices = [0]
    neighbors: list[int] = []
    for node in range(n):
        nbrs = sorted(G.neighbors(node))
        neighbors.extend(nbrs)
        indices.append(indices[-1] + len(nbrs))
    return indices, neighbors


def compute_motif(
    G: nx.Graph,
    level: int,
    *,
    gpu: bool = True,
    device: int = 0,
) -> dict[int, tuple]:
    """Compute motif3 (level=3) or motif4 (level=4) per-node counts for an
    ALREADY-CANONICAL graph (nodes exactly [0, n-1]; graphfx.api handles this
    for callers -- calling this function directly with non-canonical node
    labels will silently produce wrong results, since the wire format has no
    way to represent arbitrary labels).

    Returns dict[node] -> tuple of motif counts (13-tuple for level=3,
    199-tuple for level=4 on directed graphs; fewer for undirected).
    """
    if level not in (3, 4):
        raise ValueError(f"level must be 3 or 4, got {level}")
    logger.info("motif%d: dispatching to graph-measures subprocess, gpu=%s%s.",
                level, gpu, f" (device={device})" if gpu else "")

    motif_env_python = os.environ.get("GRAPHFX_MOTIF_ENV_PYTHON", _DEFAULT_MOTIF_ENV_PYTHON)
    graph_measures_path = os.environ.get("GRAPHFX_GRAPH_MEASURES_PATH", _DEFAULT_GRAPH_MEASURES_PATH)
    if not os.path.exists(motif_env_python):
        raise RuntimeError(
            f"motif-env Python not found at {motif_env_python!r}. Set "
            "GRAPHFX_MOTIF_ENV_PYTHON, or build it with scripts/setup.sh."
        )
    if not os.path.isdir(graph_measures_path):
        raise RuntimeError(
            f"graph-measures checkout not found at {graph_measures_path!r}. Set "
            "GRAPHFX_GRAPH_MEASURES_PATH, or run scripts/setup.sh."
        )

    indices, neighbors = _to_csr(G)
    spec = {
        "indices": indices,
        "neighbors": neighbors,
        "directed": G.is_directed(),
        "level": level,
        "gpu": gpu,
        "device": device,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.json")
        output_path = os.path.join(tmpdir, "output.csv")
        with open(input_path, "w") as f:
            json.dump(spec, f)

        env = dict(os.environ)
        env["PYTHONPATH"] = graph_measures_path + os.pathsep + env.get("PYTHONPATH", "")

        result = subprocess.run(
            [motif_env_python, _WORKER_SCRIPT, input_path, output_path],
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"motif{level} subprocess failed (exit {result.returncode}):\n"
                f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
            )

        with open(output_path, newline="") as f:
            reader = csv.reader(f)
            next(reader)  # header
            out: dict[int, tuple] = {}
            for row in reader:
                node = int(row[0])
                out[node] = tuple(int(float(v)) for v in row[1:])

    return out
