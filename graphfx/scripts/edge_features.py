"""Compute graphfx node and edge features for the graphs in a manifest.

Each manifest entry needs ``G_path``, ``node_features_path`` and
``edge_features_path`` (``helm.io.load_manifest`` fills the last two in). The
CSVs are written to those paths; entries whose CSVs already exist are skipped
unless ``--force`` is given.

Run from the repo root with graphfx's own environment:

    graphfx/.venv/bin/python graphfx/scripts/edge_features.py \\
        --manifest runs/mb/manifests/train.json --gpu

See ../docs/FEATURES.md for the feature list.
"""
import argparse
import json
import os
import sys
from importlib import metadata
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import graphfx  # noqa: E402
from graphfx.features import (  # noqa: E402
    DEFAULT_EDGE_FEATURES,
    DEFAULT_NODE_FEATURES,
    EDGE_FEATURES,
    NODE_FEATURES,
    Backend,
)
from helm.io import load_graph  # noqa: E402


def split_only_features(names):
    """Split requested feature names into (node_names, edge_names); fail fast on unknowns."""
    unknown = [n for n in names if n not in NODE_FEATURES and n not in EDGE_FEATURES]
    if unknown:
        valid = sorted(set(NODE_FEATURES) | set(EDGE_FEATURES))
        raise SystemExit(f"Unknown feature name(s): {unknown}. Valid names: {valid}")
    return [n for n in names if n in NODE_FEATURES], [n for n in names if n in EDGE_FEATURES]


def _version(package):
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def write_backend_record(directory, gpu_requested, node_df):
    """Record which implementation produced the features. Louvain and Leiden differ between the
    CPU (NetworkX) and GPU (cuGraph) implementations, so features from the two backends should
    not be mixed between training and applying a model."""
    record = {
        "gpu_requested": bool(gpu_requested),
        "gpu_backend_active": bool(graphfx.nx_backend.gpu_backend_active()),
        "has_motifs": any(c.startswith("motifs_node_") for c in node_df.columns),
        "versions": {
            "networkx": _version("networkx"),
            "nx-cugraph": _version("nx-cugraph-cu12"),
            "cugraph": _version("cugraph-cu12"),
            "graphfx": _version("graphfx"),
        },
    }
    with open(os.path.join(directory, "backend.json"), "w") as f:
        json.dump(record, f, indent=2)


def process_entry(entry, gpu, only_node_features, only_edge_features, force):
    """Compute features for one manifest entry. Returns True on success."""
    name = f"{entry['collection']}/{entry['graph_id']}"
    node_path, edge_path = entry["node_features_path"], entry["edge_features_path"]
    if not force and os.path.exists(node_path) and os.path.exists(edge_path):
        print(f"[skip] {name}: CSVs already exist (use --force to recompute)", flush=True)
        return True
    try:
        G = load_graph(entry["G_path"])
        node_df = graphfx.compute_node_features(G, features=only_node_features, gpu=gpu)
        edge_df = graphfx.compute_edge_features(G, features=only_edge_features, gpu=gpu)
    except Exception as e:
        print(f"[FAIL] {name}: {e}", flush=True)
        return False
    os.makedirs(os.path.dirname(node_path), exist_ok=True)
    node_df.to_csv(node_path, index=False)
    edge_df.to_csv(edge_path, index=False)
    write_backend_record(os.path.dirname(node_path), gpu, node_df)
    print(f"[ok] {name}: node={node_df.shape} edge={edge_df.shape}", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description="Compute graphfx features for a manifest's graphs.")
    parser.add_argument("--manifest", required=True, help="Manifest JSON with feature paths filled in")
    parser.add_argument("--collection", default=None, help="Only process entries for this collection")
    parser.add_argument(
        "--only-features", nargs="+", default=None,
        help="Restrict to these feature names (replaces the default set for this run, not additive)",
    )
    parser.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=True,
                        help="GPU path runs sequentially (one CUDA context); --no-gpu uses --workers parallel CPU workers")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="CPU workers (--no-gpu only; default min(cpu_count, num_graphs)). Ignored with --gpu.",
    )
    parser.add_argument(
        "--skip-motifs", action="store_true",
        help="Drop the motif3/motif4 features (they need the graph-measures motif environment). "
             "This changes the feature set, so scores are not comparable to runs that include motifs.",
    )
    parser.add_argument("--force", action="store_true", help="Recompute even if CSVs already exist")
    args = parser.parse_args()

    with open(args.manifest) as f:
        entries = json.load(f)
    if args.collection:
        entries = [e for e in entries if e.get("collection") == args.collection]
    if not entries:
        raise SystemExit("No manifest entries to process (check --manifest / --collection).")

    only_node_features, only_edge_features = (None, None)
    if args.only_features:
        only_node_features, only_edge_features = split_only_features(args.only_features)
    elif args.skip_motifs:
        only_node_features = [n for n in DEFAULT_NODE_FEATURES if NODE_FEATURES[n].backend is not Backend.MOTIF]
        only_edge_features = [n for n in DEFAULT_EDGE_FEATURES if EDGE_FEATURES[n].backend is not Backend.MOTIF]

    if args.gpu and args.workers and args.workers > 1:
        print(f"[warn] --gpu with --workers {args.workers} requested; a CUDA context is "
              "process-bound, running sequentially instead.", flush=True)

    if args.gpu:
        ok = [process_entry(e, True, only_node_features, only_edge_features, args.force) for e in entries]
    else:
        workers = args.workers or min(os.cpu_count() or 1, len(entries))
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(process_entry, e, False, only_node_features, only_edge_features, args.force)
                for e in entries
            ]
            ok = [f.result() for f in as_completed(futures)]

    failed = len(ok) - sum(ok)
    print(f"\n{len(ok) - failed}/{len(ok)} graphs done", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
