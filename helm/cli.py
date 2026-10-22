"""Command line interface: ``python -m helm.cli {run,convert,summary}``.

    run      features -> train -> score -> arborescence -> metrics
    convert  plain edge-list CSVs -> HELM graph files and manifest
    summary  mean/std of the per-graph metrics of a finished run

The stages are also runnable on their own:
``python -m helm.scoring.xgb``, ``helm.scoring.gnn``, ``helm.arborescence.edmonds``.
"""
import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from helm.io import GraphFormatError, load_graph, load_manifest, save_graph, validate_graph

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRIC_COLUMNS = [
    "directed_recall",
    "undirected_recall",
    "directed_precision",
    "root_accuracy",
    "root_distance",
]


def _graphfx_python():
    return os.environ.get("HELM_GRAPHFX_PYTHON", os.path.join(REPO_ROOT, "graphfx", ".venv", "bin", "python"))


def _has_gpu():
    return shutil.which("nvidia-smi") is not None and subprocess.run(
        ["nvidia-smi", "-L"], capture_output=True
    ).returncode == 0


def _entries(manifest_path):
    with open(manifest_path) as f:
        return json.load(f)


def _collections(manifest_path):
    return sorted({e["collection"] for e in _entries(manifest_path)})


def _compute_features(manifest, gpu, skip_motifs, workers, force):
    py = _graphfx_python()
    if not os.path.exists(py):
        raise SystemExit(
            f"graphfx environment not found at {py}. Run scripts/setup.sh, or set "
            "HELM_GRAPHFX_PYTHON to the python of an environment with graphfx installed."
        )
    cmd = [py, os.path.join(REPO_ROOT, "graphfx", "scripts", "edge_features.py"), "--manifest", manifest]
    cmd.append("--gpu" if gpu else "--no-gpu")
    if workers:
        cmd += ["--workers", str(workers)]
    if skip_motifs:
        cmd.append("--skip-motifs")
    if force:
        cmd.append("--force")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def _check_backends(manifests):
    """Warn if the features of the graphs in this run come from different implementations.

    graphfx writes features/backend.json next to the feature files. The Louvain and Leiden
    features differ between the CPU and the GPU implementation (see graphfx/docs/FEATURES.md),
    and motif features may be absent, so a model should be trained and applied on features that
    were computed the same way. Precomputed features without a record cannot be checked.
    """
    records, missing = {}, 0
    for manifest in manifests:
        for e in _entries(manifest):
            path = os.path.join(os.path.dirname(e["node_features_path"]), "backend.json")
            if os.path.exists(path):
                with open(path) as f:
                    records[(e["collection"], e["graph_id"])] = json.load(f)
            else:
                missing += 1
    if missing:
        print(f"[helm] note: {missing} graph(s) have no feature backend record (precomputed features?); "
              "cannot check that all features come from the same backend", file=sys.stderr)
    for key, label in (("gpu_backend_active", "GPU and CPU"), ("has_motifs", "with and without motif")):
        if len({r[key] for r in records.values()}) > 1:
            print(f"[helm] WARNING: the features were computed with {label} implementations mixed in one run. "
                  "Scores from a model trained on one kind are not reliable for the other "
                  "(see graphfx/docs/FEATURES.md).", file=sys.stderr)


def _train_config(args, collection):
    """Path of the hyperparameter JSON to use, or None to tune with Optuna."""
    if args.hyperparams:
        return args.hyperparams
    if args.tune:
        return None
    default = os.path.join(REPO_ROOT, "configs", f"{args.model}_{collection}.json")
    return default if os.path.exists(default) else None


def cmd_run(args):
    out = os.path.abspath(args.output_dir)
    if os.path.exists(out) and os.listdir(out) and not args.overwrite:
        raise SystemExit(f"{out} already exists and is not empty; choose another --output-dir or pass --overwrite")
    train_src = args.train
    apply_src = args.apply
    if args.overwrite and os.path.exists(out):
        for c in _collections(train_src) + (_collections(apply_src) if apply_src else []):
            shutil.rmtree(os.path.join(out, c), ignore_errors=True)
        shutil.rmtree(os.path.join(out, "manifests"), ignore_errors=True)
    train_m = load_manifest(train_src, out, "train", args.features_dir)
    apply_m = load_manifest(apply_src, out, "apply", args.features_dir) if apply_src else None

    gpu = _has_gpu() if args.gpu == "auto" else args.gpu == "on"
    if args.features_dir:
        print(f"[helm] using precomputed features from {args.features_dir}", flush=True)
    else:
        print(f"[helm] features (gpu={gpu}, skip_motifs={args.skip_motifs})", flush=True)
        for m in filter(None, [train_m, apply_m]):
            _compute_features(m, gpu, args.skip_motifs, args.workers, False)
    _check_backends([m for m in (train_m, apply_m) if m])

    # Heavy imports only after the features stage so `convert`/`summary` stay light.
    from helm.arborescence.edmonds import process_manifest, write_results_csv

    if args.model == "xgb":
        from helm.scoring.xgb import score_manifest, train_manifest
    else:
        from helm.scoring import gnn

    all_rows = {"train": [], "apply": []}
    for collection in _collections(train_src):
        first = next(e for e in _entries(train_m) if e["collection"] == collection)
        directed_graph = load_graph(first["G_path"]).is_directed()
        directed_mode = (not directed_graph) if args.directed_mode == "auto" else args.directed_mode == "on"
        print(f"[helm] {collection}: graph directed={directed_graph}, directed_mode={directed_mode}", flush=True)

        # Edge sampling uses the global RNGs, so seed them as the stage CLIs do.
        random.seed(args.seed)
        np.random.seed(args.seed)
        config = _train_config(args, collection)
        common = dict(seed=args.seed, directed_mode=directed_mode)
        print(f"[helm] {collection}: training {args.model} ({'config ' + config if config else 'Optuna'})", flush=True)
        if args.model == "xgb":
            train_manifest(
                train_m, collection, out,
                n_trials=args.n_trials, n_jobs=args.n_jobs,
                optuna_enabled=config is None, hyperparams_config=config, **common,
            )
        else:
            import torch

            if args.deterministic:
                # CUDA scatter/index operations accumulate with atomics, so the GPU result
                # can change from run to run even with a fixed seed.
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
                torch.use_deterministic_algorithms(True, warn_only=True)
            device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
            gnn.train_manifest(
                train_m, collection, out, device,
                n_trials=args.n_trials,
                optuna_enabled=config is None, hyperparams_config=config, **common,
            )
        model_dir = os.path.join(out, collection, "model")
        if apply_m and collection in _collections(apply_m):
            print(f"[helm] {collection}: scoring apply set", flush=True)
            if args.model == "xgb":
                score_manifest(apply_m, collection, out, model_dir)
            else:
                gnn.score_manifest(apply_m, collection, out, model_dir, device)

        for split, manifest in (("train", train_m), ("apply", apply_m)):
            if not manifest or collection not in _collections(manifest):
                continue
            print(f"[helm] {collection}: arborescence on {split} set", flush=True)
            rows = process_manifest(
                manifest, collection, out,
                workers=args.workers or 1, directed_mode=directed_mode, solver=args.solver,
            )
            write_results_csv(rows, os.path.join(out, collection, f"results_{split}.csv"))
            all_rows[split] += rows

    for split, rows in all_rows.items():
        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(out, f"results_{split}.csv"), index=False)
    print(f"[helm] done. Per-graph metrics: {out}/results_*.csv", flush=True)
    for split in ("train", "apply"):
        path = os.path.join(out, f"results_{split}.csv")
        if os.path.exists(path):
            print(f"\n== {split} ==")
            _print_summary(path)


def _print_summary(path):
    df = pd.read_csv(path)
    if "error" in df:
        failed = df[df["error"].notna()]
        for _, r in failed.iterrows():
            print(f"  FAILED {r['collection']}/{r['graph_id']}: {r['error']}")
        df = df[df["error"].isna()]
    cols = [c for c in METRIC_COLUMNS if c in df and df[c].notna().any()]
    if not cols:
        print(f"  {len(df)} graphs processed (no ground truth, so no metrics)")
        return
    summary = df.groupby("collection")[cols].agg(["mean", "std"]).round(3)
    print(summary.to_string())


def cmd_summary(args):
    _print_summary(args.results)


def _read_pairs(path):
    pairs = []
    with open(path, newline="") as f:
        for lineno, row in enumerate(csv.reader(f), 1):
            if not row or row[0].lstrip().startswith("#"):
                continue
            if lineno == 1 and not row[0].strip().lstrip("-").isdigit():
                continue  # header row
            if len(row) < 2:
                raise GraphFormatError(f"{path}:{lineno}: expected two columns, got {row}")
            try:
                pairs.append((int(row[0]), int(row[1])))
            except ValueError:
                raise GraphFormatError(f"{path}:{lineno}: vertex IDs must be integers, got {row[:2]}") from None
    return pairs


def cmd_convert(args):
    import networkx as nx

    for path in (args.edges, args.hierarchy):
        if path and not os.path.isfile(path):
            raise GraphFormatError(f"file not found: {path}")

    gdir = os.path.join(args.data_dir, args.collection, args.name)
    if os.path.exists(gdir) and not args.overwrite:
        raise SystemExit(f"{gdir} exists; pass --overwrite to replace it")
    G = nx.DiGraph() if args.directed else nx.Graph()
    G.add_edges_from(_read_pairs(args.edges))
    validate_graph(G, args.edges)
    T = None
    if args.hierarchy:
        T = nx.DiGraph()
        T.add_edges_from(_read_pairs(args.hierarchy))
        if not nx.is_arborescence(T):
            raise GraphFormatError(f"{args.hierarchy}: parent->child edges do not form a rooted tree")
        missing = set(T.nodes) - set(G.nodes)
        if missing:
            raise GraphFormatError(f"{args.hierarchy}: {len(missing)} tree vertices are not in the graph")
    save_graph(G, os.path.join(gdir, "G.edgelist"))
    entry = {"graph_id": args.name, "collection": args.collection, "G_path": os.path.join(gdir, "G.edgelist")}
    if T is not None:
        save_graph(T, os.path.join(gdir, "T.edgelist"))
        entry["T_path"] = os.path.join(gdir, "T.edgelist")
    import json

    mpath = args.manifest
    entries = json.load(open(mpath)) if os.path.exists(mpath) else []
    entries = [e for e in entries if e["graph_id"] != args.name or e["collection"] != args.collection] + [entry]
    os.makedirs(os.path.dirname(os.path.abspath(mpath)), exist_ok=True)
    with open(mpath, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"wrote {gdir}/ and added {args.name} to {mpath}")


def build_parser():
    p = argparse.ArgumentParser(prog="helm", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run the whole pipeline")
    r.add_argument("--train", required=True, help="manifest of graphs with (partial) ground truth: the model is trained on a sample of their edges (few-shot)")
    r.add_argument("--apply", help="manifest of graphs to score with the trained model, no labels used (zero-shot)")
    r.add_argument("--output-dir", required=True)
    r.add_argument("--model", choices=["xgb", "gnn"], default="xgb")
    r.add_argument("--directed-mode", choices=["auto", "on", "off"], default="auto",
                   help="bidirect undirected graphs and let the arborescence pick orientation (auto: on for undirected graphs)")
    r.add_argument("--gpu", choices=["auto", "on", "off"], default="auto")
    r.add_argument("--skip-motifs", action="store_true",
                   help="drop motif3/motif4 features (changes the feature set, so results differ from runs with motifs)")
    r.add_argument("--features-dir",
                   help="use precomputed features at <dir>/<collection>/<graph_id>/{node,edge}_features.csv instead of computing them")
    r.add_argument("--hyperparams", help="model hyperparameter JSON (default: configs/<model>_<collection>.json when present)")
    r.add_argument("--tune", action="store_true", help="run Optuna instead of using a config")
    r.add_argument("--n-trials", type=int, default=50, help="Optuna trials when tuning")
    r.add_argument("--n-jobs", type=int, default=-1, help="XGBoost threads (-1: all cores)")
    r.add_argument("--workers", type=int, default=None, help="parallel workers for CPU features and the arborescence stage")
    r.add_argument("--solver", choices=["tarjan", "networkx"], default="tarjan")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--deterministic", action="store_true",
                   help="GNN only: use deterministic PyTorch kernels (slower; makes GPU runs repeatable for a fixed seed)")
    r.add_argument("--overwrite", action="store_true", help="reuse an existing output directory, replacing what this run writes")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("convert", help="convert edge-list CSVs into HELM graph files and a manifest")
    c.add_argument("--edges", required=True, help="CSV with one 'source,target' pair per row")
    c.add_argument("--hierarchy", help="optional CSV with one 'parent,child' pair per row")
    g = c.add_mutually_exclusive_group(required=True)
    g.add_argument("--directed", action="store_true")
    g.add_argument("--undirected", dest="directed", action="store_false")
    c.add_argument("--name", required=True, help="graph ID")
    c.add_argument("--collection", default="custom")
    c.add_argument("--data-dir", default="data")
    c.add_argument("--manifest", default="manifests/custom.json", help="manifest to create or append to")
    c.add_argument("--overwrite", action="store_true")
    c.set_defaults(func=cmd_convert)

    s = sub.add_parser("summary", help="print metrics of a finished run")
    s.add_argument("results", help="results_train.csv or results_apply.csv")
    s.set_defaults(func=cmd_summary)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (GraphFormatError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
