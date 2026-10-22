"""Minimum spanning arborescence of the scored graph: structure and root in one step.

Every edge (u, v) gets the log-odds weight w(u, v) = -log(s / (1 - s)), with s = score(u, v)
clipped to [1e-9, 1 - 1e-9]. If every edge is hierarchical independently with probability s,
conditioned on the hierarchical edges forming a spanning arborescence, the most likely tree
is the one that minimises the total weight. Chu-Liu/Edmonds then returns the rooted directed
tree of minimum weight, which fixes the edge set, the edge directions and the root together.

Scores are read as directed (u, v) pairs, so score(u -> v) and score(v -> u) can differ.
For undirected input graphs (``directed_mode``) the score file holds both orientations of
every edge and the arborescence chooses between them.

Output per graph, under <output_dir>/<collection>/<graph_id>/edmonds/:
    arborescence.pkl, arborescence.edgelist   the predicted tree (nx.DiGraph)
    edmonds_results.json                      its metrics (only with a ground-truth tree)
"""

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from helm.io import load_graph, save_graph

# ---------------------------------------------------------------------------
# Output file names
# ---------------------------------------------------------------------------

EDMONDS_DIR = "edmonds"
ARBORESCENCE_FILE = "arborescence.pkl"
RESULTS_FILE = "edmonds_results.json"


def _graph_dir(output_dir: str, collection: str, gid: str) -> str:
    return os.path.join(output_dir, collection, gid)


# ---------------------------------------------------------------------------
# Score loading - preserves direction
# ---------------------------------------------------------------------------


def load_directed_scores(score_path: str) -> Dict[Tuple, float]:
    """
    Load edge scores from CSV as directed (u, v) -> float mapping.

    Unlike the MST pipeline this does NOT convert to frozensets.
    Columns expected: source, target, score.
    """
    df = pd.read_csv(score_path)
    required = {"source", "target", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"score CSV {score_path} missing columns: {missing}. "
            f"Found: {list(df.columns)}"
        )
    return {(row.source, row.target): float(row.score) for row in df.itertuples()}


# ---------------------------------------------------------------------------
# Build directed weighted graph
# ---------------------------------------------------------------------------


_SCORE_EPS = 1e-9


def build_directed_weighted_graph(
    G_directed: nx.DiGraph,
    directed_scores: Dict[Tuple, float],
) -> Tuple[nx.DiGraph, int]:
    """
    Build a weighted DiGraph for Edmonds' algorithm.

    The weight of an edge is -log(s / (1 - s)), with s clipped to [1e-9, 1 - 1e-9], so
    minimising the total weight gives the most likely arborescence.

    Returns
    -------
    D_weighted : nx.DiGraph with 'weight' attribute on edges
    n_missing  : number of edges in G that had no score (given weight 0, i.e. a neutral
                 score of 0.5)
    """
    D = nx.DiGraph()
    D.add_nodes_from(G_directed.nodes())

    n_missing = 0
    for u, v in G_directed.edges():
        score = directed_scores.get((u, v))
        if score is None:
            # No score for this directed edge: weight 0 (neutral, s = 0.5 under logodds)
            n_missing += 1
            w = 0.0
        else:
            s = min(max(score, _SCORE_EPS), 1.0 - _SCORE_EPS)
            w = -math.log(s / (1.0 - s))
        D.add_edge(u, v, weight=w)

    return D, n_missing


# ---------------------------------------------------------------------------
# Edmonds' algorithm wrapper
# ---------------------------------------------------------------------------


SOLVERS = ("networkx", "tarjan")


def run_edmonds(
    D_weighted: nx.DiGraph,
    solver: str = "tarjan",
) -> Tuple[Optional[nx.DiGraph], Optional[str]]:
    """
    Run Edmonds' minimum spanning arborescence on D_weighted.

    solver: "tarjan" (default; helm/arborescence/dmst_tarjan.py, O(E log V)) or
    "networkx" (nx.minimum_spanning_arborescence, the reference implementation). Both
    return an arborescence of optimal total weight; ties may be broken differently.

    Returns
    -------
    arborescence : nx.DiGraph that is a valid arborescence, or None on failure
    error        : None on success, error message string on failure
    """
    if solver not in SOLVERS:
        raise ValueError(f"unknown solver {solver!r}; choose from {SOLVERS}")
    try:
        if solver == "tarjan":
            from helm.arborescence.dmst_tarjan import minimum_spanning_arborescence_nx

            arborescence = minimum_spanning_arborescence_nx(D_weighted, attr="weight")
        else:
            arborescence = nx.minimum_spanning_arborescence(D_weighted, attr="weight")
    except ValueError as exc:
        return None, f"ValueError: {exc}"
    except nx.exception.NetworkXException as exc:
        return None, f"NetworkXException: {exc}"
    except Exception as exc:
        return None, f"Unexpected error: {exc}"

    return arborescence, None


# ---------------------------------------------------------------------------
# Metric helpers (self-contained, no imports from optimal_root)
# ---------------------------------------------------------------------------


def compute_root_distance(
    pred_root,
    true_tree: nx.DiGraph,
    true_root,
) -> Optional[int]:
    """
    Shortest-path hop distance from pred_root to true_root in the
    undirected version of true_tree.  Returns None if pred_root is not
    a node in true_tree.
    """
    if pred_root not in true_tree.nodes():
        return None
    T_ud = true_tree.to_undirected()
    try:
        return nx.shortest_path_length(T_ud, pred_root, true_root)
    except nx.NetworkXNoPath:
        return None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_arborescence(
    arborescence: nx.DiGraph,
    true_tree: nx.DiGraph,
) -> Dict[str, Any]:
    """
    Compute all quality metrics for a predicted arborescence vs ground truth.

    Metrics
    -------
    directed_recall      TP / |true_edges|
    directed_precision   TP / |pred_edges|
    directed_f1
    undirected_recall    edge-set recall ignoring direction
    tp / fp / fn         counts
    root_pred            node id with in_degree 0 in arborescence
    root_true            node id with in_degree 0 in true_tree
    root_accuracy        1 if root_pred == root_true else 0
    root_distance        hop distance in T (undirected) between pred and true root
    """
    # ---- roots --------------------------------------------------------
    pred_roots = [n for n in arborescence.nodes() if arborescence.in_degree(n) == 0]
    true_roots = [n for n in true_tree.nodes() if true_tree.in_degree(n) == 0]

    root_pred = pred_roots[0] if pred_roots else None
    root_true = true_roots[0] if true_roots else None

    root_accuracy = int(root_pred is not None and root_pred == root_true)
    root_distance = (
        compute_root_distance(root_pred, true_tree, root_true)
        if (root_pred is not None and root_true is not None)
        else None
    )

    # ---- edge metrics -------------------------------------------------
    pred_edges = set(arborescence.edges())
    true_edges = set(true_tree.edges())

    tp = len(pred_edges & true_edges)
    fp = len(pred_edges) - tp
    fn = len(true_edges) - tp

    directed_recall = tp / len(true_edges) if true_edges else 0.0
    directed_precision = tp / len(pred_edges) if pred_edges else 0.0
    directed_f1 = (
        2
        * directed_precision
        * directed_recall
        / (directed_precision + directed_recall)
        if (directed_precision + directed_recall) > 0
        else 0.0
    )

    pred_ud = {frozenset(e) for e in pred_edges}
    true_ud = {frozenset(e) for e in true_edges}
    tp_ud = len(pred_ud & true_ud)
    undirected_recall = tp_ud / len(true_ud) if true_ud else 0.0

    return {
        "directed_recall": float(directed_recall),
        "directed_precision": float(directed_precision),
        "directed_f1": float(directed_f1),
        "undirected_recall": float(undirected_recall),
        "tp_directed": int(tp),
        "fp_directed": int(fp),
        "fn_directed": int(fn),
        "pred_edges": int(len(pred_edges)),
        "true_edges": int(len(true_edges)),
        "root_pred": root_pred,
        "root_true": root_true,
        "root_accuracy": int(root_accuracy),
        "root_distance": int(root_distance) if root_distance is not None else None,
    }


# ---------------------------------------------------------------------------
# Per-graph processing (top-level function - must be picklable for workers)
# ---------------------------------------------------------------------------


def process_graph(
    entry: Dict[str, Any],
    output_dir: str,
    collection: str,
    verbose: bool = False,
    directed_mode: bool = False,
    solver: str = "tarjan",
) -> Dict[str, Any]:
    """
    Run Edmonds' arborescence pipeline for a single graph.

    Saves the arborescence and its metrics under <output_dir>/<collection>/<gid>/edmonds/.

    Returns
    -------
    Result dict with all metrics + error field (None on success).
    """
    gid = entry["graph_id"]

    base_result = {
        "graph_id": gid,
        "collection": collection,
        "split": entry.get("split", "unknown"),
        "directed_recall": None,
        "directed_precision": None,
        "directed_f1": None,
        "undirected_recall": None,
        "tp_directed": None,
        "fp_directed": None,
        "fn_directed": None,
        "pred_edges": None,
        "true_edges": None,
        "root_pred": None,
        "root_true": None,
        "root_accuracy": None,
        "root_distance": None,
        "n_nodes": None,
        "n_scored_edges": None,
        "n_missing_scores": None,
        "solver": solver,
        "error": None,
    }

    try:
        # ---- load inputs -----------------------------------------------
        g_path = entry.get("G_path", "")
        t_path = entry.get("T_path", "")
        score_path = entry.get("score_path", "")

        # T_path is optional: without ground truth the tree is produced, not scored.
        required = [("G_path", g_path), ("score_path", score_path)]
        if t_path:
            required.append(("T_path", t_path))
        for label, path in required:
            if not path or not os.path.exists(path):
                base_result["error"] = f"Missing {label}: {path!r}"
                return base_result

        G = load_graph(g_path)
        T = load_graph(t_path) if t_path else None

        if not G.is_directed():
            if directed_mode:
                # Bidirected mode: score CSV already has both directions;
                # convert G so build_directed_weighted_graph sees all edges.
                G = G.to_directed()
            else:
                base_result["error"] = (
                    f"Entity graph is undirected (type={type(G).__name__}). "
                    "Edmonds' directed-only strategy requires a DiGraph. "
                    "Pass --directed-mode for undirected datasets (microbiome, memetracker)."
                )
                return base_result

        directed_scores = load_directed_scores(score_path)

        # ---- build weighted digraph ------------------------------------
        D_weighted, n_missing = build_directed_weighted_graph(G, directed_scores)

        base_result["n_nodes"] = G.number_of_nodes()
        base_result["n_scored_edges"] = len(directed_scores)
        base_result["n_missing_scores"] = n_missing

        if verbose:
            print(
                f"  {gid}: {G.number_of_nodes()} nodes, "
                f"{G.number_of_edges()} edges, "
                f"{n_missing} missing scores"
            )

        # ---- Edmonds' --------------------------------------------------
        arborescence, err = run_edmonds(D_weighted, solver=solver)

        if err is not None:
            base_result["error"] = err
            return base_result

        # ---- evaluate --------------------------------------------------
        if T is None:
            base_result["root_pred"] = next(
                n for n in arborescence if arborescence.in_degree(n) == 0
            )
            base_result["pred_edges"] = arborescence.number_of_edges()
        else:
            if not T.is_directed():
                base_result["error"] = (
                    "Ground-truth tree T is undirected - cannot compute directed recall."
                )
                return base_result

            metrics = evaluate_arborescence(arborescence, T)
            base_result.update(metrics)

            if verbose:
                print(
                    f"  {gid}: directed_recall={metrics['directed_recall']:.3f} "
                    f"root_acc={metrics['root_accuracy']} "
                    f"root_dist={metrics['root_distance']}"
                )

        # ---- save results ----------------------------------------------
        out_dir = os.path.join(_graph_dir(output_dir, collection, gid), EDMONDS_DIR)
        os.makedirs(out_dir, exist_ok=True)

        save_graph(arborescence, os.path.join(out_dir, ARBORESCENCE_FILE))
        save_graph(arborescence, os.path.join(out_dir, "arborescence.edgelist"))

        with open(os.path.join(out_dir, RESULTS_FILE), "w") as f:
            json.dump(base_result, f, indent=2, default=str)

    except Exception as exc:
        base_result["error"] = f"{type(exc).__name__}: {exc}"

    return base_result


# ---------------------------------------------------------------------------
# Manifest-level processing
# ---------------------------------------------------------------------------


def process_manifest(
    manifest_path: str,
    collection: str,
    output_dir: str,
    workers: int = 1,
    verbose: bool = False,
    directed_mode: bool = False,
    solver: str = "tarjan",
) -> List[Dict[str, Any]]:
    """
    Process all graphs in manifest with optional parallelism.

    Returns list of result dicts (one per graph).
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    entries = [e for e in manifest if e.get("collection") == collection]
    if not entries:
        raise ValueError(f"No entries for collection '{collection}' in {manifest_path}")

    if verbose:
        print(
            f"Processing {len(entries)} graphs "
            f"(collection={collection}, workers={workers})"
        )

    if workers == 1:
        results = []
        for i, entry in enumerate(entries):
            if verbose:
                print(f"[{i+1}/{len(entries)}] {entry['graph_id']}", flush=True)
            results.append(process_graph(entry, output_dir, collection, verbose, directed_mode, solver))
        return results

    results_map: Dict[str, Dict] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(process_graph, entry, output_dir, collection, False, directed_mode, solver): entry
            for entry in entries
        }
        done = 0
        for future in as_completed(futures):
            done += 1
            entry = futures[future]
            gid = entry["graph_id"]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "graph_id": gid,
                    "collection": collection,
                    "error": f"Worker exception: {exc}",
                }
            results_map[gid] = result
            if verbose:
                err = result.get("error")
                tag = (
                    f"ERROR: {err}"
                    if err
                    else f"dr={result.get('directed_recall', '?'):.3f}"
                )
                print(f"  [{done}/{len(entries)}] {gid} - {tag}", flush=True)

    # Preserve original manifest order
    gid_order = [e["graph_id"] for e in entries]
    return [results_map[gid] for gid in gid_order if gid in results_map]


def write_results_csv(results: List[Dict[str, Any]], output_path: str) -> None:
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Results written to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(
        description="Minimum spanning arborescence of graphs with directed edge scores."
    )
    p.add_argument(
        "--manifest",
        required=True,
        help="manifest JSON: list of entries with graph_id, collection, G_path, score_path "
        "(and T_path to also compute metrics)",
    )
    p.add_argument("--collection", required=True, help="dataset collection to process")
    p.add_argument("--output-dir", required=True, help="base output directory")
    p.add_argument("--results-csv", default=None, help="write the per-graph results to this CSV")
    p.add_argument("-w", "--workers", type=int, default=1, help="parallel workers [default: 1]")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument(
        "--directed-mode",
        dest="directed_mode",
        action="store_true",
        default=False,
        help="treat undirected graphs as bidirected: the score file holds both (u,v) and (v,u) "
        "for every edge and the arborescence picks the orientation (use together with the "
        "same flag at scoring time; microbiome and memetracker)",
    )
    p.add_argument(
        "--solver",
        choices=SOLVERS,
        default="tarjan",
        help="tarjan (default, O(E log V)) or networkx (reference implementation: same optimal "
        "weight, but orders of magnitude slower and far more memory on graphs with 100K edges)",
    )
    return p.parse_args()


def main():
    args = _parse_args()

    print("Edmonds arborescence search")
    print(f"  Manifest   : {args.manifest}")
    print(f"  Collection : {args.collection}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Workers    : {args.workers}")
    print(f"  Solver     : {args.solver}")

    results = process_manifest(
        args.manifest,
        args.collection,
        args.output_dir,
        workers=args.workers,
        verbose=args.verbose,
        directed_mode=args.directed_mode,
        solver=args.solver,
    )

    # Summary
    errors = [r for r in results if r.get("error")]
    successes = [r for r in results if not r.get("error")]

    print(f"\nSummary ({len(results)} graphs):")
    if errors:
        print(f"  Errors ({len(errors)}):")
        for r in errors:
            print(f"     {r['graph_id']}: {r['error']}")

    if successes:
        dr = [
            r["directed_recall"] for r in successes if r["directed_recall"] is not None
        ]
        ur = [
            r["undirected_recall"]
            for r in successes
            if r["undirected_recall"] is not None
        ]
        ra = [r["root_accuracy"] for r in successes if r["root_accuracy"] is not None]
        rd = [r["root_distance"] for r in successes if r["root_distance"] is not None]

        def _fmt(vals, fmt=".3f"):
            if not vals:
                return "n/a"
            return f"{np.mean(vals):{fmt}} ± {np.std(vals):{fmt}}"

        print(f"  Successful: {len(successes)}")
        print(f"  directed_recall    : {_fmt(dr)}")
        print(f"  undirected_recall  : {_fmt(ur)}")
        print(f"  root_accuracy      : {_fmt(ra, '.3f')}")
        print(f"  root_distance      : {_fmt(rd, '.2f')}")

    if args.results_csv:
        write_results_csv(results, args.results_csv)


if __name__ == "__main__":
    main()
