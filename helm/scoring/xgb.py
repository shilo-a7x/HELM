"""XGBoost edge scorer: trains a pooled model on sampled edges of the graphs in a manifest and
writes one probability per edge for every graph it scores.

Pipeline per collection (one model per graph family):
    1. read the node and edge feature CSVs of all graphs, pool them
    2. sample a few percent of the true tree edges as positives and four times as many other
       graph edges as negatives (the few-shot sample), split into train / validation
    3. fit XGBoost, either tuning hyperparameters with Optuna or using a fixed config
    4. score every edge of every graph and, where a ground-truth tree exists, report the
       outer-test metrics

Usage through the CLI: ``helm run``. Direct use: ``python -m helm.scoring.xgb --manifest ...``.
"""
import argparse
import os
import json
import pickle
import random
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    auc,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from helm.io import (
    EDGE_FEATURES,
    EDGE_SCORES,
    FEATURE_DIR,
    MODEL_DIR,
    NODE_FEATURES,
    SCORE_DIR,
    load_graph,
)

try:
    import optuna
    from optuna_integration import XGBoostPruningCallback

    _optuna_available = True
except ImportError:
    optuna = None
    XGBoostPruningCallback = None
    _optuna_available = False


# ------------------------
# Path Policy (enforced conventions)
# ------------------------
SCALER_FILE = "scaler.pkl"
META_FILE = "meta.json"
BEST_PARAMS_FILE = "best_params.json"
METRICS_FILE = "metrics.json"


def get_graph_dir(base_dir, collection, gid):
    """Get graph directory: base/collection/gid or base/gid if no collection."""
    if collection:
        return os.path.join(base_dir, collection, gid)
    return os.path.join(base_dir, gid)


def get_model_dir(base_dir, collection, gid=None):
    """Get model directory: base/collection/model or base/gid/model."""
    if collection:
        return os.path.join(base_dir, collection, MODEL_DIR)
    if gid:
        return os.path.join(base_dir, gid, MODEL_DIR)
    raise ValueError("Must provide either collection or gid")


def get_model_filename(model_type):
    """Get model filename without optuna suffix."""
    return f"{model_type}_model.pkl"


def validate_manifest_entry(entry):
    """Validate manifest entry has required fields, raise if missing."""
    required = ["graph_id", "collection"]
    missing = [f for f in required if f not in entry or not entry[f]]
    if missing:
        raise ValueError(f"Manifest entry missing required fields: {missing}")

    # Require explicit feature paths
    if "node_features_path" not in entry or "edge_features_path" not in entry:
        raise ValueError(
            f"Manifest entry for {entry.get('graph_id')} must specify node_features_path and edge_features_path"
        )

    return entry


def parse_args():
    parser = argparse.ArgumentParser(description="Edge scoring with Optuna + GBDT")
    parser.add_argument("--graph", type=str, help="Path to graph pkl")
    parser.add_argument("--tree", type=str, help="Path to tree pkl")
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output base directory"
    )
    parser.add_argument("--gid", type=str, help="Graph ID (required for single-graph)")
    parser.add_argument("--collection", type=str, help="Collection name")
    parser.add_argument(
        "--manifest", type=str, help="Manifest JSON for collection operations"
    )
    parser.add_argument(
        "--score-only", action="store_true", help="Only score (requires trained model)"
    )
    parser.add_argument(
        "--model-path", type=str, help="Path to model directory or file"
    )
    parser.add_argument("--n-trials", type=int, default=50, help="Optuna trial count")
    parser.add_argument(
        "--plateau-patience",
        type=int,
        default=20,
        help="PlateauCallback: patience (trials with no improvement before stopping study) [default: 20]",
    )
    parser.add_argument(
        "--plateau-min-trials",
        type=int,
        default=50,
        help="PlateauCallback: min_trials (minimum trials before early stopping allowed) [default: 50]",
    )
    parser.add_argument(
        "--plateau-min-delta",
        type=float,
        default=0.0001,
        help="PlateauCallback: min_delta (minimum improvement to reset patience) [default: 0.0001]",
    )
    parser.add_argument(
        "--pruner-startup-trials",
        type=int,
        default=15,
        help="MedianPruner: n_startup_trials (trials before pruning allowed) [default: 15]",
    )
    parser.add_argument(
        "--pruner-warmup-steps",
        type=int,
        default=10,
        help="MedianPruner: n_warmup_steps (iterations before pruning allowed in each trial) [default: 10]",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=50,
        help="XGBoost early_stopping_rounds (stop training if no improvement) [default: 50]",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs for XGBoost (default: -1 = all cores)",
    )
    parser.add_argument(
        "--optuna-enabled",
        dest="optuna_enabled",
        action="store_true",
        help="Run Optuna hyperparameter search (default)",
    )
    parser.add_argument(
        "--no-optuna",
        dest="optuna_enabled",
        action="store_false",
        help="Skip Optuna and use provided hyperparameters",
    )
    parser.set_defaults(optuna_enabled=True)
    parser.add_argument(
        "--hyperparams-config",
        type=str,
        help="Path to JSON with model hyperparameters. Required when Optuna is disabled; optional override otherwise.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (sets Python and NumPy RNG seeds) [default: 42]",
    )
    parser.add_argument(
        "--directed-mode",
        dest="directed_mode",
        action="store_true",
        default=False,
        help=(
            "For undirected graphs (microbiome, memetracker): double every edge (u,v) into (u,v) "
            "and (v,u) with signed difference features, label only the correctly oriented edge "
            "of T as positive, and write one score per orientation, so that the arborescence "
            "stage can recover edge directions. Use the same flag at the arborescence stage. "
            "Not needed for graphs that are already directed (wiki)."
        ),
    )
    return parser.parse_args()


def prepare_features(node_df, edge_df, use_abs_diff: bool = False):
    """Prepare features by computing avg/diff vectors from node features.

    If use_abs_diff is True (recommended for undirected graphs), the diff
    component is made sign-invariant to avoid arbitrary orientation.
    Does NOT scale - scaling is done once during training or loaded from saved scaler.
    """
    node_df.index.name = "node"

    # Extract source and target nodes from edge index
    # Handle both MultiIndex and tuple index cases
    if isinstance(edge_df.index, pd.MultiIndex):
        sources = edge_df.index.get_level_values(0)
        targets = edge_df.index.get_level_values(1)
    else:
        # Index is tuples: extract u and v from each tuple
        sources = pd.Index([e[0] for e in edge_df.index])
        targets = pd.Index([e[1] for e in edge_df.index])

    # Vectorized computation: get node feature vectors for source and target nodes
    u_vecs = node_df.loc[sources].values  # All source nodes
    v_vecs = node_df.loc[targets].values  # All target nodes

    # Compute avg and diff vectors for all edges at once
    avg_vec = (u_vecs + v_vecs) / 2
    diff_raw = (u_vecs - v_vecs) / 2
    diff_vec = np.abs(diff_raw) if use_abs_diff else diff_raw

    # Concatenate horizontally
    combined = np.concatenate([avg_vec, diff_vec], axis=1)

    avg_diff_df = pd.DataFrame(
        combined,
        index=edge_df.index,
        columns=[f"avg_{col}" for col in node_df.columns]
        + [f"diff_{col}" for col in node_df.columns],
    )

    edge_df = pd.concat([edge_df, avg_diff_df], axis=1)
    feature_cols = edge_df.columns.difference(["source", "target"])

    return edge_df, feature_cols


def save_roc_curve(y, y_proba, path, title=None):
    """Save a ROC curve plot to `path`. Requires >=2 classes present in y."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y, y_proba)
    auc_val = roc_auc_score(y, y_proba)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_val:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title or "ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def _pr_auc(y, y_proba):
    """Literal area under the precision-recall curve (trapezoidal rule via
    sklearn's auc()). Distinct from average_precision_score (see evaluate()'s
    docstring) -- provided alongside it so both are on record."""
    precision, recall, _ = precision_recall_curve(y, y_proba)
    return auc(recall, precision)


def evaluate(model, X, y, curve_path=None, curve_title=None):
    """Compute classification metrics. Precision/recall/f1 are macro-averaged
    (not the default binary-only average) since these datasets have extreme
    class imbalance and macro-averaging keeps the negative class's performance
    visible instead of collapsing to only the positive class's score. roc_auc
    stays the main/primary metric.

    Two PR-curve summaries are both reported, since they aren't the same
    number: average_precision (average_precision_score -- a step-function
    weighted sum, sklearn's recommended way to summarize a PR curve) and
    pr_auc (literal trapezoidal area under the precision-recall curve via
    auc(recall, precision) -- sklearn's own docs note this can be overly
    optimistic since linear interpolation between PR points isn't always
    achievable, which is why average_precision is usually preferred, but
    both are kept so it's a deliberate choice, not an accident).

    If curve_path is given, also saves a ROC curve plot there (requires both
    classes present in y).
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return metrics_from_proba(y, y_proba, y_pred=y_pred, curve_path=curve_path,
                              curve_title=curve_title)


def metrics_from_proba(y, y_proba, y_pred=None, curve_path=None, curve_title=None):
    """The metric dict evaluate() reports, computed from predicted
    probabilities directly (y_pred defaults to y_proba > 0.5). Shared with
    edge_scores_gnn.py so both models report identical metric definitions."""
    if y_pred is None:
        y_pred = (np.asarray(y_proba) > 0.5).astype(int)
    metrics = {
        "n_samples": int(len(y)),
        "n_positive": int(sum(y)),
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y, y_pred, average="macro", zero_division=0),
        "f1_score": f1_score(y, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y, y_proba),
        "average_precision": average_precision_score(y, y_proba),
        "pr_auc": _pr_auc(y, y_proba),
    }
    if curve_path and len(set(y)) > 1:
        save_roc_curve(y, y_proba, curve_path, title=curve_title)
    return metrics


# ------------------------
# Atomic I/O operations
# ------------------------


def load_features(graph_dir):
    """Load node and edge features from graph_dir/features/."""
    node_path = os.path.join(graph_dir, FEATURE_DIR, NODE_FEATURES)
    edge_path = os.path.join(graph_dir, FEATURE_DIR, EDGE_FEATURES)

    if not os.path.exists(node_path) or not os.path.exists(edge_path):
        raise FileNotFoundError(f"Features missing: {node_path} or {edge_path}")

    node_df = pd.read_csv(node_path)
    edge_df = pd.read_csv(edge_path)

    if "node" in node_df.columns:
        node_df = node_df.set_index("node")
    else:
        print(
            "WARNING: node_features.csv has no 'node' column. Using the integer index 0..n-1; "
            "the vertex IDs in the edge features must match these integers.",
            file=sys.stderr,
        )
    if "source" in edge_df.columns and "target" in edge_df.columns:
        edge_df["edge"] = list(zip(edge_df["source"], edge_df["target"]))
        edge_df = edge_df.set_index("edge")
    elif all(isinstance(e, tuple) and len(e) == 2 for e in edge_df.index):
        # edge_df index already contains (u, v) tuples
        pass
    else:
        raise ValueError(
            "edge_features.csv must either:\n"
            "  - contain 'source' and 'target' columns, OR\n"
            "  - have an index of (u, v) tuples.\n"
            "Neither condition is satisfied."
        )

    # ---- HARD VALIDATION: edge nodes must exist in node index ----
    edge_nodes = set(u for e in edge_df.index for u in e)
    missing_nodes = edge_nodes - set(node_df.index)

    if missing_nodes:
        raise ValueError(
            f"Edge/node index mismatch: {len(missing_nodes)} edge nodes "
            f"are missing from node features. "
            f"Examples: {list(missing_nodes)[:5]}. "
            "This would silently produce zero features, so execution is stopped."
        )

    # Aggressive: drop any node/edge feature column that contains any NaN
    dropped_node = []
    try:
        cols_before = list(node_df.columns)
        node_df.dropna(axis=1, inplace=True)
        dropped_node = [c for c in cols_before if c not in node_df.columns]
        if dropped_node:
            print(f"Warning: dropped node feature columns due to NaN: {dropped_node}")
    except Exception:
        print("Warning: failed to drop node feature columns with NaN", file=sys.stderr)

    dropped_edge = []
    try:
        cols_before = [c for c in edge_df.columns if c not in ("source", "target")]
        # preserve source/target if present, drop other cols with any NaN
        for c in cols_before:
            if edge_df[c].isna().any():
                dropped_edge.append(c)
                edge_df.drop(columns=[c], inplace=True)
        if dropped_edge:
            print(f"Warning: dropped edge feature columns due to NaN: {dropped_edge}")
    except Exception:
        print("Warning: failed to drop edge feature columns with NaN", file=sys.stderr)

    return node_df, edge_df


def load_model_artifacts(model_dir):
    """Load model, scaler, and meta from model_dir."""
    model_files = [f for f in os.listdir(model_dir) if f.endswith("_model.pkl")]
    if not model_files:
        raise FileNotFoundError(f"No model file found in {model_dir}")

    with open(os.path.join(model_dir, model_files[0]), "rb") as f:
        model = pickle.load(f)

    scaler = None
    scaler_path = os.path.join(model_dir, SCALER_FILE)
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    meta = {}
    meta_path = os.path.join(model_dir, META_FILE)
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)

    return model, scaler, meta


def save_model_artifacts(model_dir, model, scaler, meta, model_type):
    """Save model, scaler, and meta to model_dir."""
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, get_model_filename(model_type)), "wb") as f:
        pickle.dump(model, f)
    if scaler is not None:
        with open(os.path.join(model_dir, SCALER_FILE), "wb") as f:
            pickle.dump(scaler, f)
    if meta:
        with open(os.path.join(model_dir, META_FILE), "w") as f:
            json.dump(meta, f, indent=2)


# ------------------------
# Scoring
# ------------------------


def build_scoring_frame(meta, node_df, edge_df):
    """Build the exact edge rows + feature matrix a trained model scores.

    In directed_mode (from meta) the edge rows are bidirected first, so the
    result has one row per directed edge. Returns (processed_edge_df,
    feature_cols); processed_edge_df is indexed by (source, target) tuples and
    keeps source/target columns. Shared by score_graph, evaluate_manifest and
    edge_scores_gnn.py so every model scores the same rows.
    """
    # In directed_mode the model expects bidirected (doubled) edge input
    directed_mode = bool(meta.get("directed_mode")) if meta else False
    if directed_mode:
        # Ensure source/target columns exist before mirroring
        if "source" not in edge_df.columns or "target" not in edge_df.columns:
            edge_df = edge_df.reset_index()
            if "edge" in edge_df.columns:
                edge_df["source"] = [e[0] for e in edge_df["edge"]]
                edge_df["target"] = [e[1] for e in edge_df["edge"]]
        edge_df.index = pd.Index(list(zip(edge_df["source"], edge_df["target"])))
        mirror = _mirror_edge_df(edge_df)
        edge_df = pd.concat([edge_df, mirror])

    # Compute avg/diff features (unscaled)
    use_abs_diff = bool(meta.get("use_abs_diff")) if meta else False
    processed_edge_df, all_cols = prepare_features(
        node_df, edge_df, use_abs_diff=use_abs_diff
    )

    # Select feature columns from meta if available
    if meta and "feature_cols" in meta:
        feature_cols = meta["feature_cols"]
    else:
        feature_cols = list(all_cols)

    missing = [c for c in feature_cols if c not in processed_edge_df.columns]
    if missing:
        raise RuntimeError(f"Missing feature columns: {missing}")

    return processed_edge_df, feature_cols


def score_graph(model, scaler, meta, node_df, edge_df, out_path):
    """Score a single graph and write results to out_path.

    Args:
        model: Trained model
        scaler: Fitted scaler (or None)
        meta: Dict with 'feature_cols' (or empty)
        node_df: Node features DataFrame
        edge_df: Edge features DataFrame indexed by (u,v) tuples
        out_path: Where to write edge_scores.csv

    Returns:
        DataFrame with columns [source, target, score].  When the model was
        trained with directed_mode=True the CSV contains two rows per original
        undirected edge -- one for each direction -- so that Edmonds'
        arborescence can pick the correct orientation.
    """
    processed_edge_df, feature_cols = build_scoring_frame(meta, node_df, edge_df)

    # Keep a DataFrame with feature names so transformers that were fitted
    # on DataFrames with column names receive the same feature names at
    # prediction time. This avoids sklearn's "X does not have valid
    # feature names" warning and ensures correct feature -> column mapping.
    X_df = processed_edge_df[feature_cols]

    # Apply scaler (required for proper prediction)
    if scaler is not None:
        # scaler.transform accepts DataFrame and will preserve/validate
        # feature names when it was fitted with a DataFrame.
        X = scaler.transform(X_df)
    else:
        raise ValueError("Scaler is required for scoring but was not provided")

    # Predict
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    else:
        scores = model.predict(X)

    # Build result
    df = pd.DataFrame(
        {
            "source": processed_edge_df["source"].values,
            "target": processed_edge_df["target"].values,
            "score": scores,
        }
    )

    # Write to file
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    return df


# ------------------------
# Training
# ------------------------


def _build_params(
    trial,
    scale_pos_weight,
    seed=42,
    early_stopping_rounds=50,
    n_jobs=-1,
):
    """Build XGBoost hyperparameters from Optuna trial suggestions."""
    return {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "random_state": seed,
        "n_jobs": n_jobs,
        "early_stopping_rounds": early_stopping_rounds,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
        # Upper bound 50: microbiome's tuned alpha (8.3) sits well inside it.
        "alpha": trial.suggest_float("alpha", 1e-3, 50.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "scale_pos_weight": scale_pos_weight,
    }


def _build_model(params):
    """Build an XGBoost classifier with the given params."""
    return xgb.XGBClassifier(**params)


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    model_dir,
    n_trials,
    plateau_patience=20,
    plateau_min_trials=50,
    plateau_min_delta=0.0001,
    pruner_startup_trials=15,
    pruner_warmup_steps=10,
    early_stopping_rounds=50,
    n_jobs=-1,
    optuna_enabled=True,
    hyperparams_config=None,
    seed=42,
    warm_start_params=None,
):
    """Train model with Optuna hyperparameter optimization.

    Reproducibility:
        - Optuna sampler seeded with seed parameter
        - random.seed() and np.random.seed() initialized from seed
        - train_test_split uses random_state from seed
        - XGBoost uses random_state from seed

    Early Stopping Strategy (Dual-Layer):
        1. Trial-level pruning (MedianPruner):
           - Stops bad trials mid-training
           - n_startup_trials: First N trials complete without pruning
           - n_warmup_steps: Warmup iterations before pruning decision

        2. Study-level plateau detection (PlateauCallback):
           - Stops entire study when no improvement for patience trials
           - min_trials: Minimum trials before allowing stop
           - min_delta: Improvement threshold to count as progress

    Tuning Guidelines:
        - Too aggressive (low patience, high pruning): Risk missing good params
        - Too conservative (high patience, low pruning): Wasted compute
        - Recommended: patience=20, min_trials=50, n_startup_trials=15

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        model_dir: Where to save artifacts
        n_trials: Max Optuna trials

    Returns:
        (model, best_params, metrics)
    """
    # Compute scale_pos_weight for class imbalance (only used in Optuna branch)
    n_pos = int(sum(1 for v in y_train if v == 1))
    n_neg = int(sum(1 for v in y_train if v == 0))
    computed_scale_pos_weight = float(n_neg / max(1, n_pos))

    def _params_from_config(config_path):
        """Load model hyperparameters from JSON config file.

        Expected format (flat dict):
            {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "tree_method": "hist",
                "random_state": 42,
                "n_jobs": -1,
                "early_stopping_rounds": 50,
                "learning_rate": 0.1,
                "max_depth": 4,
                "subsample": 0.6,
                "colsample_bytree": 0.8,
                "lambda": 0.1,
                "alpha": 0.001,
                "n_estimators": 100,
                "scale_pos_weight": 4.0
            }

        All required fields must be present. Templates: configs/xgb_*.json.
        """
        with open(config_path, "r") as f:
            cfg = json.load(f)

        if not isinstance(cfg, dict):
            raise ValueError(f"Config must be a JSON object (dict), got {type(cfg)}")

        # Require these base fields (must be present in config, not filled with defaults)
        required_fields = {
            "objective",
            "random_state",
            "learning_rate",
            "max_depth",
            "n_estimators",
            "scale_pos_weight",
        }

        required_fields.update(
            [
                "eval_metric",
                "tree_method",
                "subsample",
                "colsample_bytree",
                "lambda",
                "alpha",
            ]
        )

        missing = required_fields - set(cfg.keys())
        if missing:
            raise ValueError(
                f"Config missing required fields: {missing}\n"
                "See the templates in configs/xgb_*.json"
            )

        model_params = cfg.copy()
        # The config's n_jobs is replaced; random_state and scale_pos_weight are kept.
        model_params["n_jobs"] = n_jobs  # always the caller's setting, never the config's
        return model_params

    # Branch: Optuna disabled -> train with provided config and skip tuning
    if not optuna_enabled:
        if hyperparams_config is None:
            raise ValueError("Optuna disabled but no --hyperparams-config was provided")

        params = _params_from_config(hyperparams_config)
        model = _build_model(params)

        # Train final model with early stopping (configured in model params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        best_params = params.copy()
        metrics = {
            "train": evaluate(
                model, X_train, y_train,
                curve_path=os.path.join(model_dir, "roc_train.png"),
                curve_title="ROC - train",
            ),
            "validation": evaluate(
                model, X_val, y_val,
                curve_path=os.path.join(model_dir, "roc_validation.png"),
                curve_title="ROC - validation",
            ),
        }

        # Save best params and metrics
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, BEST_PARAMS_FILE), "w") as f:
            json.dump(best_params, f, indent=4)
        with open(os.path.join(model_dir, METRICS_FILE), "w") as f:
            json.dump(metrics, f, indent=4)

        return model, best_params, metrics

    # Branch: Optuna enabled -> run hyperparameter tuning
    def objective(trial):
        params = _build_params(
            trial,
            computed_scale_pos_weight,
            seed=seed,
            early_stopping_rounds=early_stopping_rounds,
            n_jobs=n_jobs,
        )
        model = _build_model(params)

        # Fit with trial-level pruning (stops bad trials early during training)
        # Some XGBoost versions require callbacks passed at constructor time; inject here
        model.set_params(callbacks=[XGBoostPruningCallback(trial, "validation_0-auc")])
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        y_pred = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_pred)

    # Study-level early stopping: stop if no improvement for patience trials
    class PlateauCallback:
        """Stop Optuna study when plateau is detected (no improvement for N trials).

        Args:
            patience: Stop after N consecutive trials without improvement
            min_trials: Require minimum trials before stopping (avoid premature stop)
            min_delta: Improvement threshold (avoid stopping due to noise)
        """

        def __init__(self, patience=10, min_trials=30, min_delta=0.0001):
            self.patience = patience
            self.min_trials = min_trials
            self.min_delta = min_delta
            self.best_value = None
            self.trials_without_improvement = 0

        def __call__(self, study, trial):
            # Don't stop until we've seen enough trials
            if len(study.trials) < self.min_trials:
                return

            current_best = study.best_value
            if self.best_value is None:
                self.best_value = current_best
                return

            # Only count as improvement if it exceeds min_delta threshold
            if current_best > self.best_value + self.min_delta:
                self.best_value = current_best
                self.trials_without_improvement = 0
            else:
                self.trials_without_improvement += 1

            if self.trials_without_improvement >= self.patience:
                print(f"   Early stopping: no improvement for {self.patience} trials")
                study.stop()

    # Run Optuna with:
    # 1. MedianPruner: Prunes bad trials during training (trial-level)
    # 2. PlateauCallback: Stops entire study when plateau detected (study-level)
    # 3. TPESampler with seed for reproducibility
    if not _optuna_available:
        raise ImportError(
            "optuna is required for hyperparameter tuning. "
            "Install with: pip install optuna 'optuna-integration[xgboost]'"
        )
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),  # Seed Optuna's sampler!
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=pruner_startup_trials,
            n_warmup_steps=pruner_warmup_steps,
        ),
    )
    if warm_start_params:
        # Run the previously-tuned config as trial 0 so a modest --n-trials
        # budget can't do worse than the known-good baseline.
        print(f"   Warm-starting Optuna from previously-tuned params: {warm_start_params}")
        study.enqueue_trial(warm_start_params)
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=1,
        callbacks=[
            PlateauCallback(
                patience=plateau_patience,
                min_trials=plateau_min_trials,
                min_delta=plateau_min_delta,
            )
        ],
    )

    # Reconstruct best params with all base settings from best trial
    best_trial_params = {}
    for key, value in study.best_params.items():
        best_trial_params[key] = value

    print(
        f"   Completed {len(study.trials)} trials (stopped early: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])} pruned)"
    )

    # Create a mock trial to rebuild full params
    class MockTrial:
        def suggest_float(self, name, low, high, **kwargs):
            return best_trial_params.get(name, (low + high) / 2)

        def suggest_int(self, name, low, high, **kwargs):
            return best_trial_params.get(name, (low + high) // 2)

    # Replay best trial through _build_params to get full parameter dict
    full_params = _build_params(
        MockTrial(),
        computed_scale_pos_weight,
        seed=seed,
        early_stopping_rounds=early_stopping_rounds,
        n_jobs=n_jobs,
    )

    # Store best params for return (full params dict)
    best_params = full_params.copy()

    # Train final model with early stopping
    final_model = _build_model(full_params)
    final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Evaluate
    metrics = {
        "train": evaluate(
            final_model, X_train, y_train,
            curve_path=os.path.join(model_dir, "roc_train.png"),
            curve_title="ROC - train",
        ),
        "validation": evaluate(
            final_model, X_val, y_val,
            curve_path=os.path.join(model_dir, "roc_validation.png"),
            curve_title="ROC - validation",
        ),
    }

    # Save best params and metrics
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, BEST_PARAMS_FILE), "w") as f:
        json.dump(best_params, f, indent=4)
    with open(os.path.join(model_dir, METRICS_FILE), "w") as f:
        json.dump(metrics, f, indent=4)

    return final_model, best_params, metrics


# ------------------------
# Manifest operations (thin wrappers)
# ------------------------


def _mirror_edge_df(edge_df: "pd.DataFrame") -> "pd.DataFrame":
    """Return a copy of edge_df with source and target swapped.

    Symmetric edge-level features (e.g. edge_betweenness) are preserved as-is
    because they are the same regardless of direction.  The index is rebuilt as
    (target, source) tuples so it is compatible with prepare_features().
    """
    mirrored = edge_df.copy()
    # swap source / target columns
    mirrored["source"], mirrored["target"] = (
        edge_df["target"].values,
        edge_df["source"].values,
    )
    # rebuild tuple index - use names=None to avoid MultiIndex promotion crash
    # when the current index has a scalar name (e.g. "edge")
    mirrored.index = pd.Index(
        list(zip(mirrored["source"], mirrored["target"])), name=None
    )
    return mirrored


def pool_features_and_samples(
    manifest_path, collection, base_dir, directed_mode: bool = False
):
    """Load all features from manifest, pool them, and sample edges for training.

    Parameters
    ----------
    directed_mode : bool
        When True, undirected graphs are treated as bidirected: every edge
        (u, v) is doubled into (u, v) and (v, u) with signed diff features
        (use_abs_diff=False).  Labels match only the correct directed edge in T.
        Use this when you want Edmonds' arborescence on an undirected dataset.

    Returns:
        pooled_df: DataFrame with all features and 'graph_id' column
        feature_cols: List of feature column names
        scaler: None (scaling happens after the train/val split to avoid leakage)
        graphs: List of (gid, G, T) tuples
    """
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    entries = [
        validate_manifest_entry(e)
        for e in manifest
        if e.get("collection") == collection
    ]
    if not entries:
        raise RuntimeError(f"No manifest entries found for collection {collection}")

    dfs = []
    feature_cols = None
    graphs = []
    use_abs_diff_collection = False

    for entry in entries:
        gid = entry["graph_id"]

        # Load features from explicit paths in manifest
        node_path = entry["node_features_path"]
        edge_path = entry["edge_features_path"]

        if not os.path.exists(node_path):
            raise FileNotFoundError(f"Node features not found: {node_path}")
        if not os.path.exists(edge_path):
            raise FileNotFoundError(f"Edge features not found: {edge_path}")

        node_df = pd.read_csv(node_path)
        edge_df = pd.read_csv(edge_path)

        if "node" in node_df.columns:
            node_df = node_df.set_index("node")

        # Ensure edge index is tuple
        if "source" in edge_df.columns and "target" in edge_df.columns:
            edge_df["edge"] = list(zip(edge_df["source"], edge_df["target"]))
            edge_df = edge_df.set_index("edge")

        # Load graphs if available
        G = None
        T = None
        if entry.get("G_path") and os.path.exists(entry["G_path"]):
            G = load_graph(entry["G_path"])
        if entry.get("T_path") and os.path.exists(entry["T_path"]):
            T = load_graph(entry["T_path"])

        # In directed_mode, convert undirected G to bidirected and double edge rows
        if directed_mode and G is not None and not G.is_directed():
            # Reset index so source/target columns are accessible for mirroring
            edge_df_reset = (
                edge_df.reset_index(drop=True)
                if edge_df.index.name not in ("source", "target", None)
                else edge_df.copy()
            )
            # Rebuild with explicit source/target columns from index if needed
            if "source" not in edge_df_reset.columns:
                edge_df_reset["source"] = [i[0] for i in edge_df.index]
                edge_df_reset["target"] = [i[1] for i in edge_df.index]
            edge_df_reset.index = pd.Index(
                list(zip(edge_df_reset["source"], edge_df_reset["target"]))
            )
            mirror = _mirror_edge_df(edge_df_reset)
            edge_df = pd.concat([edge_df_reset, mirror])
            G = G.to_directed()  # adds both (u,v) and (v,u)
            use_abs_diff = False  # signed diffs give directional signal
        else:
            # Decide on abs diff for undirected graphs
            if G is not None:
                use_abs_diff = not G.is_directed()
            elif T is not None:
                use_abs_diff = not T.is_directed()
            else:
                use_abs_diff = False
        use_abs_diff_collection = use_abs_diff_collection or use_abs_diff

        # Prepare features
        processed_edge_df, cols = prepare_features(
            node_df, edge_df, use_abs_diff=use_abs_diff
        )

        # Verify feature dimension consistency
        if feature_cols is None:
            feature_cols = list(cols)
        else:
            if len(cols) != len(feature_cols):
                raise RuntimeError(
                    f"Feature dimension mismatch for {gid}: "
                    f"expected {len(feature_cols)} features, got {len(cols)}"
                )
            if list(cols) != feature_cols:
                raise RuntimeError(
                    f"Feature column names mismatch for {gid}: "
                    f"expected {feature_cols}, got {list(cols)}"
                )

        processed_edge_df = processed_edge_df.reset_index()
        processed_edge_df["graph_id"] = gid
        dfs.append(processed_edge_df)
        graphs.append((gid, G, T))

    # Pool all features unscaled; scaling happens after the split so validation rows do not leak into it.
    pooled = pd.concat(dfs, ignore_index=True)

    # Add edge tuple and set MultiIndex (graph_id, edge)
    pooled["edge"] = list(zip(pooled["source"], pooled["target"]))
    pooled = pooled.set_index(["graph_id", "edge"])

    return pooled, feature_cols, None, graphs


def sample_pooled_edges(pooled_df, graphs, feature_cols, fraction_pos=0.1, min_pos=10):
    """Sample positive/negative edges from pooled features.

    Edges are compared as frozensets {u, v} when either G or T is undirected (an undirected G
    paired with a directed T, as in microbiome and memetracker, must not depend on the
    orientation the features happened to use), and as (u, v) tuples when both are directed.

    Returns:
        X: Feature matrix
        y: Labels
        sampled_edges: Dict mapping gid -> list of (edge, label) tuples
    """

    def edge_to_key(u, v, is_undirected):
        """Convert edge to hashable key handling undirected semantics.

        - Undirected entity: use frozenset({u, v}) - direction doesn't matter
        - Directed entity:   use tuple (u, v) - direction matters
        """
        if is_undirected:
            return frozenset({u, v})
        else:
            return (u, v)

    sampled_edges_list = []
    labels = []
    sampled_edges = {gid: [] for gid, _, _ in graphs}

    for gid, G, T in graphs:
        if T is None or G is None:
            continue

        # Get edges that actually have features for this graph
        graph_edges = pooled_df.loc[gid].index.tolist()

        G_is_undirected = not G.is_directed()
        T_is_undirected = not T.is_directed()

        # A directed G with an undirected T would lose the direction the labels depend on.
        if not G_is_undirected and T_is_undirected:
            raise ValueError(
                f"Graph {gid}: G is directed but T is undirected, which is not supported. "
                "Give the tree directed edges or convert the graph to undirected."
            )

        use_frozenset = G_is_undirected or T_is_undirected

        # Convert all edge sets to same key representation for comparison
        feature_edges_keys = {
            frozenset({u, v}) if use_frozenset else (u, v) for u, v in graph_edges
        }

        T_edges_keys = {
            frozenset({u, v}) if use_frozenset else (u, v) for u, v in T.edges()
        }

        G_edges_keys = {
            frozenset({u, v}) if use_frozenset else (u, v) for u, v in G.edges()
        }

        # Find intersection: which graph/tree edges are in features?
        T_edges = T_edges_keys & feature_edges_keys
        G_edges = G_edges_keys & feature_edges_keys

        if not T_edges:
            raise RuntimeError(f"No positive edges found for graph {gid}")

        num_pos = int(max(min_pos, fraction_pos * len(T_edges)))
        pos_edges = random.sample(list(T_edges), min(num_pos, len(T_edges)))
        neg_candidates = list(G_edges - set(pos_edges))

        if not neg_candidates:
            raise RuntimeError(f"No negative edges found for graph {gid}")

        neg_edges = random.sample(
            neg_candidates, min(4 * len(pos_edges), len(neg_candidates))
        )

        # Build lookup map from edge key to feature tuple (for DataFrame indexing)
        feature_lookup = {}
        for actual_edge in graph_edges:
            key = (
                frozenset({actual_edge[0], actual_edge[1]})
                if use_frozenset
                else actual_edge
            )
            feature_lookup[key] = actual_edge

        # Build lookup for tree edges (for correct direction in positive.pkl)
        tree_lookup = {}
        if use_frozenset:
            for u, v in T.edges():
                key = frozenset({u, v})
                tree_lookup[key] = (u, v)

        # all_edges: Feature tuples for DataFrame lookup
        # all_edges_for_saving: Tuples with correct direction for saving to positive.pkl
        all_edges = []
        all_edges_for_saving = []
        edge_labels = []

        for edge_key in pos_edges:
            if edge_key not in feature_lookup:
                raise RuntimeError(f"Positive edge {edge_key} not found in features")

            # Use feature tuple for DataFrame lookup
            feature_tuple = feature_lookup[edge_key]
            all_edges.append(feature_tuple)

            # Use tree direction for saving (if undirected graph, otherwise same)
            if use_frozenset and edge_key in tree_lookup:
                tree_tuple = tree_lookup[edge_key]
                all_edges_for_saving.append(tree_tuple)
            else:
                all_edges_for_saving.append(feature_tuple)

            edge_labels.append(1)

        for edge_key in neg_edges:
            if edge_key not in feature_lookup:
                raise RuntimeError(f"Negative edge {edge_key} not found in features")

            feature_tuple = feature_lookup[edge_key]
            all_edges.append(feature_tuple)
            all_edges_for_saving.append(
                feature_tuple
            )  # Direction doesn't matter for negatives
            edge_labels.append(0)

        sampled_edges_list.extend([(gid, e) for e in all_edges])
        labels.extend(edge_labels)

        # Store edges with correct direction for saving to positive.pkl
        for e, label in zip(all_edges_for_saving, edge_labels):
            sampled_edges[gid].append((e, label))

    # Batch lookup for all sampled edges
    if sampled_edges_list:
        X = pooled_df.loc[sampled_edges_list, feature_cols].values
    else:
        X = np.empty((0, len(feature_cols)))

    return X, labels, sampled_edges


def train_manifest(
    manifest_path,
    collection,
    base_dir,
    model_type="xgb",
    n_trials=50,
    plateau_patience=20,
    plateau_min_trials=50,
    plateau_min_delta=0.0001,
    pruner_startup_trials=15,
    pruner_warmup_steps=10,
    early_stopping_rounds=50,
    n_jobs=-1,
    seed=42,
    optuna_enabled=True,
    hyperparams_config=None,
    directed_mode: bool = False,
    warm_start_from_tuned=True,
):
    """Train a pooled model on a manifest collection.

    Loads features from all graphs, pools them, samples edges, trains model,
    and saves all artifacts to base/collection/model/.

    Parameters
    ----------
    directed_mode : bool
        When True, undirected graphs are bidirected before feature computation:
        every edge (u,v) becomes two rows (u,v) and (v,u) with signed diff
        features, and a label of 1 is assigned only to the correctly directed
        edge in T.  The resulting score CSV has one score per directed edge so
        Edmonds' arborescence can recover both structure and direction in one
        step.  Does not affect already-directed graphs (e.g. wiki).
    warm_start_from_tuned : bool
        When True (default) and model_type == "xgb" and Optuna is enabled,
        looks for configs/xgb_{collection}.json (the
        previously-tuned hyperparameters for this collection) and seeds
        Optuna's search with them via study.enqueue_trial() so a modest
        --n-trials budget is guaranteed to see at least the known-good
        config. Silently skipped if no such config file exists.
    """
    pooled_df, feature_cols, _, graphs = pool_features_and_samples(
        manifest_path, collection, base_dir, directed_mode=directed_mode
    )

    # Determine if any graph in collection is undirected (after possible conversion)
    # When directed_mode=True, G has been converted to DiGraph inside pool_features_and_samples
    use_abs_diff_collection = (
        any(
            (G and not G.is_directed()) or (T and not T.is_directed())
            for _, G, T in graphs
        )
        and not directed_mode
    )
    X_raw, y, sampled_edges = sample_pooled_edges(pooled_df, graphs, feature_cols)

    if len(y) < 10:
        raise RuntimeError("Not enough sampled examples to train")

    # Split train/val (80/20) from sampled edges
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_raw, y, test_size=0.2, stratify=y, random_state=seed
    )

    # Convert to DataFrames with feature names to avoid sklearn warnings
    # about missing feature names at transform time
    X_train_df = pd.DataFrame(X_train_raw, columns=feature_cols)
    X_val_df = pd.DataFrame(X_val_raw, columns=feature_cols)

    # Fit scaler on TRAIN ONLY to avoid leakage
    scaler = StandardScaler()
    scaler.fit(X_train_df)

    # Transform splits
    X_train = scaler.transform(X_train_df)
    X_val = scaler.transform(X_val_df)

    test_edges, test_labels = build_inner_test_edges(pooled_df, graphs, sampled_edges)

    # Train model
    model_dir = get_model_dir(base_dir, collection)

    warm_start_params = None
    if warm_start_from_tuned and optuna_enabled and model_type == "xgb":
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        tuned_config_path = os.path.join(repo_root, "configs", f"xgb_{collection}.json")
        if os.path.exists(tuned_config_path):
            with open(tuned_config_path) as f:
                tuned_cfg = json.load(f)
            search_space_keys = {
                "learning_rate", "max_depth", "subsample", "colsample_bytree",
                "lambda", "alpha", "n_estimators",
            }
            warm_start_params = {k: tuned_cfg[k] for k in search_space_keys if k in tuned_cfg}

    model, best_params, metrics = train_model(
        X_train,
        y_train,
        X_val,
        y_val,
        model_dir,
        n_trials,
        plateau_patience=plateau_patience,
        plateau_min_trials=plateau_min_trials,
        plateau_min_delta=plateau_min_delta,
        pruner_startup_trials=pruner_startup_trials,
        pruner_warmup_steps=pruner_warmup_steps,
        early_stopping_rounds=early_stopping_rounds,
        n_jobs=n_jobs,
        seed=seed,
        optuna_enabled=optuna_enabled,
        hyperparams_config=hyperparams_config,
        warm_start_params=warm_start_params,
    )

    # Evaluate on test set (from unsampled edges) -- this is the "inner test":
    # edges from the few-shot TRAIN manifest's own graphs that weren't used
    # for train/val (most of each graph's edges, since sampling is few-shot).
    if test_edges:
        X_test_raw = pooled_df.loc[test_edges, feature_cols]
        X_test = scaler.transform(X_test_raw)
        y_test = test_labels
        metrics["test"] = evaluate(
            model, X_test, y_test,
            curve_path=os.path.join(model_dir, "roc_test_inner.png"),
            curve_title=f"ROC - inner test ({collection})",
        )
    else:
        metrics["test"] = {}

    # Save model artifacts with updated metrics
    meta = {
        "feature_cols": feature_cols,
        "n_samples": len(y),
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(test_labels),
        "collection": collection,
        "model_type": model_type,
        "use_abs_diff": use_abs_diff_collection,
        "directed_mode": directed_mode,
    }
    save_model_artifacts(model_dir, model, scaler, meta, model_type)

    # Re-save metrics with test set
    with open(os.path.join(model_dir, METRICS_FILE), "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Trained {model_type} model for collection {collection}")
    print(f"   Model saved to: {model_dir}")
    print(f"   Validation AUC: {metrics['validation']['roc_auc']:.4f}")
    print(f"   Test AUC: {metrics['test']['roc_auc']:.4f}")

    # Auto-score all graphs after training
    print("\nScoring all graphs in manifest...")
    score_written = score_manifest(manifest_path, collection, base_dir, model_dir)
    print(f"Scored {len(score_written)} graph(s)")
    for path in score_written:
        print(f"   {path}")

    return model, scaler, meta


def build_inner_test_edges(pooled_df, graphs, sampled_edges):
    """The "inner test" set: every feature edge of each train-manifest graph
    that was not sampled for train/val, labeled against T.

    Returns (test_edges, test_labels), test_edges as [(gid, edge_tuple)].
    """
    test_edges = []
    test_labels = []
    for gid, G, T in graphs:
        if T is None or G is None:
            continue

        # Same directedness rules as in sample_pooled_edges.
        G_is_undirected = not G.is_directed()
        T_is_undirected = not T.is_directed()

        if not G_is_undirected and T_is_undirected:
            raise ValueError(
                f"Graph {gid}: G is directed but T is undirected. Not supported."
            )

        use_frozenset = G_is_undirected or T_is_undirected

        # Take the edges from the feature table: G.edges() reports an arbitrary orientation
        # for undirected graphs, the feature table has the orientation the model saw.
        feature_edges = pooled_df.loc[gid].index.tolist()

        # Sampled edges, already in the feature table's orientation
        sampled_tuples = [e for e, _ in sampled_edges[gid]]

        # Find unused edges with proper direction handling
        if use_frozenset:
            # Build mappings: frozenset key -> original tuple
            feature_key_to_tuple = {frozenset({u, v}): (u, v) for u, v in feature_edges}
            sampled_keys = {frozenset({u, v}) for u, v in sampled_tuples}

            # Find unused feature keys
            unused_keys = feature_key_to_tuple.keys() - sampled_keys
            unused_edges = [feature_key_to_tuple[k] for k in unused_keys]
        else:
            # Both directed: direct set difference
            unused_edges = list(set(feature_edges) - set(sampled_tuples))

        if not unused_edges:
            continue

        # Build T edges as keys for labeling
        T_edges_keys = {
            frozenset({u, v}) if use_frozenset else (u, v) for u, v in T.edges()
        }

        # Assign labels: 1 if in tree, else 0
        labels_for_unused = []
        for e in unused_edges:
            key = frozenset({e[0], e[1]}) if use_frozenset else e
            labels_for_unused.append(1 if key in T_edges_keys else 0)

        test_edges.extend([(gid, e) for e in unused_edges])
        test_labels.extend(labels_for_unused)

    return test_edges, test_labels


def read_entry_features(entry):
    """Read a manifest entry's node/edge feature CSVs: node_df indexed by
    node, edge_df indexed by (source, target) tuples (source/target columns
    kept)."""
    node_df = pd.read_csv(entry["node_features_path"])
    edge_df = pd.read_csv(entry["edge_features_path"])
    if "node" in node_df.columns:
        node_df = node_df.set_index("node")
    if "source" in edge_df.columns and "target" in edge_df.columns:
        edge_df["edge"] = list(zip(edge_df["source"], edge_df["target"]))
        edge_df = edge_df.set_index("edge")
    return node_df, edge_df


def load_directed_tree_edges(t_path, gid):
    """T's edge set, for exact (source, target) labeling. T must be directed:
    a directed/undirected mismatch would silently mislabel every edge."""
    T = load_graph(t_path)
    if T.is_directed() is False:
        raise ValueError(
            f"Graph {gid}: ground-truth tree T is undirected, but T is "
            f"required to always be directed for edge-scoring evaluation. "
            f"Fix the tree, or don't pass it here."
        )
    return set(T.edges())


def outer_graph_metrics(labels, y_proba):
    """One graph's outer-test metrics (threshold 0.5 on y_proba)."""
    y_pred = (np.asarray(y_proba) >= 0.5).astype(int)
    graph_metrics = {
        "n_edges": int(len(labels)),
        "n_positive": int(sum(labels)),
        "accuracy": accuracy_score(labels, y_pred),
        "precision": precision_score(labels, y_pred, average="macro", zero_division=0),
        "recall": recall_score(labels, y_pred, average="macro", zero_division=0),
        "f1_score": f1_score(labels, y_pred, average="macro", zero_division=0),
    }
    if len(set(labels)) > 1:
        graph_metrics["roc_auc"] = roc_auc_score(labels, y_proba)
        graph_metrics["average_precision"] = average_precision_score(labels, y_proba)
        graph_metrics["pr_auc"] = _pr_auc(labels, y_proba)
    return graph_metrics


def write_outer_test_result(all_y, all_proba, per_graph, skipped, model_dir,
                            collection, label=None):
    """Pool the per-graph outer-test predictions into one metric set and save
    outer_test_metrics{_label}.json + roc_outer_test{_label}.png under
    model_dir. Returns the result dict."""
    suffix = f"_{label}" if label else ""
    pooled_pred = [1 if p >= 0.5 else 0 for p in all_proba]
    pooled_metrics = {
        "n_edges": int(len(all_y)),
        "n_positive": int(sum(all_y)),
        "accuracy": accuracy_score(all_y, pooled_pred),
        "precision": precision_score(all_y, pooled_pred, average="macro", zero_division=0),
        "recall": recall_score(all_y, pooled_pred, average="macro", zero_division=0),
        "f1_score": f1_score(all_y, pooled_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(all_y, all_proba),
        "average_precision": average_precision_score(all_y, all_proba),
        "pr_auc": _pr_auc(all_y, all_proba),
    }
    save_roc_curve(
        all_y, all_proba,
        os.path.join(model_dir, f"roc_outer_test{suffix}.png"),
        title=f"Outer-test ROC - {collection}{suffix}",
    )

    result = {"pooled": pooled_metrics, "per_graph": per_graph, "skipped": skipped}
    out_path = os.path.join(model_dir, f"outer_test_metrics{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[evaluate_manifest] {collection}{suffix}: outer-test AUC={pooled_metrics['roc_auc']:.4f} "
          f"avg_precision={pooled_metrics['average_precision']:.4f} pr_auc={pooled_metrics['pr_auc']:.4f} "
          f"n_edges={pooled_metrics['n_edges']} "
          f"(skipped {len(skipped)} entries without T_path/features) -> {out_path}")
    return result


def evaluate_manifest(manifest_path, collection, model_dir, label=None):
    """Evaluate a trained model against ALL edges of every graph in a
    manifest that has ground truth (T_path) -- the "outer test" set, as
    opposed to train_manifest's internal "test" metric which only covers
    edges left unsampled within the TRAIN manifest's own few-shot graphs.

    Every edge in each graph's edge_features CSV is labeled against T (T is
    required to be directed -- raises if it isn't, since a directed/
    undirected mismatch here would silently mislabel every edge) and scored,
    pooling predictions across all graphs in the manifest for one aggregate
    metric set, plus a per-graph breakdown. Reads directed_mode/use_abs_diff/
    feature_cols from the trained model's own meta.json (same mechanism
    score_graph uses), so this works correctly regardless of whether the
    underlying graphs are directed (wiki) or were bidirected at train time
    (microbiome/memetracker with --directed-mode).

    Saves outer_test_metrics{_label}.json and roc_outer_test{_label}.png
    under model_dir. Returns the metrics dict.
    """
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    entries = [e for e in manifest if e.get("collection") == collection]
    if not entries:
        raise RuntimeError(f"No manifest entries found for collection {collection}")

    model, scaler, meta = load_model_artifacts(model_dir)

    all_y, all_proba = [], []
    per_graph = {}
    skipped = []

    for entry in entries:
        gid = entry["graph_id"]
        node_path = entry.get("node_features_path")
        edge_path = entry.get("edge_features_path")
        t_path = entry.get("T_path")
        if not (t_path and os.path.exists(t_path) and node_path and edge_path
                and os.path.exists(node_path) and os.path.exists(edge_path)):
            skipped.append(gid)
            continue

        T_edges_keys = load_directed_tree_edges(t_path, gid)
        node_df, edge_df = read_entry_features(entry)

        # Every row is a genuinely directed (source, target) pair (either the
        # graph was already directed, or it is bidirected in directed_mode) --
        # exact-tuple match against T's directed edges, never frozenset.
        processed_edge_df, cols = build_scoring_frame(meta, node_df, edge_df)
        labels = [1 if e in T_edges_keys else 0 for e in processed_edge_df.index.tolist()]

        X_df = processed_edge_df[cols]
        X = scaler.transform(X_df) if scaler is not None else X_df.values

        y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)
        per_graph[gid] = outer_graph_metrics(labels, y_proba)

        all_y.extend(labels)
        all_proba.extend(y_proba.tolist())

    if not all_y:
        print(f"[evaluate_manifest] No entries with T_path + features found for "
              f"{collection} in {manifest_path}; skipping outer-test evaluation.")
        return None

    return write_outer_test_result(all_y, all_proba, per_graph, skipped, model_dir,
                                   collection, label=label)


def score_manifest(manifest_path, collection, base_dir, model_dir, evaluate_output=True):
    """Score all graphs in a manifest using a trained model.

    Features are always loaded from the paths recorded in each manifest entry
    (node_features_path / edge_features_path) so that base_dir can be a
    different directory than where the features were originally computed.
    Score CSVs are written under base_dir/{collection}/{gid}/scores/.

    Loads model once, then scores each graph individually.

    If evaluate_output is True (default) and any entry has a T_path, also
    runs evaluate_manifest() against this same manifest right after scoring,
    so a single scoring command produces both score CSVs and evaluation
    metrics/ROC curve -- see evaluate_manifest() to run evaluation alone.

    Returns list of output paths.
    """
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    entries = [
        validate_manifest_entry(e)
        for e in manifest
        if e.get("collection") == collection
    ]
    if not entries:
        raise RuntimeError(f"No manifest entries found for collection {collection}")

    # Load model once
    model, scaler, meta = load_model_artifacts(model_dir)

    written = []
    for entry in entries:
        gid = entry["graph_id"]

        # Load features from the paths stored in the manifest entry so this
        # works correctly when base_dir differs from the original output dir.
        node_path = entry.get("node_features_path")
        edge_path = entry.get("edge_features_path")
        if (
            node_path
            and edge_path
            and os.path.exists(node_path)
            and os.path.exists(edge_path)
        ):
            node_df, edge_df = read_entry_features(entry)
        else:
            # Fallback: derive from base_dir (original behaviour)
            graph_dir = get_graph_dir(base_dir, collection, gid)
            node_df, edge_df = load_features(graph_dir)

        # Score; the output always goes under base_dir
        graph_dir = get_graph_dir(base_dir, collection, gid)
        out_path = os.path.join(graph_dir, SCORE_DIR, EDGE_SCORES)
        score_graph(model, scaler, meta, node_df, edge_df, out_path)
        written.append(out_path)

        # Update manifest with score_path
        entry["score_path"] = out_path
        print(f"Scored {gid} -> {out_path}")

    # Write updated manifest
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    if evaluate_output and any(e.get("T_path") for e in entries):
        label = os.path.splitext(os.path.basename(manifest_path))[0]
        evaluate_manifest(manifest_path, collection, model_dir, label=label)

    return written


def main():
    args = parse_args()

    # Initialize random seeds for reproducibility across entire script
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Convert single-graph mode to manifest-of-1
    if not args.manifest:
        if not args.gid:
            raise ValueError("Either --manifest or --gid is required")

        # Derive paths from convention
        if not args.collection:
            args.collection = ""  # empty collection: <output-dir>/<gid>/...

        graph_dir = get_graph_dir(args.output_dir, args.collection, args.gid)

        # Build temp manifest entry
        manifest_entry = {
            "graph_id": args.gid,
            "collection": args.collection,
            "node_features_path": os.path.join(graph_dir, FEATURE_DIR, NODE_FEATURES),
            "edge_features_path": os.path.join(graph_dir, FEATURE_DIR, EDGE_FEATURES),
        }

        # Add G_path and T_path if provided
        if args.graph:
            manifest_entry["G_path"] = args.graph
        if args.tree:
            manifest_entry["T_path"] = args.tree

        # Create temp manifest file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([manifest_entry], f, indent=2)
            temp_manifest = f.name

        args.manifest = temp_manifest

    # At this point, args.manifest is always set

    # Manifest scoring
    if args.score_only:
        if not args.collection:
            raise ValueError("--collection required for scoring")
        if not args.model_path:
            # Default to base/collection/model
            args.model_path = get_model_dir(args.output_dir, args.collection)

        written = score_manifest(
            args.manifest, args.collection, args.output_dir, args.model_path
        )
        print(f"Scored {len(written)} graphs from manifest")
        return

    # Manifest training
    if not args.collection:
        raise ValueError("--collection required for training")

    train_manifest(
        args.manifest,
        args.collection,
        args.output_dir,
        n_trials=args.n_trials,
        plateau_patience=args.plateau_patience,
        plateau_min_trials=args.plateau_min_trials,
        plateau_min_delta=args.plateau_min_delta,
        pruner_startup_trials=args.pruner_startup_trials,
        pruner_warmup_steps=args.pruner_warmup_steps,
        early_stopping_rounds=args.early_stopping_rounds,
        n_jobs=args.n_jobs,
        seed=args.seed,
        optuna_enabled=args.optuna_enabled,
        hyperparams_config=args.hyperparams_config,
        directed_mode=args.directed_mode,
    )


if __name__ == "__main__":
    main()
