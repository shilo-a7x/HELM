"""
GNN edge scoring (GINEConv) -- the graph-neural-network counterpart of
xgb.py (XGBoost).

Only the model differs. Everything else is imported from xgb.py: the
manifest pooling, the few-shot positive/negative sampling (same --seed -> the
exact same sampled edges as the XGBoost run), the stratified 80/20
train/validation split, the inner-test set, the edge-feature scaler, the
scored edge rows, the metric definitions and the outer-test evaluation. CLI
flags and the output layout mirror xgb.py:

  {output_dir}/{collection}/model/
      gnn_model.pt               state_dict + architecture hyperparameters
      scaler.pkl                 edge-feature StandardScaler (fit on train rows)
      node_scaler.pkl            node-feature StandardScaler (fit on the
                                 train-manifest graphs' nodes; no labels involved)
      meta.json                  feature columns, split sizes, directed_mode, ...
      best_params.json, optuna_results.json
      metrics.json               train / validation / test (= inner test)
      roc_{train,validation,test_inner}.png
      outer_test_metrics_<manifest>.json, roc_outer_test_<manifest>.png
      training_history.json      per-epoch train loss / validation AUC (final fit)
  {output_dir}/{collection}/{gid}/scores/edge_scores.csv

Scoring writes score_path into the manifest it is given, so give the GNN its own copy
of the manifest (``helm run`` does this) rather than one shared with the XGBoost scorer.

Model: node features -> Linear, edge features -> Linear, then L x [GINEConv
-> LayerNorm (-> BatchNorm with --batch-norm) -> ReLU -> dropout]. An edge
(u, v) is scored by an MLP on [h_u || h_v || e_uv], where e_uv is the edge's
own encoded feature vector (--head endpoints drops e_uv, for ablation).
Message passing follows edge direction: directed graphs (wiki) are used as
they are; undirected graphs in --directed-mode are bidirected, so messages
flow both ways.

Usage (repo root):
  # train on the train manifest (also scores it + outer-test on it)
  python -m helm.scoring.gnn --manifest M_train.json \\
      --collection microbiome --output-dir runs/gnn --directed-mode
  # score the held-out test manifest (+ outer-test on it)
  python -m helm.scoring.gnn --manifest M_test.json \\
      --collection microbiome --output-dir runs/gnn --score-only
"""

import argparse
import copy
import json
import os
import pickle
import random
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GINEConv
    from torch_geometric.utils import dropout_edge
except ImportError as _e:
    raise ImportError(
        f"GNN dependencies not available ({_e}).\n"
        "Install torch + torch-geometric (see requirements-gnn.txt).\n"
        "Skip this module entirely if you only need the XGBoost pipeline."
    )

import optuna

from helm.scoring.xgb import (
    BEST_PARAMS_FILE,
    EDGE_FEATURES,
    EDGE_SCORES,
    FEATURE_DIR,
    META_FILE,
    METRICS_FILE,
    NODE_FEATURES,
    SCALER_FILE,
    SCORE_DIR,
    build_inner_test_edges,
    build_scoring_frame,
    get_graph_dir,
    get_model_dir,
    load_directed_tree_edges,
    metrics_from_proba,
    outer_graph_metrics,
    pool_features_and_samples,
    read_entry_features,
    sample_pooled_edges,
    validate_manifest_entry,
    write_outer_test_result,
)

MODEL_FILE = "gnn_model.pt"
NODE_SCALER_FILE = "node_scaler.pkl"
OPTUNA_RESULTS_FILE = "optuna_results.json"
HISTORY_FILE = "training_history.json"

# Optuna search space for the GNN hyperparameters.
DEFAULT_HIDDEN_DIM_RANGE = [64, 96, 128, 192]
NUM_LAYERS_RANGE = (2, 3)
DROPOUT_RANGE = (0.2, 0.6)
EDGE_DROPOUT_RANGE = (0.2, 0.6)
LEARNING_RATE_RANGE = (1e-4, 5e-3)
WEIGHT_DECAY_RANGE = (1e-4, 1e-2)
REQUIRED_PARAMS = {
    "hidden_dim",
    "num_layers",
    "dropout_rate",
    "edge_dropout_rate",
    "learning_rate",
    "weight_decay",
}

HEADS = ("endpoints+edge", "endpoints")
GRADIENT_CLIP = 1.0
CHUNK_SIZE = 256  # target edges per optimisation step (all from one graph)
SCORE_CHUNK = 65536  # target edges per scoring call at inference


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------
# Model
# ------------------------


class GINEEdgeScorerModel(nn.Module):
    """GINEConv message passing over the whole graph; each target edge (u, v)
    is scored from [h_u || h_v || e_uv]."""

    def __init__(
        self,
        node_feature_dim,
        edge_feature_dim,
        hidden_dim=128,
        num_layers=2,
        dropout_rate=0.2,
        edge_dropout_rate=0.0,
        batch_norm=False,
        head="endpoints+edge",
    ):
        super().__init__()
        if head not in HEADS:
            raise ValueError(f"head must be one of {HEADS}, got {head!r}")
        self.edge_dropout_rate = edge_dropout_rate
        self.head = head
        self.node_encoder = nn.Linear(node_feature_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_feature_dim, hidden_dim)

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList() if batch_norm else None
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.conv_layers.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.norm_layers.append(nn.LayerNorm(hidden_dim))
            if batch_norm:
                self.batch_norm_layers.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = nn.Dropout(dropout_rate)

        self.score_head = nn.Sequential(
            nn.Linear((3 if head == "endpoints+edge" else 2) * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
        )

    def embed(self, x, edge_index, edge_attr):
        """Node embeddings after message passing, plus the encoded edge
        features of every edge (before edge dropout)."""
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)
        mp_index, mp_attr = edge_index, e
        if self.training and self.edge_dropout_rate > 0:
            mp_index, edge_mask = dropout_edge(
                edge_index, p=self.edge_dropout_rate, training=True
            )
            mp_attr = e[edge_mask]
        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.norm_layers)):
            h = conv(h, mp_index, mp_attr)
            h = norm(h)
            if self.batch_norm_layers is not None:
                h = self.batch_norm_layers[i](h)
            h = F.relu(h)
            h = self.dropout(h)
        return h, e

    def score(self, h, e, edge_index, target_ids):
        """Logits for the edges at positions target_ids of edge_index."""
        src = edge_index[0, target_ids]
        dst = edge_index[1, target_ids]
        parts = [h[src], h[dst]]
        if self.head == "endpoints+edge":
            parts.append(e[target_ids])
        z = torch.cat(parts, dim=-1)
        return self.score_head(z).squeeze(-1)

    def forward(self, x, edge_index, edge_attr, target_ids):
        h, e = self.embed(x, edge_index, edge_attr)
        return self.score(h, e, edge_index, target_ids)


def build_model(params, node_feature_dim, edge_feature_dim, batch_norm, head):
    return GINEEdgeScorerModel(
        node_feature_dim,
        edge_feature_dim,
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        dropout_rate=params["dropout_rate"],
        edge_dropout_rate=params["edge_dropout_rate"],
        batch_norm=batch_norm,
        head=head,
    )


# ------------------------
# Graph tensors
# ------------------------


def node_feature_matrix(node_df, node_feature_cols, gid):
    missing = [c for c in node_feature_cols if c not in node_df.columns]
    if missing:
        raise RuntimeError(f"{gid}: node features missing columns {missing}")
    values = node_df[node_feature_cols].to_numpy(dtype=np.float64)
    if np.isnan(values).any():
        raise ValueError(f"{gid}: node features contain NaN")
    return values


def build_graph_tensors(
    gid, node_df, processed_edge_df, feature_cols, node_feature_cols,
    scaler, node_scaler, device,
):
    """Tensors for one graph. The message-passing edges are exactly the rows
    the model scores (processed_edge_df, from build_scoring_frame): directed
    graphs keep their direction, bidirected graphs carry both directions.

    Returns dict(x, edge_index, edge_attr on device; edge_pos: {(u, v): row}).
    """
    node_ids = node_df.index.tolist()
    node_to_idx = {n: i for i, n in enumerate(node_ids)}
    sources = processed_edge_df["source"].tolist()
    targets = processed_edge_df["target"].tolist()
    unknown = {n for n in sources + targets if n not in node_to_idx}
    if unknown:
        raise ValueError(
            f"{gid}: {len(unknown)} edge endpoints missing from node features, "
            f"e.g. {list(unknown)[:5]}"
        )

    x = node_scaler.transform(node_feature_matrix(node_df, node_feature_cols, gid))
    edge_attr = scaler.transform(processed_edge_df[feature_cols])
    if np.isnan(edge_attr).any():
        raise ValueError(f"{gid}: edge features contain NaN")
    edge_index = np.array(
        [[node_to_idx[u] for u in sources], [node_to_idx[v] for v in targets]],
        dtype=np.int64,
    )
    edge_pos = {e: i for i, e in enumerate(zip(sources, targets))}
    if len(edge_pos) != len(sources):
        raise ValueError(f"{gid}: duplicate (source, target) rows")
    return {
        "x": torch.tensor(x, dtype=torch.float32, device=device),
        "edge_index": torch.tensor(edge_index, dtype=torch.long, device=device),
        "edge_attr": torch.tensor(edge_attr, dtype=torch.float32, device=device),
        "edge_pos": edge_pos,
    }


def group_targets(edge_list, graph_tensors):
    """[(gid, edge, label)] -> {gid: (row ids tensor, labels tensor, order)}
    where order is the position of each target in edge_list."""
    grouped = {}
    for i, (gid, edge, label) in enumerate(edge_list):
        rows, labels, order = grouped.setdefault(gid, ([], [], []))
        if edge not in graph_tensors[gid]["edge_pos"]:
            raise KeyError(f"{gid}: target edge {edge} is not a scored row")
        rows.append(graph_tensors[gid]["edge_pos"][edge])
        labels.append(float(label))
        order.append(i)
    device = next(iter(graph_tensors.values()))["x"].device
    return {
        gid: (
            torch.tensor(rows, dtype=torch.long, device=device),
            torch.tensor(labels, dtype=torch.float32, device=device),
            order,
        )
        for gid, (rows, labels, order) in grouped.items()
    }


@torch.no_grad()
def predict_proba(model, graph_tensors, grouped):
    """Probabilities for grouped targets, returned in the original edge_list
    order. One message-passing pass per graph."""
    model.eval()
    n = sum(len(order) for _, _, order in grouped.values())
    out = np.empty(n, dtype=np.float64)
    for gid, (rows, _, order) in grouped.items():
        g = graph_tensors[gid]
        h, e = model.embed(g["x"], g["edge_index"], g["edge_attr"])
        out[order] = np.concatenate([
            torch.sigmoid(model.score(h, e, g["edge_index"], rows[s : s + SCORE_CHUNK])).cpu().numpy()
            for s in range(0, len(rows), SCORE_CHUNK)
        ])
    return out


# ------------------------
# Training
# ------------------------


def fit_model(
    params, graph_tensors, train_grouped, val_grouped, y_val,
    node_feature_dim, edge_feature_dim, pos_weight, batch_norm, head,
    max_epochs, patience, seed, trial=None, pruner_warmup_steps=0,
):
    """Adam + BCE (pos_weight = n_neg/n_pos on train) with gradient clipping;
    early stopping on validation AUC, keeping the best epoch's weights.
    Each step is one chunk of <= CHUNK_SIZE target edges from a single graph.
    Returns (model at best epoch, best validation AUC, history)."""
    set_seed(seed)
    device = next(iter(graph_tensors.values()))["x"].device
    model = build_model(params, node_feature_dim, edge_feature_dim, batch_norm, head).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"]
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    steps = []
    for gid, (rows, labels, _) in train_grouped.items():
        for s in range(0, len(rows), CHUNK_SIZE):
            steps.append((gid, rows[s : s + CHUNK_SIZE], labels[s : s + CHUNK_SIZE]))
    rng = random.Random(seed)

    best_auc, best_state, best_epoch = -1.0, None, -1
    history = []
    for epoch in range(max_epochs):
        model.train()
        rng.shuffle(steps)
        total_loss = 0.0
        for gid, rows, labels in steps:
            g = graph_tensors[gid]
            optimizer.zero_grad()
            logits = model(g["x"], g["edge_index"], g["edge_attr"], rows)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            total_loss += loss.item() * len(rows)

        val_auc = roc_auc_score(y_val, predict_proba(model, graph_tensors, val_grouped))
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / max(1, sum(len(r) for _, r, _ in steps)),
            "val_auc": val_auc,
        })
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            best_state = copy.deepcopy(model.state_dict())

        if trial is not None:
            trial.report(val_auc, epoch)
            if epoch >= pruner_warmup_steps and trial.should_prune():
                raise optuna.TrialPruned()
        if epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_auc, history


def _load_params(hyperparams_config):
    with open(hyperparams_config) as f:
        params = json.load(f)
    missing = REQUIRED_PARAMS - set(params)
    if missing:
        raise ValueError(f"{hyperparams_config} missing hyperparameters: {sorted(missing)}")
    return params


def train_manifest(
    manifest_path,
    collection,
    base_dir,
    device,
    n_trials=30,
    max_epochs=200,
    patience=20,
    plateau_patience=20,
    plateau_min_trials=50,
    plateau_min_delta=0.0001,
    pruner_startup_trials=15,
    pruner_warmup_steps=10,
    hidden_dim_range=None,
    batch_norm=False,
    head="endpoints+edge",
    seed=42,
    optuna_enabled=True,
    hyperparams_config=None,
    directed_mode=False,
):
    """Train a pooled GNN on a manifest collection, mirroring
    xgb.train_manifest: same samples, split, scaler, inner test and
    saved artifacts; then scores (and outer-tests) the same manifest."""
    hidden_dim_range = hidden_dim_range or DEFAULT_HIDDEN_DIM_RANGE
    model_dir = get_model_dir(base_dir, collection)
    if os.path.exists(os.path.join(model_dir, MODEL_FILE)):
        raise FileExistsError(
            f"{model_dir} already holds a trained GNN; refusing to overwrite. "
            f"Use a new --output-dir."
        )

    # Same seeding as xgb.main() so sampling matches the XGBoost run.
    random.seed(seed)
    np.random.seed(seed)

    pooled_df, feature_cols, _, graphs = pool_features_and_samples(
        manifest_path, collection, base_dir, directed_mode=directed_mode
    )
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

    # Same split call as xgb.train_manifest. sample_pooled_edges
    # builds X/y graph by graph in sampled_edges order, so row i <-> edge i.
    sampled_list = [
        (gid, e, lbl) for gid, edges in sampled_edges.items() for e, lbl in edges
    ]
    if [lbl for _, _, lbl in sampled_list] != list(y):
        raise RuntimeError("sampled edge order does not match the sampled labels")
    idx = np.arange(len(y))
    X_train_raw, X_val_raw, y_train, y_val, idx_train, idx_val = train_test_split(
        X_raw, y, idx, test_size=0.2, stratify=y, random_state=seed
    )
    train_edges = [sampled_list[i] for i in idx_train]
    val_edges = [sampled_list[i] for i in idx_val]

    # Fit the edge-feature scaler on TRAIN rows only (identical to XGBoost's).
    scaler = StandardScaler()
    scaler.fit(pd.DataFrame(X_train_raw, columns=feature_cols))

    test_pairs, test_labels = build_inner_test_edges(pooled_df, graphs, sampled_edges)
    test_edges = [(gid, e, lbl) for (gid, e), lbl in zip(test_pairs, test_labels)]
    del pooled_df

    meta = {
        "feature_cols": feature_cols,
        "n_samples": len(y),
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(test_labels),
        "collection": collection,
        "model_type": "gnn",
        "use_abs_diff": use_abs_diff_collection,
        "directed_mode": directed_mode,
        "batch_norm": batch_norm,
        "head": head,
    }

    # Node features: one scaler fitted on all nodes of the train-manifest graphs.
    with open(manifest_path) as f:
        entries = {
            e["graph_id"]: validate_manifest_entry(e)
            for e in json.load(f)
            if e.get("collection") == collection
        }
    features = {gid: read_entry_features(entries[gid]) for gid, _, _ in graphs}
    node_feature_cols = list(next(iter(features.values()))[0].columns)
    node_scaler = StandardScaler()
    node_scaler.fit(np.vstack([
        node_feature_matrix(node_df, node_feature_cols, gid)
        for gid, (node_df, _) in features.items()
    ]))
    meta["node_feature_cols"] = node_feature_cols

    graph_tensors = {}
    for gid, (node_df, edge_df) in features.items():
        processed, _ = build_scoring_frame(meta, node_df, edge_df)
        graph_tensors[gid] = build_graph_tensors(
            gid, node_df, processed, feature_cols, node_feature_cols,
            scaler, node_scaler, device,
        )
    del features

    train_grouped = group_targets(train_edges, graph_tensors)
    val_grouped = group_targets(val_edges, graph_tensors)
    test_grouped = group_targets(test_edges, graph_tensors)
    y_val_ordered = np.array([lbl for _, _, lbl in val_edges])

    n_pos = int(sum(y_train))
    pos_weight = float((len(y_train) - n_pos) / max(1, n_pos))
    node_feature_dim = len(node_feature_cols)
    edge_feature_dim = len(feature_cols)
    meta["pos_weight"] = pos_weight
    print(
        f"GNN {collection}: {len(graph_tensors)} graphs, train/val/test = "
        f"{len(train_edges)}/{len(val_edges)}/{len(test_edges)} edges, "
        f"node dim {node_feature_dim}, edge dim {edge_feature_dim}, "
        f"pos_weight {pos_weight:.2f}, device {device}"
    )

    fit_kwargs = dict(
        graph_tensors=graph_tensors,
        train_grouped=train_grouped,
        val_grouped=val_grouped,
        y_val=y_val_ordered,
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        pos_weight=pos_weight,
        batch_norm=batch_norm,
        head=head,
        max_epochs=max_epochs,
        patience=patience,
        seed=seed,
    )

    os.makedirs(model_dir, exist_ok=True)
    if optuna_enabled:

        def objective(trial):
            params = {
                "hidden_dim": trial.suggest_categorical("hidden_dim", hidden_dim_range),
                "num_layers": trial.suggest_int("num_layers", *NUM_LAYERS_RANGE),
                "dropout_rate": trial.suggest_float("dropout_rate", *DROPOUT_RANGE),
                "edge_dropout_rate": trial.suggest_float("edge_dropout_rate", *EDGE_DROPOUT_RANGE),
                "learning_rate": trial.suggest_float("learning_rate", *LEARNING_RATE_RANGE, log=True),
                "weight_decay": trial.suggest_float("weight_decay", *WEIGHT_DECAY_RANGE, log=True),
            }
            t0 = time.time()
            _, best_auc, history = fit_model(
                params, trial=trial, pruner_warmup_steps=pruner_warmup_steps, **fit_kwargs
            )
            print(
                f"   trial {trial.number}: val AUC {best_auc:.4f} "
                f"({len(history)} epochs, {time.time() - t0:.0f}s)"
            )
            return best_auc

        class PlateauCallback:
            """Stop the study after `patience` trials without improvement
            (same rule as xgb.py)."""

            def __init__(self, patience, min_trials, min_delta):
                self.patience = patience
                self.min_trials = min_trials
                self.min_delta = min_delta
                self.best_value = None
                self.trials_without_improvement = 0

            def __call__(self, study, trial):
                if len(study.trials) < self.min_trials:
                    return
                current_best = study.best_value
                if self.best_value is None:
                    self.best_value = current_best
                    return
                if current_best > self.best_value + self.min_delta:
                    self.best_value = current_best
                    self.trials_without_improvement = 0
                else:
                    self.trials_without_improvement += 1
                if self.trials_without_improvement >= self.patience:
                    print(f"   Early stopping: no improvement for {self.patience} trials")
                    study.stop()

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=pruner_startup_trials,
                n_warmup_steps=pruner_warmup_steps,
            ),
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=1,
            callbacks=[PlateauCallback(plateau_patience, plateau_min_trials, plateau_min_delta)],
        )
        best_params = dict(study.best_params)
        n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        print(f"   Completed {len(study.trials)} trials ({n_pruned} pruned); "
              f"best val AUC {study.best_value:.4f}")
        with open(os.path.join(model_dir, OPTUNA_RESULTS_FILE), "w") as f:
            json.dump(
                {
                    "best_trial": study.best_trial.number,
                    "best_value": float(study.best_value),
                    "best_params": best_params,
                    "n_trials": len(study.trials),
                    "n_pruned": n_pruned,
                    "trials": [
                        {"number": t.number, "state": t.state.name, "value": t.value,
                         "params": t.params}
                        for t in study.trials
                    ],
                },
                f,
                indent=2,
            )
    else:
        if hyperparams_config is None:
            raise ValueError("Optuna disabled but no --hyperparams-config was provided")
        best_params = _load_params(hyperparams_config)

    # Final fit with the best hyperparameters.
    model, _, history = fit_model(best_params, **fit_kwargs)
    with open(os.path.join(model_dir, HISTORY_FILE), "w") as f:
        json.dump(history, f, indent=2)

    y_train_ordered = np.array([lbl for _, _, lbl in train_edges])
    metrics = {
        "train": metrics_from_proba(
            y_train_ordered, predict_proba(model, graph_tensors, train_grouped),
            curve_path=os.path.join(model_dir, "roc_train.png"),
            curve_title="ROC - train",
        ),
        "validation": metrics_from_proba(
            y_val_ordered, predict_proba(model, graph_tensors, val_grouped),
            curve_path=os.path.join(model_dir, "roc_validation.png"),
            curve_title="ROC - validation",
        ),
    }
    if test_edges:
        metrics["test"] = metrics_from_proba(
            np.array(test_labels), predict_proba(model, graph_tensors, test_grouped),
            curve_path=os.path.join(model_dir, "roc_test_inner.png"),
            curve_title=f"ROC - inner test ({collection})",
        )
    else:
        metrics["test"] = {}

    save_model_artifacts(model_dir, model, best_params, scaler, node_scaler, meta,
                         node_feature_dim, edge_feature_dim)
    with open(os.path.join(model_dir, BEST_PARAMS_FILE), "w") as f:
        json.dump(best_params, f, indent=4)
    with open(os.path.join(model_dir, METRICS_FILE), "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Trained GNN for collection {collection}")
    print(f"   Model saved to: {model_dir}")
    print(f"   Validation AUC: {metrics['validation']['roc_auc']:.4f}")
    if metrics["test"]:
        print(f"   Test AUC: {metrics['test']['roc_auc']:.4f}")

    del graph_tensors
    print("\nScoring all graphs in manifest...")
    written = score_manifest(manifest_path, collection, base_dir, model_dir, device)
    print(f"Scored {len(written)} graph(s)")
    return model_dir


# ------------------------
# Artifacts
# ------------------------


def save_model_artifacts(model_dir, model, params, scaler, node_scaler, meta,
                         node_feature_dim, edge_feature_dim):
    os.makedirs(model_dir, exist_ok=True)
    torch.save(
        {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "params": params,
            "node_feature_dim": node_feature_dim,
            "edge_feature_dim": edge_feature_dim,
            "batch_norm": meta["batch_norm"],
            "head": meta["head"],
        },
        os.path.join(model_dir, MODEL_FILE),
    )
    with open(os.path.join(model_dir, SCALER_FILE), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(model_dir, NODE_SCALER_FILE), "wb") as f:
        pickle.dump(node_scaler, f)
    with open(os.path.join(model_dir, META_FILE), "w") as f:
        json.dump(meta, f, indent=2)


def load_model_artifacts(model_dir, device):
    ckpt = torch.load(os.path.join(model_dir, MODEL_FILE), map_location=device)
    model = build_model(
        ckpt["params"], ckpt["node_feature_dim"], ckpt["edge_feature_dim"],
        ckpt["batch_norm"], ckpt["head"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with open(os.path.join(model_dir, SCALER_FILE), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(model_dir, NODE_SCALER_FILE), "rb") as f:
        node_scaler = pickle.load(f)
    with open(os.path.join(model_dir, META_FILE)) as f:
        meta = json.load(f)
    return model, scaler, node_scaler, meta


# ------------------------
# Scoring
# ------------------------


@torch.no_grad()
def score_graph(model, scaler, node_scaler, meta, gid, node_df, edge_df, out_path, device):
    """Score every row build_scoring_frame produces (same rows as the XGBoost
    score CSV) and write [source, target, score] to out_path."""
    processed, feature_cols = build_scoring_frame(meta, node_df, edge_df)
    g = build_graph_tensors(
        gid, node_df, processed, feature_cols, meta["node_feature_cols"],
        scaler, node_scaler, device,
    )
    model.eval()
    h, e = model.embed(g["x"], g["edge_index"], g["edge_attr"])
    all_ids = torch.arange(g["edge_index"].shape[1], device=device)
    scores = np.concatenate([
        torch.sigmoid(model.score(h, e, g["edge_index"], all_ids[s : s + SCORE_CHUNK])).cpu().numpy()
        for s in range(0, len(all_ids), SCORE_CHUNK)
    ])
    df = pd.DataFrame(
        {
            "source": processed["source"].values,
            "target": processed["target"].values,
            "score": scores,
        }
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def score_manifest(manifest_path, collection, base_dir, model_dir, device, evaluate_output=True):
    """Score all graphs of a manifest (scores CSV per graph, score_path
    recorded in the manifest) and, when entries have T_path, write the
    outer-test metrics exactly as xgb.evaluate_manifest does."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    entries = [
        validate_manifest_entry(e) for e in manifest if e.get("collection") == collection
    ]
    if not entries:
        raise RuntimeError(f"No manifest entries found for collection {collection}")

    model, scaler, node_scaler, meta = load_model_artifacts(model_dir, device)

    written = []
    all_y, all_proba, per_graph, skipped = [], [], {}, []
    for entry in entries:
        gid = entry["graph_id"]
        node_df, edge_df = read_entry_features(entry)
        out_path = os.path.join(get_graph_dir(base_dir, collection, gid), SCORE_DIR, EDGE_SCORES)
        df = score_graph(model, scaler, node_scaler, meta, gid, node_df, edge_df, out_path, device)
        written.append(out_path)
        entry["score_path"] = out_path
        print(f"Scored {gid} -> {out_path}")

        t_path = entry.get("T_path")
        if evaluate_output:
            if t_path and os.path.exists(t_path):
                T_edges_keys = load_directed_tree_edges(t_path, gid)
                labels = [
                    1 if e in T_edges_keys else 0
                    for e in zip(df["source"].tolist(), df["target"].tolist())
                ]
                y_proba = df["score"].to_numpy()
                per_graph[gid] = outer_graph_metrics(labels, y_proba)
                all_y.extend(labels)
                all_proba.extend(y_proba.tolist())
            else:
                skipped.append(gid)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    if evaluate_output and all_y:
        label = os.path.splitext(os.path.basename(manifest_path))[0]
        write_outer_test_result(all_y, all_proba, per_graph, skipped, model_dir,
                                collection, label=label)
    return written


# ------------------------
# CLI
# ------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Edge scoring with Optuna + GNN (GINEConv)")
    parser.add_argument("--graph", type=str, help="Path to graph pkl")
    parser.add_argument("--tree", type=str, help="Path to tree pkl")
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output base directory. Required (no default) so a GNN run can never "
             "land in an existing XGBoost output tree by accident.",
    )
    parser.add_argument("--gid", type=str, help="Graph ID (required for single-graph)")
    parser.add_argument("--collection", type=str, help="Collection name")
    parser.add_argument("--manifest", type=str, help="Manifest JSON for collection operations")
    parser.add_argument("--score-only", action="store_true", help="Only score (requires trained model)")
    parser.add_argument("--model-path", type=str, help="Path to model directory")
    parser.add_argument("--n-trials", type=int, default=30, help="Optuna trial count")
    parser.add_argument("--plateau-patience", type=int, default=20)
    parser.add_argument("--plateau-min-trials", type=int, default=50)
    parser.add_argument("--plateau-min-delta", type=float, default=0.0001)
    parser.add_argument("--pruner-startup-trials", type=int, default=15)
    parser.add_argument(
        "--pruner-warmup-steps", type=int, default=10,
        help="MedianPruner: epochs before a trial can be pruned [default: 10]",
    )
    parser.add_argument("--max-epochs", type=int, default=200, help="Max epochs per fit")
    parser.add_argument(
        "--patience", type=int, default=20,
        help="Early stopping: epochs without validation-AUC improvement [default: 20]",
    )
    parser.add_argument(
        "--hidden-dim-range", type=int, nargs="+", default=DEFAULT_HIDDEN_DIM_RANGE,
        help="Hidden dimension choices for Optuna [default: 64 96 128 192]",
    )
    parser.add_argument(
        "--batch-norm", action="store_true",
        help="Add BatchNorm after each LayerNorm (off by default)",
    )
    parser.add_argument(
        "--head", choices=HEADS, default="endpoints+edge",
        help="Score-head input: [h_u||h_v||e_uv] (default) or [h_u||h_v] (ablation)",
    )
    parser.add_argument("--gpu-id", type=int, default=-1, help="GPU index; -1 = auto (GPU if available)")
    parser.add_argument("--optuna-enabled", dest="optuna_enabled", action="store_true")
    parser.add_argument("--no-optuna", dest="optuna_enabled", action="store_false")
    parser.set_defaults(optuna_enabled=True)
    parser.add_argument(
        "--hyperparams-config", type=str,
        help="JSON with hidden_dim, num_layers, dropout_rate, edge_dropout_rate, "
             "learning_rate, weight_decay. Required with --no-optuna.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--directed-mode", dest="directed_mode", action="store_true", default=False,
        help="Bidirect undirected graphs (microbiome, memetracker), as in xgb.py",
    )
    return parser.parse_args()


def _device(gpu_id):
    if gpu_id >= 0:
        if not torch.cuda.is_available():
            print(f"WARNING: GPU {gpu_id} requested but CUDA is not available; using CPU")
            return torch.device("cpu")
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = _device(args.gpu_id)
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))

    if not args.manifest:
        if not args.gid:
            raise ValueError("Either --manifest or --gid is required")
        if not args.collection:
            args.collection = ""
        graph_dir = get_graph_dir(args.output_dir, args.collection, args.gid)
        manifest_entry = {
            "graph_id": args.gid,
            "collection": args.collection,
            "node_features_path": os.path.join(graph_dir, FEATURE_DIR, NODE_FEATURES),
            "edge_features_path": os.path.join(graph_dir, FEATURE_DIR, EDGE_FEATURES),
        }
        if args.graph:
            manifest_entry["G_path"] = args.graph
        if args.tree:
            manifest_entry["T_path"] = args.tree
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([manifest_entry], f, indent=2)
            args.manifest = f.name

    if not args.collection:
        raise ValueError("--collection required")

    if args.score_only:
        model_path = args.model_path or get_model_dir(args.output_dir, args.collection)
        written = score_manifest(args.manifest, args.collection, args.output_dir, model_path, device)
        print(f"Scored {len(written)} graphs from manifest")
        return

    train_manifest(
        args.manifest,
        args.collection,
        args.output_dir,
        device,
        n_trials=args.n_trials,
        max_epochs=args.max_epochs,
        patience=args.patience,
        plateau_patience=args.plateau_patience,
        plateau_min_trials=args.plateau_min_trials,
        plateau_min_delta=args.plateau_min_delta,
        pruner_startup_trials=args.pruner_startup_trials,
        pruner_warmup_steps=args.pruner_warmup_steps,
        hidden_dim_range=args.hidden_dim_range,
        batch_norm=args.batch_norm,
        head=args.head,
        seed=args.seed,
        optuna_enabled=args.optuna_enabled,
        hyperparams_config=args.hyperparams_config,
        directed_mode=args.directed_mode,
    )


if __name__ == "__main__":
    sys.exit(main())
