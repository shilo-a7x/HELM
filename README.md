# HELM: Hierarchy via Edge Learning and MST

This repository is the reference implementation of *HELM: Hierarchy via Edge Learning and MST*.

HELM recovers a hierarchy, a rooted directed tree, from a graph. A model trained on topological
features scores every edge with the probability that it belongs to the hierarchy; a minimum
spanning arborescence on the log-odds weights `-log(score / (1 - score))` then returns the tree,
choosing its edges, their directions and its root in one step. A small share of labelled edges
(10% in the experiments) is enough to train the scorer (few-shot), and a trained model can be
applied to new graphs of the same kind with no labels (zero-shot).

```
graph  ->  graphfx features  ->  XGBoost (or GNN) edge scores  ->  arborescence on log-odds weights  ->  tree + root
```

The repository contains the code, 60 graphs with ground-truth hierarchies (Wikipedia categories,
microbiome co-occurrence networks, MemeTracker cascades), and the scripts to build the
environments and run everything with one command.

## Requirements

- Linux (tested on AlmaLinux 9), `git`, `make`, `g++`, `curl`, and [`uv`](https://docs.astral.sh/uv/)
  (it also downloads Python 3.11 and 3.13 when they are missing).
- About 10 GB of disk for the environments.
- Optional, for the GPU: an NVIDIA driver and the CUDA toolkit with `nvcc` (tested with CUDA 12.5
  on an L40S). Without a GPU everything runs on the CPU.
- [conda](https://docs.conda.io) for the motif-features environment. If it is not installed, setup
  installs Miniconda into the repository directory.

## Setup

```bash
scripts/setup.sh                # detects a GPU and builds everything
```

This creates, inside the repository directory:

| | |
|---|---|
| `.venv` | Python 3.11: scoring, arborescence, the `helm` command (pinned in `requirements.lock`) |
| `graphfx/.venv` | Python 3.13: the feature code, with cuGraph and nx-cugraph on a GPU machine |
| `.motif-env`, `graph-measures/` | a Python 3.7 conda environment and the compiled upstream [graph-measures](https://github.com/louzounlab/graph-measures) package, used for the motif3 and motif4 features |

and ends by checking that the compiled motif code reproduces known counts. It is safe to re-run.

Options: `--gpu auto|on|off`, `--with-gnn` (installs PyTorch and PyTorch Geometric for the GNN
scorer), `--skip-motif` (skips the motif environment; run with `--skip-motifs` then, see below),
`--rebuild-motif`, `--accept-conda-tos` (newer conda versions refuse the Anaconda channels used
by graph-measures' environment file until their terms of service are accepted; read them first).

## Quick start

```bash
scripts/run_helm.sh --dataset microbiome --output-dir runs/microbiome
```

This computes the features for the 20 microbiome graphs, trains on the 10 training graphs
(few-shot), scores the 10 test graphs with that model (zero-shot), builds the trees and prints
the metrics. It takes under a minute. With GPU features the means over the graphs are:

| | directed recall | undirected recall | root accuracy | root distance |
|---|---|---|---|---|
| train graphs (few-shot) | 0.898 | 0.932 | 1.0 | 0.0 |
| test graphs (zero-shot) | 0.879 | 0.928 | 0.9 | 0.4 |

`--dataset` accepts `wiki`, `microbiome` and `memetracker`. It trains on `manifests/<name>_train.json`
and applies the model to `manifests/<name>_test.json`.

## Reading the results

```
<output-dir>/
  results_train.csv, results_apply.csv     per-graph metrics, all families of the run
  <family>/
    model/                                 trained model, scaler, hyperparameters, ROC curves, AUCs
    <graph_id>/
      features/                            node and edge feature CSVs, backend.json
      scores/edge_scores.csv               one probability per (directed) edge
      edmonds/arborescence.edgelist        the predicted tree, parent -> child
      edmonds/arborescence.pkl, edmonds_results.json
  manifests/                               the per-run manifests the stages used
```

Metrics, per graph (`results_*.csv`) and as mean and standard deviation over graphs (printed):

| | |
|---|---|
| `directed_recall` | share of the true parent -> child edges the tree contains with the right direction |
| `undirected_recall` | the same, ignoring direction |
| `directed_precision` | share of predicted edges that are correct |
| `root_accuracy`, `root_distance` | 1 if the predicted root is the true root; hops between predicted and true root |

"Train" results are on the graphs the model was trained on (a sample of their edges, the few-shot
setting); "apply" results are on graphs the model has not seen. `scripts/run_helm.sh summary
runs/microbiome/results_apply.csv` prints the table again.

## Your own graphs

Graphs are edge lists (see `data/README.md`). To turn plain CSVs into HELM files:

```bash
# graph.csv: source,target   hierarchy.csv (optional): parent,child
scripts/run_helm.sh convert --edges graph.csv --hierarchy hierarchy.csv \
    --undirected --name mygraph --collection microbiome --manifest mine.json
```

The vertex IDs must be integers, the graph connected and without self-loops, and the hierarchy a
rooted tree over vertices of the graph; otherwise `convert` says what is wrong and writes nothing.

- **No ground truth, apply a model trained on a bundled family.** Give the graph the family it
  resembles as its collection (`--collection microbiome`, `wiki` or `memetracker`) and
  ```bash
  scripts/run_helm.sh --train manifests/microbiome_train.json --apply mine.json --output-dir runs/mine
  ```
  The tree is in `runs/mine/microbiome/mygraph/edmonds/arborescence.edgelist`. Results are only as
  good as the resemblance to the training family.
- **Your own labelled graphs.** Convert them with `--hierarchy` and pass them as `--train`; HELM
  trains on a sample of their edges. Add `--apply` for graphs without labels.
- Use `--directed-mode auto` (the default): undirected graphs are bidirected and the arborescence
  chooses the orientation; directed graphs are used as they are.

## Options of `helm run`

`scripts/run_helm.sh --help` lists them all. The main ones:

| | |
|---|---|
| `--model xgb\|gnn` | XGBoost (default) or the GINE graph neural network (needs `setup.sh --with-gnn`) |
| `--gpu auto\|on\|off` | GPU use for features (and the GNN) |
| `--skip-motifs` | drop the motif features, which saves building the motif environment |
| `--hyperparams FILE`, `--tune` | the hyperparameters in `configs/` are used by default; `--tune` runs Optuna instead |
| `--seed N` | the seed for sampling and training [42] |
| `--solver tarjan\|networkx` | arborescence solver; `tarjan` (default) finds the same optimal weight much faster |
| `--features-dir DIR` | use precomputed features |
| `--deterministic` | GNN only: deterministic GPU kernels |
| `--overwrite` | reuse an existing output directory (without it, a non-empty one is refused) |

## Reproducibility

```bash
reproduce/check.sh microbiome        # runs the dataset and compares with the stored reference results
```

`reproduce/expected/` holds the per-graph metrics of the reference runs for the three families.

- **XGBoost.** A GPU run with motifs reproduces the reference results exactly.
- **CPU and GPU features differ.** `louvain` and `leiden` are computed by different implementations
  (NetworkX on the CPU, cuGraph on the GPU) and give different communities on larger graphs. A
  CPU run therefore gives slightly different metrics (on microbiome about 0.01 lower in directed
  recall, on wiki about 0.002). Train and apply on features from the same backend; `features/backend.json` records which one produced them,
  and `helm run` warns about mixtures. Use a looser tolerance when checking on a CPU:
  `reproduce/check.sh microbiome --gpu off --tol 0.08`.
- **`--skip-motifs`** changes the feature set, so the results differ from runs with motifs.
- **GNN.** On a GPU, training is not repeatable even with a fixed seed (the scatter operations
  accumulate in varying order), and with the small training sets the result varies by about
  0.02 in mean recall between seeds. `--deterministic` makes a fixed seed repeatable. Report GNN
  results over several seeds.

## Layout

```
helm/            the package: cli, io, scoring/ (xgb, gnn), arborescence/
graphfx/         feature computation (own environment); docs/FEATURES.md lists the features
scripts/         setup.sh, run_helm.sh
manifests/       the train / test lists of the 60 graphs
configs/         tuned hyperparameters per family
data/            the graphs (data/README.md)
reproduce/       reference results and the comparison script
```

## License

GPL-3.0-or-later, see `LICENSE`. Third-party software and the terms of the data are in
`THIRD_PARTY_NOTICES.md` and `data/README.md`.
