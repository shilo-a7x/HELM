#!/usr/bin/env bash
# Run the HELM pipeline: features -> train -> score -> arborescence -> metrics.
#
#   scripts/run_helm.sh --dataset {wiki|microbiome|memetracker} --output-dir DIR [options]
#   scripts/run_helm.sh --train TRAIN.json [--apply APPLY.json] --output-dir DIR [options]
#   scripts/run_helm.sh convert ...      # edge-list CSVs -> HELM graph files and a manifest
#   scripts/run_helm.sh summary FILE     # metrics table of a finished run
#
# --dataset trains on manifests/<dataset>_train.json and applies the model to
# manifests/<dataset>_test.json. Every other option is passed to `helm run`;
# `scripts/run_helm.sh --help` lists them. Run scripts/setup.sh first.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "core environment missing: run scripts/setup.sh first" >&2
  exit 1
fi
export HELM_GRAPHFX_PYTHON="${HELM_GRAPHFX_PYTHON:-$ROOT/graphfx/.venv/bin/python}"

case "${1:-}" in
  convert|summary) exec "$PY" -m helm.cli "$@" ;;
  -h|--help) exec "$PY" -m helm.cli run --help ;;
esac

ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --dataset)
      D="${2:?--dataset needs wiki, microbiome or memetracker}"
      case "$D" in wiki|microbiome|memetracker) ;; *) echo "unknown dataset: $D" >&2; exit 2 ;; esac
      ARGS+=(--train "$ROOT/manifests/${D}_train.json" --apply "$ROOT/manifests/${D}_test.json")
      shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

exec "$PY" -m helm.cli run "${ARGS[@]}"
