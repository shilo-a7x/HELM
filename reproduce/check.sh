#!/usr/bin/env bash
# Run one dataset and compare the per-graph metrics with the stored reference results.
#
#   reproduce/check.sh {wiki|microbiome|memetracker} [--tol X] [run_helm.sh options]
#
# The output goes to runs/check_<dataset> (add --output-dir to change it). The reference was produced
# with GPU features including motifs, so a GPU run with motifs should match it exactly (tolerance
# 1e-9, the default). On a CPU the Louvain and Leiden features come from different implementations
# and the metrics move slightly (README, "Reproducibility"); use a looser --tol, for example 0.08.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ $# -lt 1 ]; then
  echo "usage: reproduce/check.sh {wiki|microbiome|memetracker} [--tol X] [run_helm.sh options]" >&2
  exit 2
fi
D="$1"
shift
TOL=1e-9
OUT="runs/check_$D"
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --tol) TOL="${2:?--tol needs a number}"; shift 2 ;;
    --output-dir) OUT="${2:?--output-dir needs a path}"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

"$ROOT/scripts/run_helm.sh" --dataset "$D" --output-dir "$OUT" ${ARGS[@]+"${ARGS[@]}"}

STATUS=0
for split in train apply; do
  echo
  echo "== $D, $split =="
  "$ROOT/.venv/bin/python" "$ROOT/reproduce/compare.py" "$OUT/results_$split.csv" \
    "$ROOT/reproduce/expected/${D}_$split.csv" --tol "$TOL" || STATUS=1
done
exit $STATUS
