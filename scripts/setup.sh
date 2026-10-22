#!/usr/bin/env bash
# Build every HELM environment. Safe to re-run: finished steps are detected and skipped.
#
#   scripts/setup.sh [--gpu auto|on|off] [--with-gnn] [--skip-motif] [--rebuild-motif] [--accept-conda-tos]
#
# Creates, inside the repository directory:
#   .venv            core environment (Python 3.11): scoring, arborescence, the helm CLI
#   graphfx/.venv    feature environment (Python 3.13): graphfx, plus cuGraph / nx-cugraph on a GPU machine
#   .motif-env       conda environment (Python 3.7, Boost.Python) for the motif3/motif4 features
#   graph-measures/  clone of louzounlab/graph-measures (GPL-3.0), compiled inside .motif-env
# Needs: uv, git, make, g++, curl. GPU machines also need the CUDA toolkit (nvcc) and an NVIDIA driver.
# conda is used if installed, otherwise Miniconda is installed into .miniconda3.
set -euo pipefail

GPU=auto
WITH_GNN=0
SKIP_MOTIF=0
REBUILD_MOTIF=0
ACCEPT_TOS=0
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) GPU="${2:?--gpu needs auto, on or off}"; shift 2 ;;
    --with-gnn) WITH_GNN=1; shift ;;
    --skip-motif) SKIP_MOTIF=1; shift ;;
    --rebuild-motif) REBUILD_MOTIF=1; shift ;;
    --accept-conda-tos) ACCEPT_TOS=1; shift ;;
    -h|--help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# an activated virtual environment from another project would hijack `uv` and `python`
unset VIRTUAL_ENV
say() { printf '[setup] %s\n' "$*"; }
die() { printf '[setup] ERROR: %s\n' "$*" >&2; exit 1; }
DONE=()
CHECK_FAILED=0

# graph-measures revision the motif counts were verified against
GM_URL=https://github.com/louzounlab/graph-measures.git
GM_COMMIT=04802ff52f4cc8ac773194fb3c08639eb9cb3cb3

command -v uv >/dev/null 2>&1 || die "uv is required (it also fetches Python 3.11 and 3.13). Install: curl -LsSf https://astral.sh/uv/install.sh | sh"

case "$GPU" in
  on)   USE_GPU=1 ;;
  off)  USE_GPU=0 ;;
  auto) if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then USE_GPU=1; else USE_GPU=0; fi ;;
  *)    die "--gpu must be auto, on or off" ;;
esac
say "variant: $([ "$USE_GPU" = 1 ] && echo 'GPU (CUDA)' || echo 'CPU only')"

FREE_GB=$(df -Pk "$ROOT" | awk 'NR==2 {printf "%d", $4/1048576}')
say "free disk on the repository volume: ${FREE_GB} GB (the environments need about 10 GB)"
[ "$FREE_GB" -ge 10 ] || say "WARNING: less than 10 GB free; the graphfx GPU environment alone is ~7.5 GB"

# ---------------------------------------------------------------- 1. core environment
if [ -x .venv/bin/python ]; then say "core environment exists (.venv), re-syncing"; else uv venv --python 3.11 .venv; DONE+=("created .venv (Python 3.11)"); fi
uv pip sync --python .venv/bin/python requirements.lock
uv pip install --python .venv/bin/python --no-deps -e .
DONE+=("installed pinned core packages from requirements.lock, and the helm package (editable)")

# ---------------------------------------------------------------- 2. optional GNN scorer
if [ "$WITH_GNN" = 1 ]; then
  if [ "$USE_GPU" = 1 ]; then
    uv pip install --python .venv/bin/python -r requirements-gnn.txt
    DONE+=("installed torch (default CUDA build) and torch-geometric")
  else
    uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
    uv pip install --python .venv/bin/python torch-geometric
    DONE+=("installed torch (CPU build) and torch-geometric")
  fi
fi

# ---------------------------------------------------------------- 3. feature environment
if [ "$USE_GPU" = 1 ]; then EXTRAS=(--extra gpu --extra leiden-cpu); else EXTRAS=(--extra leiden-cpu); fi
( cd graphfx && uv sync --frozen "${EXTRAS[@]}" )
DONE+=("synced graphfx/.venv from graphfx/uv.lock (${EXTRAS[*]})")
.venv/bin/python -c "import xgboost, optuna, networkx, sklearn, pandas, helm.cli"
graphfx/.venv/bin/python -c "import graphfx, springrank"
DONE+=("import checks passed (core and graphfx)")

# ---------------------------------------------------------------- 4. motif environment
MENV="$ROOT/.motif-env"
GM="$ROOT/graph-measures"
if [ "$SKIP_MOTIF" = 1 ]; then
  say "skipping the motif environment (--skip-motif): run the pipeline with --skip-motifs"
else
  for tool in git make g++ curl; do command -v "$tool" >/dev/null 2>&1 || die "$tool is required for the motif build"; done

  # 4a. conda
  if command -v conda >/dev/null 2>&1; then
    CONDA="$(command -v conda)"
  else
    CONDA="$ROOT/.miniconda3/bin/conda"
    if [ ! -x "$CONDA" ]; then
      say "conda not found: installing Miniconda into $ROOT/.miniconda3"
      curl -fsSL -o "$ROOT/.miniconda-installer.sh" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
      bash "$ROOT/.miniconda-installer.sh" -b -p "$ROOT/.miniconda3" >/dev/null
      rm -f "$ROOT/.miniconda-installer.sh"
      DONE+=("installed Miniconda into .miniconda3")
    fi
  fi
  # Recent conda versions refuse the Anaconda channels (used by upstream's env.yml) until their terms
  # of service are accepted; the script does that only when asked to.
  if "$CONDA" tos --help >/dev/null 2>&1; then
    if [ "$ACCEPT_TOS" = 1 ]; then
      for ch in https://repo.anaconda.com/pkgs/main https://repo.anaconda.com/pkgs/r; do
        "$CONDA" tos accept --override-channels --channel "$ch" >/dev/null
      done
      say "accepted the Anaconda terms of service for the 'main' and 'r' channels (--accept-conda-tos)"
    else
      say "note: if conda stops because the Anaconda terms of service are not accepted, read them and re-run with --accept-conda-tos"
    fi
  fi

  # 4b. graph-measures checkout
  if [ ! -d "$GM/.git" ]; then
    git clone -q "$GM_URL" "$GM"
    git -C "$GM" checkout -q "$GM_COMMIT"
    DONE+=("cloned graph-measures at ${GM_COMMIT:0:10}")
  else
    say "graph-measures checkout exists ($(git -C "$GM" rev-parse --short HEAD))"
  fi
  SRC="$GM/graphMeasures/features_algorithms/accelerated_graph_features/src"
  [ -f "$SRC/env.yml" ] || die "unexpected graph-measures layout: $SRC/env.yml not found"

  # 4c. conda environment (pinned: Python 3.7.1, Boost 1.67)
  if [ ! -x "$MENV/bin/python" ]; then
    # a previous attempt may have left a half-built prefix; conda would refuse or prompt
    rm -rf "$MENV"
    # no --yes: old conda versions (4.14) do not accept it for `env create`, and it never prompts here
    "$CONDA" env create -f "$SRC/env.yml" -p "$MENV"
    DONE+=("created the motif conda environment (.motif-env, Python 3.7)")
  else
    say "motif environment exists (.motif-env)"
  fi

  # 4d. compile
  SO="$SRC/accelerated_graph_features/_features.so"
  if [ -f "$SO" ] && [ "$REBUILD_MOTIF" = 0 ]; then
    say "graph-measures already compiled ($SO)"
  else
    if [ "$USE_GPU" = 1 ]; then
      NVCC_BIN="/usr/local/cuda/bin/nvcc"
      [ -x "$NVCC_BIN" ] || NVCC_BIN="$(command -v nvcc || true)"
      [ -n "$NVCC_BIN" ] || die "nvcc not found. The GPU motif build needs the CUDA toolkit (e.g. /usr/local/cuda). Install it, or run with --gpu off for the CPU build."
      NV_PREFIX="$(dirname "$(dirname "$(readlink -f "$NVCC_BIN")")")"
      CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')"
      [ -n "$CC" ] || die "could not read the GPU compute capability from nvidia-smi"
      say "GPU build: nvcc at $NV_PREFIX, architecture sm_$CC"
      # Upstream hardcodes an old architecture (sm_35, rejected by current nvcc) in two places.
      sed -i "s/-arch=sm_[0-9]*/-arch=sm_${CC}/g; s#^NV_PREFIX *=.*#NV_PREFIX = ${NV_PREFIX}#" "$SRC/Makefile-gpu"
      MAKEFILE=Makefile-gpu
    else
      MAKEFILE=Makefile-conda
    fi
    say "compiling graph-measures with $MAKEFILE (a few minutes; log: $ROOT/.motif-build.log)"
    # The makefiles only use $(CONDA_PREFIX) for Python and Boost, so no conda activation is needed
    # (which also avoids depending on where this machine's conda keeps conda.sh).
    # The compiler prints hundreds of harmless deprecation warnings, so the output goes to a log file.
    if ! ( cd "$SRC" && CONDA_PREFIX="$MENV" PATH="$MENV/bin:$PATH" make -f "$MAKEFILE" CONDA_PREFIX="$MENV" ) > "$ROOT/.motif-build.log" 2>&1; then
      tail -40 "$ROOT/.motif-build.log" >&2
      die "compiling graph-measures failed (full log: $ROOT/.motif-build.log)"
    fi
    [ -f "$SO" ] || die "build finished but $SO was not produced"
    DONE+=("compiled graph-measures ($MAKEFILE)")
  fi

  # 4e. the conda runtime ships an old libstdc++ that cannot load the freshly compiled extension
  # (importing graphMeasures does not load the extension, so load it directly)
  import_check() {
    PYTHONPATH="$GM" "$MENV/bin/python" -c "
import sys
sys.path.insert(0, '$(dirname "$SO")')
import _features
from graphMeasures import FeatureCalculator" 2>&1
  }
  if ! OUT="$(import_check)"; then
    if grep -q -E "GLIBCXX|undefined symbol" <<<"$OUT"; then
      SYS_LIB="$(ldconfig -p 2>/dev/null | awk '/libstdc\+\+\.so\.6 /{print $NF; exit}')"
      [ -n "$SYS_LIB" ] || SYS_LIB="$(ls /usr/lib64/libstdc++.so.6 /usr/lib/x86_64-linux-gnu/libstdc++.so.6 2>/dev/null | head -1)"
      [ -n "$SYS_LIB" ] || die "the compiled extension does not load and no system libstdc++ was found: $OUT"
      ln -sf "$(readlink -f "$SYS_LIB")" "$MENV/lib/libstdc++.so.6"
      DONE+=("pointed .motif-env at the system libstdc++ ($SYS_LIB)")
      OUT="$(import_check)" || die "the compiled extension still does not load: $OUT"
    else
      die "the compiled extension does not load: $OUT"
    fi
  fi
  DONE+=("the compiled graph-measures extension loads inside .motif-env")

  # 4f. functional check: motif counts of one microbiome graph must equal the stored values exactly
  if GRAPHFX_MOTIF_ENV_PYTHON="$MENV/bin/python" GRAPHFX_GRAPH_MEASURES_PATH="$GM" USE_GPU="$USE_GPU" ROOT="$ROOT" \
     graphfx/.venv/bin/python - <<'PY'
import os, sys
import pandas as pd
sys.path.insert(0, os.environ["ROOT"])
import graphfx
from helm.io import load_graph
G = load_graph(os.path.join(os.environ["ROOT"], "data/microbiome/C0001/G.edgelist"))
got = graphfx.compute_node_features(G, features=["motif3", "motif4"], gpu=os.environ["USE_GPU"] == "1")
want = pd.read_csv(os.path.join(os.environ["ROOT"], "reproduce/expected/microbiome_C0001_motifs.csv"))
m = want.merge(got, on="node", suffixes=("_want", "_got"))
bad = [c for c in want.columns if c != "node" and not (m[c + "_want"] == m[c + "_got"]).all()]
assert len(m) == len(want) == len(got), "vertex sets differ"
sys.exit(1 if bad else 0)
PY
  then
    DONE+=("motif check passed (microbiome C0001 motif3/motif4 equal the stored counts)")
  else
    CHECK_FAILED=1
    say "WARNING: the motif check FAILED: the motif counts differ from the stored values"
  fi
fi

echo
say "done:"
for d in "${DONE[@]}"; do echo "  - $d"; done
echo
say "next: scripts/run_helm.sh --dataset microbiome --output-dir runs/microbiome"
exit "$CHECK_FAILED"
