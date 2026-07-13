#!/usr/bin/env bash
# Re-evaluate the already trained official DA-MUSIC and SubspaceNet pipelines on
# the signal fixture that exactly matches the frozen FROST-DOA reference.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/run_external_aligned.sh --phasec-root PATH [--skip-damusic] [--skip-subspacenet]

PATH must point to the existing frost_doa_phaseC_official directory containing:
  official_repos/, results_damusic_step3b/, results_subspacenet_step2b/,
  .venv_subspacenet/, and data/trainval_fixture.h5.
EOF
}

RELEASE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PHASEC_ROOT=""
RUN_DAMUSIC=1
RUN_SUBSPACE=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --phasec-root) PHASEC_ROOT="$2"; shift 2 ;;
    --skip-damusic) RUN_DAMUSIC=0; shift ;;
    --skip-subspacenet) RUN_SUBSPACE=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 2 ;;
  esac
done
if [[ -z "$PHASEC_ROOT" ]]; then
  echo "[ERROR] --phasec-root is required"; usage; exit 2
fi
PHASEC_ROOT="$(cd "$PHASEC_ROOT" && pwd)"
FIXTURE="$RELEASE_ROOT/data/test_fixture_mc100_aligned.h5"
PRED_DIR="$RELEASE_ROOT/results/external/predictions"
SCORE_DIR="$RELEASE_ROOT/results/external/scores"
LOG_DIR="$RELEASE_ROOT/logs/external"
mkdir -p "$PRED_DIR" "$SCORE_DIR" "$LOG_DIR"

PY="$RELEASE_ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then PY="python3"; fi
export PYTHONPATH="$RELEASE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ $RUN_DAMUSIC -eq 1 ]]; then
  echo "[DA-MUSIC] evaluating official TensorFlow builder on aligned fixture"
  if ! command -v docker >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; then
    echo "[ERROR] Docker Desktop must be installed and running for DA-MUSIC." >&2
    exit 10
  fi
  for required in \
    "$PHASEC_ROOT/official_repos/DA-MUSIC_TVT23" \
    "$PHASEC_ROOT/results_damusic_step3b/damusic_step3b_deep_aug_MUSIC_isv_K1.h5" \
    "$PHASEC_ROOT/results_damusic_step3b/damusic_step3b_deep_aug_MUSIC_isv_K2.h5" \
    "$PHASEC_ROOT/results_damusic_step3b/damusic_step3b_deep_aug_MUSIC_isv_K3.h5" \
    "$PHASEC_ROOT/env/damusic_tf24_pyroom_tqdm/Dockerfile"; do
    [[ -e "$required" ]] || { echo "[ERROR] missing $required"; exit 11; }
  done
  IMAGE="frost-damusic-tf24-pyroom-tqdm:release-1.0"
  docker build --platform=linux/amd64 -t "$IMAGE" \
    -f "$PHASEC_ROOT/env/damusic_tf24_pyroom_tqdm/Dockerfile" "$PHASEC_ROOT" \
    2>&1 | tee "$LOG_DIR/damusic_docker_build.log"
  docker run --rm --platform=linux/amd64 \
    -v "$PHASEC_ROOT":/workspace \
    -v "$RELEASE_ROOT":/release \
    -e WORKSPACE=/workspace \
    -e DAMUSIC_REPO=/workspace/official_repos/DA-MUSIC_TVT23 \
    -e DAMUSIC_TEST=/release/data/test_fixture_mc100_aligned.h5 \
    -e DAMUSIC_OUT=/release/results/external/damusic_run \
    -e DAMUSIC_RAW_PRED=/release/results/external/predictions/DA-MUSIC.csv \
    -e DAMUSIC_BUILDER=deep_aug_MUSIC_isv \
    -e DAMUSIC_SNAPSHOT_TARGET=200 \
    "$IMAGE" bash -lc 'cd /release && python scripts/external/damusic_aligned_predict.py' \
    2>&1 | tee "$LOG_DIR/damusic_aligned_predict.log"
  "$PY" "$RELEASE_ROOT/scripts/04_score_prediction_csv.py" \
    --fixture "$FIXTURE" \
    --predictions "$PRED_DIR/DA-MUSIC.csv" \
    --method "DA-MUSIC + MDL" \
    --output-dir "$SCORE_DIR" \
    2>&1 | tee "$LOG_DIR/damusic_aligned_score.log"
fi

if [[ $RUN_SUBSPACE -eq 1 ]]; then
  echo "[SubspaceNet] evaluating official model bank on aligned fixture"
  SUBSPACE_PY="$PHASEC_ROOT/.venv_subspacenet/bin/python"
  [[ -x "$SUBSPACE_PY" ]] || { echo "[ERROR] missing $SUBSPACE_PY"; exit 20; }
  for required in \
    "$PHASEC_ROOT/official_repos/SubspaceNet" \
    "$PHASEC_ROOT/results_subspacenet_step2b/subspacenet_step2b_variant_summary.csv" \
    "$PHASEC_ROOT/data/trainval_fixture.h5"; do
    [[ -e "$required" ]] || { echo "[ERROR] missing $required"; exit 21; }
  done
  PHASEC_ROOT="$PHASEC_ROOT" \
  FROST_RELEASE_ROOT="$RELEASE_ROOT" \
  SUBSPACENET_TEST_FIXTURE="$FIXTURE" \
  SUBSPACE_STEP3B_DEVICE=cpu \
  "$SUBSPACE_PY" "$RELEASE_ROOT/scripts/external/subspacenet_aligned_predict.py" \
    2>&1 | tee "$LOG_DIR/subspacenet_aligned_predict.log"
  "$PY" "$RELEASE_ROOT/scripts/04_score_prediction_csv.py" \
    --fixture "$FIXTURE" \
    --predictions "$PRED_DIR/SubspaceNet.csv" \
    --method "SubspaceNet + residual selector" \
    --output-dir "$SCORE_DIR" \
    2>&1 | tee "$LOG_DIR/subspacenet_aligned_score.log"
fi

echo "[done] aligned external predictions and scores are under results/external/"
