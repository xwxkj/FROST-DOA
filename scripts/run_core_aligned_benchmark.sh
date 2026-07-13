#!/usr/bin/env bash
# Run FROST-DOA and classical baselines on the exactly aligned signal fixture.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MODE="full"
WORKERS="${FROST_WORKERS:-8}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) MODE="smoke"; shift ;;
    --workers) WORKERS="$2"; shift 2 ;;
    -h|--help) echo "Usage: $0 [--smoke] [--workers N]"; exit 0 ;;
    *) echo "Unknown argument: $1"; exit 2 ;;
  esac
done
[[ -x .venv/bin/python ]] || bash scripts/setup_environment.sh
source .venv/bin/activate
export PYTHONPATH="$ROOT/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p results logs figures/terminal_run
pytest -q | tee logs/pytest.log

if [[ "$MODE" == "smoke" ]]; then
  FIXTURE="data/smoke_fixture.h5"
  RESULT_ROOT="results/smoke"
  SBL_FLAG="--skip-sbl"
else
  FIXTURE="data/test_fixture_mc100_aligned.h5"
  RESULT_ROOT="results"
  SBL_FLAG=""
fi

python scripts/01_run_frost_doa_parallel.py \
  --fixture "$FIXTURE" --output-dir "$RESULT_ROOT/frost_doa" --workers "$WORKERS" \
  2>&1 | tee "logs/frost_doa_${MODE}.log"

if [[ "$MODE" == "full" ]]; then
  python scripts/02_verify_frost_reference.py \
    --actual "$RESULT_ROOT/frost_doa/frost_doa_sample_scores.csv" \
    --reference reference/frost_doa_frozen_sample_scores.csv \
    --report "$RESULT_ROOT/frost_doa/reference_verification.json" \
    2>&1 | tee logs/frost_reference_verification.log
fi

python scripts/03_run_classical_baselines.py \
  --fixture "$FIXTURE" --output-dir "$RESULT_ROOT/classical" --workers "$WORKERS" $SBL_FLAG \
  2>&1 | tee "logs/classical_${MODE}.log"

python scripts/05_aggregate_results.py \
  --score-file "$RESULT_ROOT/frost_doa/frost_doa_sample_scores.csv" \
  --score-file "$RESULT_ROOT/classical/classical_sample_scores.csv" \
  --output-dir "$RESULT_ROOT/final" \
  2>&1 | tee "logs/aggregate_core_${MODE}.log"

python scripts/06_make_paper_figures.py \
  --results-dir "$RESULT_ROOT/final" \
  --output-dir "$RESULT_ROOT/figures" \
  2>&1 | tee "logs/figures_core_${MODE}.log"

zip -qr "frost_doa_${MODE}_core_results.zip" "$RESULT_ROOT" logs || true
echo "[done] $MODE core benchmark complete"
echo "[upload] $ROOT/frost_doa_${MODE}_core_results.zip"
