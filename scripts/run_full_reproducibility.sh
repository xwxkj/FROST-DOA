#!/usr/bin/env bash
# End-to-end aligned benchmark, including official DA-MUSIC and SubspaceNet adapters.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PHASEC_ROOT=""
MANUSCRIPT=""
SKIP_CORE=0
WORKERS="${FROST_WORKERS:-8}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --phasec-root) PHASEC_ROOT="$2"; shift 2 ;;
    --manuscript) MANUSCRIPT="$2"; shift 2 ;;
    --skip-core) SKIP_CORE=1; shift ;;
    --workers) WORKERS="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 --phasec-root PATH [--manuscript FILE.docx] [--skip-core] [--workers N]"; exit 0 ;;
    *) echo "Unknown argument: $1"; exit 2 ;;
  esac
done
[[ -n "$PHASEC_ROOT" ]] || { echo "[ERROR] --phasec-root is required"; exit 2; }
[[ -x .venv/bin/python ]] || bash scripts/setup_environment.sh
source .venv/bin/activate
export PYTHONPATH="$ROOT/src"

if [[ $SKIP_CORE -eq 0 ]]; then
  bash scripts/run_core_aligned_benchmark.sh --workers "$WORKERS"
fi
bash scripts/run_external_aligned.sh --phasec-root "$PHASEC_ROOT"

SCORES=(
  --score-file results/frost_doa/frost_doa_sample_scores.csv
  --score-file results/classical/classical_sample_scores.csv
)
for candidate in \
  results/external/scores/da_music_plus_mdl_sample_scores.csv \
  results/external/scores/subspacenet_plus_residual_selector_sample_scores.csv; do
  [[ -f "$candidate" ]] && SCORES+=(--score-file "$candidate")
done
python scripts/05_aggregate_results.py "${SCORES[@]}" --output-dir results/final
python scripts/06_make_paper_figures.py --results-dir results/final --output-dir figures/terminal_run

if [[ -n "$MANUSCRIPT" ]]; then
  set +e
  python scripts/07_audit_manuscript_results.py \
    --manuscript "$MANUSCRIPT" \
    --overall-summary results/final/overall_summary.csv \
    --output-dir audit/manuscript
  AUDIT_STATUS=$?
  set -e
  if [[ $AUDIT_STATUS -ne 0 ]]; then
    echo "[notice] manuscript contains stale values; replace them only with results/final and figures/terminal_run."
  fi
fi
zip -qr frost_doa_full_aligned_results_upload.zip results figures/terminal_run logs audit 2>/dev/null || \
  zip -qr frost_doa_full_aligned_results_upload.zip results figures/terminal_run logs
echo "[done] aligned full benchmark complete"
echo "[upload] $ROOT/frost_doa_full_aligned_results_upload.zip"
