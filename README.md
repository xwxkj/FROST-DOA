# FROST-DOA

**FROST-DOA: A Fuzzy Rule-Based Model for Adaptive Expert Selection in Robust Direction-of-Arrival Estimation**

FROST-DOA is a practical model-driven direction-of-arrival estimator. It combines scale-normalized diagnostics, eight fuzzy operating-regime rules, a bank of subspace and sparse experts, two fixed linear calibration rankers, deterministic bootstrap checks, and validation-fixed physics safeguards. Inference uses only the observed complex snapshots and nominal eight-sensor half-wavelength uniform-linear-array geometry; it does not use the true source number or a scenario label.

## Final aligned result

The terminal workflow evaluates ten methods on the same 1200 signal realizations. FROST-DOA obtains:

- mean cardinality-aware set RMSE: **7.818 degrees**;
- median set RMSE: **1.108 degrees**;
- source-number accuracy: **75.75%**;
- outlier rate above 5 degrees: **37.08%**.

The complete terminal summaries are under `results/final/`, and publication figures are under `figures/paper/`.

## Method components

- five covariance/subspace experts: sample-covariance, shrinkage, Toeplitz, forward-backward spatial-smoothing, and Tyler MUSIC;
- simultaneous orthogonal matching pursuit and three snapshot-weighted robust variants;
- three protected fuzzy-spectrum candidates;
- an optional residual first-order autoregressive colored-whitening family;
- a 202-feature base ranker and a 237-feature robust ranker;
- three-resample bootstrap stability;
- observation-only safeguards and a hard-mixed source-number correction.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Smoke test

```bash
bash scripts/run_core_aligned_benchmark.sh --smoke --workers 4
```

## Full aligned classical benchmark

```bash
bash scripts/run_core_aligned_benchmark.sh --workers 8
```

This command verifies assets, runs unit tests, reproduces the frozen FROST-DOA sample scores, evaluates the classical baselines, aggregates results, and creates Times New Roman figures in 600-dpi PNG and vector PDF/SVG formats.

## Re-evaluate DA-MUSIC and SubspaceNet

The external adapters reuse official repositories and trained checkpoints in an existing Phase-C directory:

```bash
bash scripts/run_external_aligned.sh \
  --phasec-root "$HOME/Documents/FROST_DOA202607/FROST_DOAexperiments/frost_doa_phaseC_official"
```

DA-MUSIC is reported as `DA-MUSIC + MDL`, and SubspaceNet is reported as `SubspaceNet + residual selector`; the labels disclose the non-oracle source-number wrappers used in the common benchmark.

## One-command workflow

```bash
bash scripts/run_full_reproducibility.sh \
  --phasec-root "$HOME/Documents/FROST_DOA202607/FROST_DOAexperiments/frost_doa_phaseC_official" \
  --workers 8
```

## Manuscript consistency gate

```bash
python scripts/07_audit_manuscript_results.py \
  --manuscript /absolute/path/to/manuscript.docx \
  --overall-summary results/final/overall_summary.csv \
  --output-dir audit/manuscript
```

Do not submit a manuscript until this audit reports `PASS`.

## Repository layout

```text
src/frost_doa/                  Public estimator, metrics, fixture, baselines
src/frost_doa/_frozen_core/     Frozen numerical controller and rankers
scripts/                        Rerun, scoring, plotting, and audit scripts
tests/                          Metric, naming, and reference tests
data/                           Smoke and aligned test fixtures
reference/                      Frozen FROST-DOA sample scores
results/final/                  Aligned terminal summaries
figures/paper/                  Terminal-generated publication figures
audit/terminal_run/             Final aligned run audits
```

## Reproducibility notice

An audit found that an earlier development fixture reused the same sample identifiers for different signal realizations. This release replaces it with `data/test_fixture_mc100_aligned.h5`, whose random streams exactly reproduce the frozen FROST-DOA sample scores. Historical development results should not be mixed with the released aligned fixture.

## Third-party code

DA-MUSIC and SubspaceNet are not redistributed. Their official repository commits and execution environments are recorded in the experiment logs. Users must obtain and comply with the upstream licenses.
