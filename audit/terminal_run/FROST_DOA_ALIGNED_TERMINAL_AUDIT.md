# FROST-DOA aligned terminal-run audit

## Submission decision

**PASS.** The manuscript table and figures now use only the terminal-generated results from the aligned 1200-sample signal fixture. The public algorithm name is `FROST-DOA`; no historical development-version label appears in the manuscript table or figures.

## Reproducibility gates

- Asset-hash verification: **PASS**.
- Unit tests: **PASS (4 tests)**.
- FROST-DOA frozen-reference verification: **PASS**.
- Maximum sample-level FROST-DOA RMSE difference from the frozen reference: **2.398e-13 degrees**.
- Common sample identifiers: **PASS**; all 10 methods contain the same 1200 sample IDs with no duplicates.
- DA-MUSIC inference flags: `uses_true_K = false`, `uses_scenario_label = false`.
- SubspaceNet inference flags: `uses_true_K = false`, `uses_scenario_label = false`.
- Plotting font selected by the terminal script: **Times New Roman**.
- Manuscript/result numeric audit: **PASS** at an absolute tolerance of 0.005.

## Final aligned overall results

| Method | Mean RMSE (deg) | Median RMSE (deg) | Source-number accuracy (%) | Outlier rate > 5 deg (%) |
|---|---:|---:|---:|---:|
| FROST-DOA | 7.818 | 1.108 | 75.75 | 37.08 |
| SubspaceNet + residual selector | 10.058 | 4.116 | 64.75 | 46.25 |
| SOMP + MDL | 12.554 | 16.711 | 48.58 | 61.58 |
| Root-MUSIC + MDL | 12.661 | 17.678 | 48.58 | 57.75 |
| Toeplitz-MUSIC + MDL | 12.827 | 17.679 | 48.58 | 57.00 |
| ESPRIT + MDL | 13.055 | 17.679 | 48.58 | 58.00 |
| FBSS-MUSIC + MDL | 14.009 | 17.693 | 48.58 | 61.00 |
| MUSIC + MDL | 16.721 | 17.860 | 48.58 | 67.58 |
| DA-MUSIC + MDL | 19.599 | 19.645 | 51.83 | 95.67 |
| SBL + MDL | 22.812 | 20.420 | 48.58 | 85.58 |

## Main quantitative findings

- FROST-DOA mean RMSE: **7.818 degrees**.
- Reduction relative to SubspaceNet + residual selector: **22.3%**.
- Reduction relative to SOMP + MDL, the strongest classical method by overall mean RMSE: **37.7%**.
- Reduction relative to DA-MUSIC + MDL: **60.1%**.
- FROST-DOA source-number accuracy: **75.75%**.
- Paired FROST-DOA win rate against SubspaceNet: **76.58%**.
- Paired FROST-DOA win rate against DA-MUSIC: **90.00%**.

## Scenario-level interpretation

Across all completed baselines, FROST-DOA has the lowest scenario-mean RMSE in **7 of 12** conditions. Root-MUSIC + MDL is slightly lower in the three clean conditions U1--U3, Toeplitz-MUSIC + MDL is lower in U8, and SubspaceNet + residual selector is lower in U5. Among the three model-driven methods (FROST-DOA, DA-MUSIC, and SubspaceNet), FROST-DOA is lowest in **11 of 12** conditions.

## Important scope note

The terminal archive does not contain reliable, directly comparable runtime measurements for the two external deep-learning pipelines: the DA-MUSIC runtime field is missing and the SubspaceNet wrapper records zero. Consequently, the manuscript does not make a cross-platform runtime claim or include runtime in the main comparison table.

## Files used as the sole numerical source

- `results/final/overall_summary.csv`
- `results/final/scenario_summary.csv`
- `results/final/paired_vs_frost_doa.csv`
- `figures/terminal_run/*.png|*.pdf|*.svg`
