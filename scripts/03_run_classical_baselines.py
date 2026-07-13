#!/usr/bin/env python3
"""Run classical DOA baselines on the aligned fixture.

All methods share the same steering-vector convention, 0.5° grid, source-number
estimator, and cardinality-aware scoring function used by the manuscript.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(name, "1")

from frost_doa.classical import METHODS, estimate_source_number_mdl
from frost_doa.metrics import penalized_set_metrics


def _scenario_names(fixture: Path) -> list[str]:
    with h5py.File(fixture, "r") as h5:
        return sorted(h5["scenarios"].keys())


def _run_scenario(
    fixture_str: str,
    scenario: str,
    output_dir_str: str,
    method_names: list[str],
    samples_per_scenario: int | None,
) -> dict[str, Any]:
    fixture = Path(fixture_str)
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    methods = {name: METHODS[name] for name in method_names}

    with h5py.File(fixture, "r") as h5:
        group = h5["scenarios"][scenario]
        n_total = int(group["Y_real"].shape[0])
        n = n_total if samples_per_scenario is None else min(n_total, samples_per_scenario)
        for index in range(n):
            y = np.asarray(group["Y_real"][index]) + 1j * np.asarray(group["Y_imag"][index])
            mask = np.asarray(group["theta_mask"][index]).astype(bool)
            theta_true = np.asarray(group["theta_deg_padded"][index])[mask].astype(float)
            p_hat = estimate_source_number_mdl(y, max_sources=4)
            for method, estimator in methods.items():
                tic = time.perf_counter()
                theta_hat = np.asarray(estimator(y, p_hat), dtype=float)
                runtime_ms = 1000.0 * (time.perf_counter() - tic)
                metric = penalized_set_metrics(theta_hat, theta_true, penalty_deg=25.0)
                rows.append(
                    {
                        "sample_id": f"{scenario}#{index:04d}",
                        "scenario": scenario,
                        "method": method,
                        "p_true": int(len(theta_true)),
                        "p_hat": int(p_hat),
                        "theta_true_deg_json": json.dumps([float(v) for v in theta_true]),
                        "theta_hat_deg_json": json.dumps([float(v) for v in theta_hat]),
                        "set_rmse_deg": float(metric.rmse_deg),
                        "set_mae_deg": float(metric.mae_deg),
                        "p_correct": int(p_hat == len(theta_true)),
                        "p_abs_error": int(abs(p_hat - len(theta_true))),
                        "outlier_gt_5deg": int(metric.rmse_deg > 5.0),
                        "runtime_ms": float(runtime_ms),
                        "implementation_id": method.lower().replace(" ", "_").replace("+", "plus")
                        + "_release_1_0",
                    }
                )
    frame = pd.DataFrame(rows)
    path = output_dir / f"classical_{scenario}.csv"
    frame.to_csv(path, index=False)
    return {
        "scenario": scenario,
        "n": n,
        "elapsed_s": time.perf_counter() - start,
        "path": str(path),
    }


def _summaries(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = (
        frame.groupby("method", sort=False)
        .agg(
            mean_rmse_deg=("set_rmse_deg", "mean"),
            median_rmse_deg=("set_rmse_deg", "median"),
            mean_mae_deg=("set_mae_deg", "mean"),
            source_number_accuracy_pct=("p_correct", lambda x: 100.0 * x.mean()),
            mean_source_number_abs_error=("p_abs_error", "mean"),
            outlier_rate_gt_5deg_pct=("outlier_gt_5deg", lambda x: 100.0 * x.mean()),
            mean_runtime_ms=("runtime_ms", "mean"),
            n=("sample_id", "count"),
        )
        .reset_index()
        .sort_values("mean_rmse_deg")
    )
    scenario = (
        frame.groupby(["scenario", "method"], sort=True)
        .agg(
            mean_rmse_deg=("set_rmse_deg", "mean"),
            median_rmse_deg=("set_rmse_deg", "median"),
            source_number_accuracy_pct=("p_correct", lambda x: 100.0 * x.mean()),
            outlier_rate_gt_5deg_pct=("outlier_gt_5deg", lambda x: 100.0 * x.mean()),
            mean_runtime_ms=("runtime_ms", "mean"),
            n=("sample_id", "count"),
        )
        .reset_index()
    )
    return overall, scenario


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/classical"))
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--samples-per-scenario", type=int, default=None)
    parser.add_argument("--skip-sbl", action="store_true")
    args = parser.parse_args()

    method_names = [name for name in METHODS if not (args.skip_sbl and name == "SBL + MDL")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial_dir = args.output_dir / "scenario_parts"
    partial_dir.mkdir(parents=True, exist_ok=True)
    scenarios = _scenario_names(args.fixture)
    reports = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _run_scenario,
                str(args.fixture),
                scenario,
                str(partial_dir),
                method_names,
                args.samples_per_scenario,
            ): scenario
            for scenario in scenarios
        }
        for future in as_completed(futures):
            report = future.result()
            reports.append(report)
            print(
                f"[done] {report['scenario']}: n={report['n']}, time={report['elapsed_s']:.1f}s",
                flush=True,
            )

    frames = [pd.read_csv(partial_dir / f"classical_{scenario}.csv") for scenario in scenarios]
    samples = pd.concat(frames, ignore_index=True)
    samples["scenario_order"] = pd.Categorical(
        samples["scenario"], categories=scenarios, ordered=True
    )
    samples = samples.sort_values(["scenario_order", "sample_id", "method"]).drop(
        columns="scenario_order"
    )
    overall, scenario_summary = _summaries(samples)
    samples.to_csv(args.output_dir / "classical_sample_scores.csv", index=False)
    overall.to_csv(args.output_dir / "classical_overall_summary.csv", index=False)
    scenario_summary.to_csv(args.output_dir / "classical_scenario_summary.csv", index=False)
    (args.output_dir / "classical_worker_report.json").write_text(
        json.dumps(sorted(reports, key=lambda x: x["scenario"]), indent=2), encoding="utf-8"
    )
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
