#!/usr/bin/env python3
"""Run FROST-DOA on an aligned HDF5 fixture in scenario-parallel workers.

This is the recommended terminal entry point for the full MC100 experiment. Each
worker owns one scenario and writes a temporary CSV; the parent process then
merges the files in deterministic scenario/sample order.
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

# Keep each process single-threaded. Parallelism is across scenarios, not BLAS.
for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(name, "1")

from frost_doa import FROSTDOAEstimator
from frost_doa.metrics import penalized_set_metrics


def _scenario_names(fixture: Path) -> list[str]:
    with h5py.File(fixture, "r") as h5:
        return sorted(h5["scenarios"].keys())


def _run_scenario(
    fixture_str: str,
    scenario: str,
    output_dir_str: str,
    samples_per_scenario: int | None,
) -> dict[str, Any]:
    """Evaluate one scenario and write a deterministic temporary CSV."""
    fixture = Path(fixture_str)
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    estimator = FROSTDOAEstimator()
    rows: list[dict[str, Any]] = []
    start_scenario = time.perf_counter()

    with h5py.File(fixture, "r") as h5:
        group = h5["scenarios"][scenario]
        n_total = int(group["Y_real"].shape[0])
        n = n_total if samples_per_scenario is None else min(n_total, samples_per_scenario)
        for index in range(n):
            y = np.asarray(group["Y_real"][index]) + 1j * np.asarray(group["Y_imag"][index])
            mask = np.asarray(group["theta_mask"][index]).astype(bool)
            theta_true = np.asarray(group["theta_deg_padded"][index])[mask].astype(float)
            tic = time.perf_counter()
            estimate = estimator.estimate(y)
            runtime_ms = (time.perf_counter() - tic) * 1000.0
            metrics = penalized_set_metrics(estimate.doas_deg, theta_true, penalty_deg=25.0)
            rows.append(
                {
                    "sample_id": f"{scenario}#{index:04d}",
                    "scenario": scenario,
                    "method": "FROST-DOA",
                    "p_true": int(len(theta_true)),
                    "p_hat": int(estimate.num_sources),
                    "theta_true_deg_json": json.dumps([float(v) for v in theta_true]),
                    "theta_hat_deg_json": json.dumps([float(v) for v in estimate.doas_deg]),
                    "set_rmse_deg": float(metrics.rmse_deg),
                    "set_mae_deg": float(metrics.mae_deg),
                    "p_correct": int(estimate.num_sources == len(theta_true)),
                    "p_abs_error": int(abs(estimate.num_sources - len(theta_true))),
                    "outlier_gt_5deg": int(metrics.rmse_deg > 5.0),
                    "runtime_ms": float(runtime_ms),
                    "selected_branch": estimate.branch,
                    "selected_expert": estimate.expert,
                }
            )
    frame = pd.DataFrame(rows)
    path = output_dir / f"frost_doa_{scenario}.csv"
    frame.to_csv(path, index=False)
    return {
        "scenario": scenario,
        "n": len(frame),
        "mean_rmse_deg": float(frame["set_rmse_deg"].mean()),
        "elapsed_s": float(time.perf_counter() - start_scenario),
        "path": str(path),
    }


def _summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = pd.DataFrame(
        [
            {
                "method": "FROST-DOA",
                "mean_rmse_deg": frame["set_rmse_deg"].mean(),
                "median_rmse_deg": frame["set_rmse_deg"].median(),
                "mean_mae_deg": frame["set_mae_deg"].mean(),
                "source_number_accuracy_pct": 100.0 * frame["p_correct"].mean(),
                "mean_source_number_abs_error": frame["p_abs_error"].mean(),
                "outlier_rate_gt_5deg_pct": 100.0 * frame["outlier_gt_5deg"].mean(),
                "mean_runtime_ms": frame["runtime_ms"].mean(),
                "n": len(frame),
            }
        ]
    )
    scenario = (
        frame.groupby("scenario", sort=True)
        .agg(
            mean_rmse_deg=("set_rmse_deg", "mean"),
            median_rmse_deg=("set_rmse_deg", "median"),
            mean_mae_deg=("set_mae_deg", "mean"),
            source_number_accuracy_pct=("p_correct", lambda x: 100.0 * x.mean()),
            outlier_rate_gt_5deg_pct=("outlier_gt_5deg", lambda x: 100.0 * x.mean()),
            mean_runtime_ms=("runtime_ms", "mean"),
            n=("sample_id", "count"),
        )
        .reset_index()
    )
    scenario.insert(1, "method", "FROST-DOA")
    return overall, scenario


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/frost_doa"))
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--samples-per-scenario", type=int, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial_dir = args.output_dir / "scenario_parts"
    partial_dir.mkdir(parents=True, exist_ok=True)
    scenarios = _scenario_names(args.fixture)
    print(f"[FROST-DOA] fixture={args.fixture}")
    print(f"[FROST-DOA] scenarios={len(scenarios)}, workers={args.workers}")

    reports: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _run_scenario,
                str(args.fixture),
                scenario,
                str(partial_dir),
                args.samples_per_scenario,
            ): scenario
            for scenario in scenarios
        }
        for future in as_completed(futures):
            report = future.result()
            reports.append(report)
            print(
                f"[done] {report['scenario']}: n={report['n']}, "
                f"RMSE={report['mean_rmse_deg']:.6f}°, time={report['elapsed_s']:.1f}s",
                flush=True,
            )

    frames = [pd.read_csv(partial_dir / f"frost_doa_{scenario}.csv") for scenario in scenarios]
    samples = pd.concat(frames, ignore_index=True)
    samples["scenario_order"] = pd.Categorical(
        samples["scenario"], categories=scenarios, ordered=True
    )
    samples = samples.sort_values(["scenario_order", "sample_id"]).drop(columns="scenario_order")
    overall, scenario_summary = _summarize(samples)
    samples.to_csv(args.output_dir / "frost_doa_sample_scores.csv", index=False)
    overall.to_csv(args.output_dir / "frost_doa_overall_summary.csv", index=False)
    scenario_summary.to_csv(args.output_dir / "frost_doa_scenario_summary.csv", index=False)
    (args.output_dir / "frost_doa_worker_report.json").write_text(
        json.dumps(sorted(reports, key=lambda x: x["scenario"]), indent=2), encoding="utf-8"
    )
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
