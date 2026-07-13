#!/usr/bin/env python3
"""Run FROST-DOA on the aligned fixture and write predictions and sample scores."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from frost_doa import FROSTDOAEstimator, cardinality_aware_set_error
from frost_doa.fixture import iter_fixture


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default="data/test_fixture_mc100_aligned.h5")
    parser.add_argument("--output-dir", default="results/frost_doa")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    estimator = FROSTDOAEstimator()
    pred_rows, score_rows, decision_rows = [], [], []

    samples = list(iter_fixture(args.fixture))
    for index, sample in enumerate(samples, start=1):
        start = time.perf_counter()
        estimate = estimator.estimate(sample.snapshots)
        runtime_ms = 1000.0 * (time.perf_counter() - start)
        rmse, mae = cardinality_aware_set_error(
            estimate.doas_deg, sample.true_doas_deg, penalty_deg=25.0
        )
        p_true = len(sample.true_doas_deg)
        p_hat = estimate.num_sources
        pred_rows.append(
            {
                "sample_id": sample.sample_id,
                "method": "FROST-DOA",
                "p_hat": p_hat,
                "theta_hat_deg_json": json.dumps([float(x) for x in estimate.doas_deg]),
                "runtime_ms": runtime_ms,
                "implementation_id": "FROST-DOA-release-1.0-frozen",
                "uses_true_source_number": False,
                "uses_scenario_label": False,
            }
        )
        score_rows.append(
            {
                "sample_id": sample.sample_id,
                "scenario": sample.scenario,
                "true_P": p_true,
                "method": "FROST-DOA",
                "display_name": "FROST-DOA",
                "set_rmse": rmse,
                "set_mae": mae,
                "p_hat": p_hat,
                "p_acc": float(p_hat == p_true),
                "p_abs_err": abs(p_hat - p_true),
                "outlier_5deg": float(rmse > 5.0),
            }
        )
        decision_rows.append(
            {
                "sample_id": sample.sample_id,
                "scenario": sample.scenario,
                "p_hat": p_hat,
                "branch": estimate.branch,
                "expert": estimate.expert,
                **{f"diag_{key}": value for key, value in estimate.diagnostics.items()},
            }
        )
        if index % 100 == 0:
            print(f"Processed {index}/{len(samples)} samples", flush=True)

    pred = pd.DataFrame(pred_rows)
    scores = pd.DataFrame(score_rows)
    decisions = pd.DataFrame(decision_rows)
    pred.to_csv(out / "predictions.csv", index=False)
    scores.to_csv(out / "sample_scores.csv", index=False)
    decisions.to_csv(out / "decisions.csv", index=False)

    overall = pd.DataFrame(
        [
            {
                "method": "FROST-DOA",
                "mean_rmse": scores["set_rmse"].mean(),
                "median_rmse": scores["set_rmse"].median(),
                "mean_mae": scores["set_mae"].mean(),
                "source_number_accuracy_pct": 100.0 * scores["p_acc"].mean(),
                "mean_source_number_abs_error": scores["p_abs_err"].mean(),
                "outlier_gt5_pct": 100.0 * scores["outlier_5deg"].mean(),
                "n": len(scores),
            }
        ]
    )
    scenario = (
        scores.groupby("scenario", sort=True)
        .agg(
            mean_rmse=("set_rmse", "mean"),
            median_rmse=("set_rmse", "median"),
            source_number_accuracy_pct=("p_acc", lambda x: 100.0 * x.mean()),
            n=("sample_id", "count"),
        )
        .reset_index()
    )
    overall.to_csv(out / "overall_summary.csv", index=False)
    scenario.to_csv(out / "scenario_summary.csv", index=False)
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
