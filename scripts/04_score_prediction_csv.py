#!/usr/bin/env python3
"""Score one prediction CSV against the aligned FROST-DOA fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from frost_doa.fixture import iter_fixture
from frost_doa.metrics import penalized_set_metrics


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--penalty-deg", type=float, default=25.0)
    args = parser.parse_args()

    truth = {sample.sample_id: sample for sample in iter_fixture(args.fixture)}
    pred = pd.read_csv(args.predictions)
    required = {"sample_id", "theta_hat_deg_json"}
    missing = sorted(required.difference(pred.columns))
    if missing:
        raise ValueError(f"Prediction CSV is missing columns: {missing}")
    if pred["sample_id"].duplicated().any():
        raise ValueError("Prediction CSV contains duplicate sample_id values")
    if "uses_true_K" in pred and pred["uses_true_K"].map(parse_bool).any():
        raise ValueError("Non-oracle score rejected: uses_true_K=true")
    if "uses_scenario_label" in pred and pred["uses_scenario_label"].map(parse_bool).any():
        raise ValueError("Non-oracle score rejected: uses_scenario_label=true")
    pred_ids = set(pred["sample_id"])
    truth_ids = set(truth)
    if pred_ids != truth_ids:
        raise ValueError(
            f"sample_id mismatch: missing={len(truth_ids-pred_ids)}, extra={len(pred_ids-truth_ids)}"
        )

    rows = []
    for record in pred.to_dict("records"):
        sample = truth[str(record["sample_id"])]
        theta_hat = [float(v) for v in json.loads(record["theta_hat_deg_json"])]
        p_hat = int(record.get("p_hat", record.get("k_hat", len(theta_hat))))
        if p_hat != len(theta_hat):
            raise ValueError(f"p_hat and angle count differ for {record['sample_id']}")
        metric = penalized_set_metrics(theta_hat, sample.true_doas_deg, args.penalty_deg)
        rows.append(
            {
                "sample_id": sample.sample_id,
                "scenario": sample.scenario,
                "method": args.method,
                "p_true": len(sample.true_doas_deg),
                "p_hat": p_hat,
                "theta_true_deg_json": json.dumps([float(v) for v in sample.true_doas_deg]),
                "theta_hat_deg_json": json.dumps(theta_hat),
                "set_rmse_deg": metric.rmse_deg,
                "set_mae_deg": metric.mae_deg,
                "p_correct": int(p_hat == len(sample.true_doas_deg)),
                "p_abs_error": abs(p_hat - len(sample.true_doas_deg)),
                "outlier_gt_5deg": int(metric.rmse_deg > 5.0),
                "runtime_ms": record.get("runtime_ms", ""),
                "implementation_id": record.get("implementation_id", ""),
            }
        )
    frame = pd.DataFrame(rows)
    overall = pd.DataFrame(
        [
            {
                "method": args.method,
                "mean_rmse_deg": frame.set_rmse_deg.mean(),
                "median_rmse_deg": frame.set_rmse_deg.median(),
                "mean_mae_deg": frame.set_mae_deg.mean(),
                "source_number_accuracy_pct": 100.0 * frame.p_correct.mean(),
                "mean_source_number_abs_error": frame.p_abs_error.mean(),
                "outlier_rate_gt_5deg_pct": 100.0 * frame.outlier_gt_5deg.mean(),
                "n": len(frame),
            }
        ]
    )
    scenario = (
        frame.groupby("scenario", sort=True)
        .agg(
            mean_rmse_deg=("set_rmse_deg", "mean"),
            median_rmse_deg=("set_rmse_deg", "median"),
            source_number_accuracy_pct=("p_correct", lambda x: 100.0 * x.mean()),
            outlier_rate_gt_5deg_pct=("outlier_gt_5deg", lambda x: 100.0 * x.mean()),
            n=("sample_id", "count"),
        )
        .reset_index()
    )
    scenario.insert(1, "method", args.method)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    slug = args.method.lower().replace(" ", "_").replace("+", "plus").replace("-", "_")
    frame.to_csv(args.output_dir / f"{slug}_sample_scores.csv", index=False)
    overall.to_csv(args.output_dir / f"{slug}_overall_summary.csv", index=False)
    scenario.to_csv(args.output_dir / f"{slug}_scenario_summary.csv", index=False)
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
