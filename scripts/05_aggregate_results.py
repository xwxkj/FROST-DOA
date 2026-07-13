#!/usr/bin/env python3
"""Aggregate aligned sample scores into manuscript tables and paired statistics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FROST_NAME = "FROST-DOA"


def _read_score_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    aliases = {
        "set_rmse": "set_rmse_deg",
        "set_mae": "set_mae_deg",
        "p_acc": "p_correct",
        "outlier_5deg": "outlier_gt_5deg",
        "true_P": "p_true",
    }
    frame = frame.rename(columns={k: v for k, v in aliases.items() if k in frame.columns})
    required = {"sample_id", "scenario", "method", "set_rmse_deg", "p_correct"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if "set_mae_deg" not in frame:
        frame["set_mae_deg"] = np.nan
    if "p_abs_error" not in frame:
        if {"p_true", "p_hat"}.issubset(frame.columns):
            frame["p_abs_error"] = (frame["p_true"] - frame["p_hat"]).abs()
        else:
            frame["p_abs_error"] = np.nan
    if "outlier_gt_5deg" not in frame:
        frame["outlier_gt_5deg"] = (frame["set_rmse_deg"] > 5.0).astype(int)
    if "runtime_ms" not in frame:
        frame["runtime_ms"] = np.nan
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-file", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/final"))
    args = parser.parse_args()

    frames = [_read_score_file(path) for path in args.score_file]
    all_scores = pd.concat(frames, ignore_index=True)
    if all_scores.duplicated(["method", "sample_id"]).any():
        duplicates = all_scores[all_scores.duplicated(["method", "sample_id"], keep=False)]
        raise ValueError(f"Duplicate method/sample pairs detected:\n{duplicates.head()}")

    sample_counts = all_scores.groupby("method")["sample_id"].nunique()
    if sample_counts.nunique() != 1:
        raise ValueError(f"Methods do not contain equal sample counts:\n{sample_counts}")

    overall = (
        all_scores.groupby("method", sort=False)
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
        all_scores.groupby(["scenario", "method"], sort=True)
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

    if FROST_NAME not in set(all_scores["method"]):
        raise ValueError("FROST-DOA sample scores are required for paired statistics")
    frost = all_scores.loc[
        all_scores.method == FROST_NAME, ["sample_id", "scenario", "set_rmse_deg"]
    ].rename(columns={"set_rmse_deg": "frost_rmse_deg"})
    paired_rows = []
    paired_samples = []
    for method in overall["method"]:
        if method == FROST_NAME:
            continue
        other = all_scores.loc[all_scores.method == method, ["sample_id", "set_rmse_deg"]].rename(
            columns={"set_rmse_deg": "comparison_rmse_deg"}
        )
        merged = frost.merge(other, on="sample_id", validate="one_to_one")
        merged["comparison_method"] = method
        merged["delta_comparison_minus_frost_deg"] = (
            merged["comparison_rmse_deg"] - merged["frost_rmse_deg"]
        )
        tolerance = 1e-12
        method_wins = merged["comparison_rmse_deg"] + tolerance < merged["frost_rmse_deg"]
        frost_wins = merged["frost_rmse_deg"] + tolerance < merged["comparison_rmse_deg"]
        ties = ~(method_wins | frost_wins)
        paired_rows.append(
            {
                "comparison_method": method,
                "mean_delta_comparison_minus_frost_deg": merged[
                    "delta_comparison_minus_frost_deg"
                ].mean(),
                "median_delta_comparison_minus_frost_deg": merged[
                    "delta_comparison_minus_frost_deg"
                ].median(),
                "comparison_win_rate_pct": 100.0 * method_wins.mean(),
                "frost_win_rate_pct": 100.0 * frost_wins.mean(),
                "tie_rate_pct": 100.0 * ties.mean(),
                "n": len(merged),
            }
        )
        paired_samples.append(merged)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_scores.to_csv(args.output_dir / "all_sample_scores.csv", index=False)
    overall.to_csv(args.output_dir / "overall_summary.csv", index=False)
    scenario.to_csv(args.output_dir / "scenario_summary.csv", index=False)
    pd.DataFrame(paired_rows).sort_values("mean_delta_comparison_minus_frost_deg").to_csv(
        args.output_dir / "paired_vs_frost_doa.csv", index=False
    )
    if paired_samples:
        pd.concat(paired_samples, ignore_index=True).to_csv(
            args.output_dir / "paired_sample_deltas.csv", index=False
        )
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
