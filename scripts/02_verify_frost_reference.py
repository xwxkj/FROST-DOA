#!/usr/bin/env python3
"""Verify exact sample-level agreement with the frozen aligned reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _normalize_actual(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "set_rmse_deg": "set_rmse",
        "set_mae_deg": "set_mae",
        "p_hat": "k_hat",
        "p_correct": "k_acc",
        "p_abs_error": "k_abs_err",
        "outlier_gt_5deg": "outlier_5deg",
    }
    return frame.rename(columns={k: v for k, v in aliases.items() if k in frame.columns})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--actual", type=Path, default=Path("results/frost_doa/frost_doa_sample_scores.csv")
    )
    parser.add_argument(
        "--reference", type=Path, default=Path("reference/frost_doa_frozen_sample_scores.csv")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("results/frost_doa/reference_verification.json")
    )
    parser.add_argument("--atol", type=float, default=1e-10)
    args = parser.parse_args()

    actual = _normalize_actual(pd.read_csv(args.actual))
    reference = pd.read_csv(args.reference)
    columns = ["sample_id", "set_rmse", "set_mae", "k_hat", "k_acc", "k_abs_err", "outlier_5deg"]
    missing_actual = set(columns).difference(actual.columns)
    missing_reference = set(columns).difference(reference.columns)
    if missing_actual or missing_reference:
        raise ValueError(
            f"Missing columns: actual={sorted(missing_actual)}, reference={sorted(missing_reference)}"
        )
    merged = actual[columns].merge(
        reference[columns],
        on="sample_id",
        suffixes=("_actual", "_reference"),
        validate="one_to_one",
    )
    differences = {}
    passed = len(merged) == len(reference) == len(actual)
    for column in columns[1:]:
        delta = np.abs(merged[f"{column}_actual"] - merged[f"{column}_reference"])
        differences[column] = {
            "max_abs_diff": float(delta.max()),
            "mismatch_count": int((delta > args.atol).sum()),
        }
        passed &= bool((delta <= args.atol).all())
    report = {
        "passed": bool(passed),
        "n_actual": int(len(actual)),
        "n_reference": int(len(reference)),
        "n_matched": int(len(merged)),
        "absolute_tolerance": args.atol,
        "differences": differences,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
