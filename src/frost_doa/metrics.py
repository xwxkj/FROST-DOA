"""Common unknown-cardinality DOA metrics used in the manuscript and code."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class SetMetrics:
    """Permutation-matched set errors in degrees."""

    rmse_deg: float
    mae_deg: float


def cardinality_aware_set_error(
    estimate_deg: Iterable[float],
    truth_deg: Iterable[float],
    penalty_deg: float = 25.0,
) -> Tuple[float, float]:
    """Return permutation-matched RMSE and MAE with cardinality penalty.

    Matched errors are augmented by ``penalty_deg`` for each unmatched source.
    The squared sum is normalized by the larger set cardinality before taking
    the square root. This is the exact metric used for the aligned reference.
    """
    estimate = np.asarray(list(estimate_deg), dtype=float)
    truth = np.asarray(list(truth_deg), dtype=float)
    n_est, n_true = len(estimate), len(truth)
    if n_est == 0 and n_true == 0:
        return 0.0, 0.0
    if n_est == 0 or n_true == 0:
        return float(penalty_deg), float(penalty_deg)

    best_squared = math.inf
    best_absolute = math.inf
    if n_est <= n_true:
        for indices in itertools.permutations(range(n_true), n_est):
            difference = estimate - truth[list(indices)]
            best_squared = min(
                best_squared,
                float(np.sum(difference**2)) + (n_true - n_est) * penalty_deg**2,
            )
            best_absolute = min(
                best_absolute,
                float(np.sum(np.abs(difference))) + (n_true - n_est) * penalty_deg,
            )
    else:
        for indices in itertools.permutations(range(n_est), n_true):
            difference = estimate[list(indices)] - truth
            best_squared = min(
                best_squared,
                float(np.sum(difference**2)) + (n_est - n_true) * penalty_deg**2,
            )
            best_absolute = min(
                best_absolute,
                float(np.sum(np.abs(difference))) + (n_est - n_true) * penalty_deg,
            )
    denominator = max(n_est, n_true)
    return float(np.sqrt(best_squared / denominator)), float(best_absolute / denominator)


def penalized_set_metrics(
    estimate_deg: Iterable[float],
    truth_deg: Iterable[float],
    penalty_deg: float = 25.0,
) -> SetMetrics:
    """Return a named metric object for terminal benchmark scripts."""
    rmse, mae = cardinality_aware_set_error(estimate_deg, truth_deg, penalty_deg)
    return SetMetrics(rmse_deg=rmse, mae_deg=mae)
