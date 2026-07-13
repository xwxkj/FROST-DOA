"""Public FROST-DOA estimator.

The public API deliberately hides historical development-version labels.  The
numerical path is the frozen estimator used to create the aligned MC100
reference results.  It uses only the observed complex snapshots and nominal ULA
geometry at inference time; neither the true source number nor a scenario label
is consumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from ._frozen_core import adaptive_controller as controller
from ._frozen_core import robust_selector


@dataclass(frozen=True)
class DOAEstimate:
    """One unordered direction-of-arrival estimate."""

    doas_deg: np.ndarray
    num_sources: int
    branch: str
    expert: str
    diagnostics: Dict[str, Any]


class FROSTDOAEstimator:
    """Fuzzy rule-based adaptive expert selector for robust DOA estimation.

    Parameters
    ----------
    grid_deg:
        Search grid in degrees. The frozen paper configuration uses
        ``np.linspace(-60, 60, 241)``.
    max_sources:
        Maximum candidate source number. The paper configuration uses four.

    Notes
    -----
    The two packaged linear rankers are fixed at inference time.  Fuzzy rule
    activations enter candidate features and fuzzy-spectrum candidates, while
    observation-only physics safeguards control high-risk regimes.
    """

    def __init__(self, grid_deg: np.ndarray | None = None, max_sources: int = 4):
        self.grid_deg = (
            np.linspace(-60.0, 60.0, 241) if grid_deg is None else np.asarray(grid_deg, dtype=float)
        )
        if max_sources != 4:
            raise ValueError(
                "The frozen release is calibrated for max_sources=4. "
                "Retraining is required for a different value."
            )
        self.max_sources = int(max_sources)

    def estimate(self, snapshots: np.ndarray) -> DOAEstimate:
        """Estimate an unordered DOA set from an ``M x T`` complex matrix."""
        y = np.asarray(snapshots)
        if y.ndim != 2 or not np.iscomplexobj(y):
            raise ValueError("snapshots must be a two-dimensional complex array")
        if y.shape[0] != 8:
            raise ValueError("The frozen paper model is calibrated for an eight-sensor ULA.")

        diag = controller.diagnostics(y)
        base = controller.choose_v10_cached(y)
        robust = controller.choose_v11_cached(y)
        stage_1 = controller.choose_v12_fresh(y, diag, base, robust)
        stage_2 = controller.choose_v13_fresh(y, diag, base, robust, stage_1)
        adaptive = controller.choose_v14(y, diag, base, robust, stage_1, stage_2)

        # Low-snapshot controller used immediately before the final correction.
        if bool(diag["fewshot_like"]) and diag["toe_max"] >= 0.65 and diag["k_mdl"] >= 3:
            est, p_hat = robust_selector.estimate_robust_baseline(
                y, "SoftTrim-SOMP-MDL", self.grid_deg, self.max_sources
            )
            prefinal = {
                "est": est,
                "k": int(p_hat),
                "mode": "few-snapshot-hard-mixed-softtrim-mdl",
                "expert": "SoftTrim-SOMP",
            }
        elif (
            bool(diag["fewshot_like"])
            and diag["k_gap"] >= 3
            and diag["k_mdl"] <= 1
            and diag["toe_max"] < 0.65
        ):
            prefinal = dict(base)
            prefinal["mode"] = "few-snapshot-high-order-preserving-base-ranker"
        else:
            prefinal = dict(adaptive)

        # Final hard-mixed source-number correction used by the released model.
        if bool(diag["fewshot_like"]) and diag["toe_max"] >= 0.65 and diag["k_mdl"] >= 3:
            p_hat = 2
            est = robust_selector.robust_somp_estimate(y, p_hat, self.grid_deg, mode="softtrim")
            selected = {
                "est": est,
                "k": p_hat,
                "mode": "hard-mixed-softtrim-source-number-correction",
                "expert": "SoftTrim-SOMP@P2",
            }
        else:
            selected = prefinal

        return DOAEstimate(
            doas_deg=np.asarray(selected["est"], dtype=float),
            num_sources=int(selected["k"]),
            branch=str(selected.get("mode", "adaptive-expert-selection")),
            expert=str(selected.get("expert", "unknown")),
            diagnostics=dict(diag),
        )
