"""DA-MUSIC Step 4: non-oracle unknown-K prediction on the frozen fixture.

Purpose
-------
This script converts the Step 3B K-specific DA-MUSIC diagnostic models into a
non-oracle benchmark entry.  It does not use true K or scenario labels at test
time.  Instead, it estimates source number from the observed sample covariance
with a standard MDL rule and routes each sample to the corresponding K-specific
DA-MUSIC model.

Algorithmic steps
-----------------
1. Read each complex array observation Y from the frozen test HDF5 fixture.
2. Estimate K_hat from Y using eigenvalue MDL, clipped to K=1,2,3.
3. Convert Y to official DA-MUSIC input X=[Re(Y); Im(Y)] with shape 2M x 200.
4. Load the official DA-MUSIC TensorFlow model builder and Step 3B weights.
5. Predict K_hat angles, convert radians to degrees, sort, and write DA-MUSIC.csv.

Fairness constraints
--------------------
- uses_true_K is false.
- uses_scenario_label is false.
- The scenario name is used only to form sample_id while iterating the fixture;
  it is not used for model selection or prediction.
"""

from __future__ import annotations

import csv
import json
import math
import os
import pathlib
import sys
import traceback
from typing import Dict, Tuple

import h5py
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

ROOT = pathlib.Path(os.environ.get("WORKSPACE", ".")).resolve()
REPO = pathlib.Path(
    os.environ.get("DAMUSIC_REPO", ROOT / "official_repos/DA-MUSIC_TVT23")
).resolve()
TEST = pathlib.Path(os.environ.get("DAMUSIC_TEST", ROOT / "data/test_fixture_mc100.h5")).resolve()
OUT = pathlib.Path(os.environ.get("DAMUSIC_OUT", ROOT / "results_damusic_step4")).resolve()
RAW_PRED = pathlib.Path(
    os.environ.get("DAMUSIC_RAW_PRED", ROOT / "external_predictions/raw/DA-MUSIC.csv")
).resolve()
BUILDER_NAME = os.environ.get("DAMUSIC_BUILDER", "deep_aug_MUSIC_isv")
SNAPSHOTS_OFFICIAL = int(os.environ.get("DAMUSIC_SNAPSHOT_TARGET", "200"))
SEED = int(os.environ.get("DAMUSIC_STEP4_SEED", "20260709"))

np.random.seed(SEED)
tf.random.set_seed(SEED)
OUT.mkdir(parents=True, exist_ok=True)
RAW_PRED.parent.mkdir(parents=True, exist_ok=True)

result = {
    "status": "unknown",
    "repo_path": str(REPO),
    "repo_commit": None,
    "builder": BUILDER_NAME,
    "snapshot_target": SNAPSHOTS_OFFICIAL,
    "k_wrapper": "MDL clipped to 1..3",
    "checkpoints": {},
    "k_hat_counts": {},
    "n_predictions": 0,
    "errors": [],
    "fairness": {"uses_true_K": False, "uses_scenario_label": False},
}


def get_commit(repo: pathlib.Path) -> str:
    try:
        head = (repo / ".git" / "HEAD").read_text().strip()
        if head.startswith("ref:"):
            ref = head.split(" ", 1)[1]
            return (repo / ".git" / ref).read_text().strip()
        return head
    except Exception:
        return "unknown"


result["repo_commit"] = get_commit(REPO)

if not REPO.exists():
    result["status"] = "fail_repo_missing"
    result["errors"].append(f"DA-MUSIC repo not found: {REPO}")
    (OUT / "damusic_step4_unknownK_nonoracle.json").write_text(json.dumps(result, indent=2))
    sys.exit(1)
if not TEST.exists():
    result["status"] = "fail_fixture_missing"
    result["errors"].append(f"Test fixture not found: {TEST}")
    (OUT / "damusic_step4_unknownK_nonoracle.json").write_text(json.dumps(result, indent=2))
    sys.exit(1)

sys.path.insert(0, str(REPO))


def pad_or_resample_complex_snapshots(
    y_complex: np.ndarray, target_t: int = SNAPSHOTS_OFFICIAL
) -> np.ndarray:
    """Complex-domain snapshot pad/resample; preserves imaginary part."""
    y_complex = np.asarray(y_complex, dtype=np.complex64)
    m, t = y_complex.shape
    if t == target_t:
        return y_complex
    if t < target_t:
        reps = int(math.ceil(target_t / t))
        return np.tile(y_complex, (1, reps))[:, :target_t]
    idx = np.linspace(0, t - 1, target_t).round().astype(int)
    return y_complex[:, idx]


def y_to_damusic_input(y_complex: np.ndarray) -> np.ndarray:
    y = pad_or_resample_complex_snapshots(y_complex, SNAPSHOTS_OFFICIAL)
    return np.concatenate([y.real, y.imag], axis=0).astype("float32")


def estimate_k_mdl(y_complex: np.ndarray, k_min: int = 1, k_max: int = 3) -> int:
    """Estimate source number by classical MDL from the sample covariance.

    The return value is clipped to [1,3] because Step 3B trained K-specific
    DA-MUSIC models for K=1,2,3 only.
    """
    y = np.asarray(y_complex, dtype=np.complex64)
    m, t = y.shape
    r = (y @ y.conj().T) / max(t, 1)
    evals = np.linalg.eigvalsh(r).real
    evals = np.sort(np.maximum(evals, 1e-10))[::-1]
    max_k = min(k_max, m - 1)
    scores = []
    for k in range(0, max_k + 1):
        noise = evals[k:]
        p = len(noise)
        if p <= 0:
            continue
        gmean = float(np.exp(np.mean(np.log(np.maximum(noise, 1e-10)))))
        amean = float(np.mean(noise)) + 1e-12
        ratio = min(max(gmean / amean, 1e-12), 1.0)
        mdl = -t * p * np.log(ratio) + 0.5 * k * (2 * m - k) * np.log(max(t, 2))
        scores.append((mdl, k))
    if not scores:
        return 1
    k_hat = min(scores)[1]
    return int(np.clip(k_hat, k_min, k_max))


def build_official_model_for_k(k: int):
    """Build official DA-MUSIC model for a fixed number of sources K."""
    tf.keras.backend.clear_session()
    for mod in ["models", "syntheticEx", "utils", "layers", "losses", "regularizers"]:
        if mod in sys.modules:
            del sys.modules[mod]
    import syntheticEx  # type: ignore

    syntheticEx.d = int(k)
    syntheticEx.snapshots = int(SNAPSHOTS_OFFICIAL)
    import models  # type: ignore

    models.d = int(k)
    if hasattr(models, "m"):
        models.n = int(models.m) - int(k)
    if hasattr(models, "snapshots"):
        models.snapshots = int(SNAPSHOTS_OFFICIAL)
    builder = getattr(models, BUILDER_NAME)
    inp, y = builder()
    model = tf.keras.models.Model(inputs=inp, outputs=y, name=f"DA_MUSIC_{BUILDER_NAME}_K{k}_step4")
    # Compile is not needed for inference, but keeps Keras state consistent.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def load_step3b_model(k: int):
    ckpt = ROOT / "results_damusic_step3b" / f"damusic_step3b_{BUILDER_NAME}_K{k}.h5"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing Step3B checkpoint: {ckpt}")
    model = build_official_model_for_k(k)
    try:
        model.load_weights(str(ckpt))
    except Exception:
        # Fallback for full-model H5.  This may be necessary for some Keras versions.
        model = tf.keras.models.load_model(str(ckpt), compile=False)
    result["checkpoints"][str(k)] = str(ckpt)
    return model


def predict_deg(model, x: np.ndarray, k: int) -> list:
    pred = model.predict(x[None, ...], verbose=0)
    pred = np.asarray(pred).reshape(-1)[:k]
    deg = np.sort(pred * 180.0 / np.pi)
    deg = np.clip(deg, -90.0, 90.0)
    return [float(v) for v in deg]


try:
    models_by_k = {k: load_step3b_model(k) for k in [1, 2, 3]}
    counts = {1: 0, 2: 0, 3: 0}
    rows = []
    with h5py.File(TEST, "r") as f:
        scenarios = f["scenarios"]
        for scenario in sorted(scenarios.keys()):
            g = scenarios[scenario]
            yr = np.asarray(g["Y_real"])
            yi = np.asarray(g["Y_imag"])
            n = yr.shape[0]
            for i in range(n):
                y_complex = yr[i].astype("float32") + 1j * yi[i].astype("float32")
                k_hat = estimate_k_mdl(y_complex, 1, 3)
                x = y_to_damusic_input(y_complex)
                theta_hat = predict_deg(models_by_k[k_hat], x, k_hat)
                counts[k_hat] += 1
                rows.append(
                    {
                        "sample_id": f"{scenario}#{i:04d}",
                        "method": "DA-MUSIC + MDL",
                        "k_hat": k_hat,
                        "theta_hat_deg_json": json.dumps(theta_hat),
                        "runtime_ms": "",
                        "implementation_id": f"official_{result['repo_commit']}_{BUILDER_NAME}_MDLK_release_1_0_80epochs",
                        "uses_true_K": "false",
                        "uses_scenario_label": "false",
                        "notes": "Official DA-MUSIC TensorFlow builder with an MDL source-number wrapper; 80-epoch fixture training.",
                    }
                )
    with RAW_PRED.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "sample_id",
                "method",
                "k_hat",
                "theta_hat_deg_json",
                "runtime_ms",
                "implementation_id",
                "uses_true_K",
                "uses_scenario_label",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    result["status"] = "pass_unknownK_nonoracle_prediction"
    result["k_hat_counts"] = {str(k): int(v) for k, v in counts.items()}
    result["n_predictions"] = len(rows)
    result["raw_prediction_csv"] = str(RAW_PRED)
except Exception as e:
    result["status"] = "fail_unknownK_nonoracle_prediction"
    result["errors"].append(repr(e))
    result["traceback"] = traceback.format_exc()
finally:
    (OUT / "damusic_step4_unknownK_nonoracle.json").write_text(json.dumps(result, indent=2))
    report = [
        "# DA-MUSIC Step 4: unknown-K non-oracle wrapper",
        "",
        f"- status: `{result['status']}`",
        f"- builder: `{BUILDER_NAME}`",
        f"- repo_commit: `{result['repo_commit']}`",
        f"- k_wrapper: `{result['k_wrapper']}`",
        f"- n_predictions: `{result['n_predictions']}`",
        f"- k_hat_counts: `{result['k_hat_counts']}`",
        "",
        "## Fairness",
        "- uses_true_K: false",
        "- uses_scenario_label: false",
        "- scenario names are used only as sample IDs, not as features.",
        "",
        "## Notes",
        "The K-specific weights are produced by Step 3B.  This Step 4 wrapper is the first non-oracle DA-MUSIC prediction CSV for the frozen fixture.",
    ]
    if result.get("errors"):
        report += ["", "## Errors", "```", json.dumps(result.get("errors"), indent=2), "```"]
    (OUT / "damusic_step4_report.md").write_text("\n".join(report))
    if result["status"].startswith("fail"):
        sys.exit(1)
