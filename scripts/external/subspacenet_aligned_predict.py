#!/usr/bin/env python3
"""
SubspaceNet Step 3B: non-oracle unknown-K wrapper using Step-2B best official model bank.

This script is intentionally conservative and transparent:

1. It loads the official SubspaceNet model class and official
   create_autocorrelation_tensor function from the cloned SubspaceNet repo.
2. It reads results_subspacenet_step2b/subspacenet_step2b_variant_summary.csv.
3. For each K=1,2,3, it selects the variant with the lowest validation RMSE.
   This is allowed because validation labels belong to training/model selection.
4. It loads the selected fixed-K official checkpoints.
5. It builds candidate DOA sets for K=1,2,3 on validation and test samples.
6. It calibrates two non-oracle K selectors on validation:
      a) MDL-K clipped to {1,2,3};
      b) residual-K selector: min normalized LS residual + alpha*K.
7. It selects the better selector by validation RMSE and applies it to test.

Fairness:
- Test-time selection does not read true K.
- Test-time selection does not read scenario labels.
- Final prediction CSV sets uses_true_K=false and uses_scenario_label=false.
- The Oracle-K Step2B diagnostic remains separate and is not used for test labels.
"""

from __future__ import annotations
import os, sys, json, math, datetime, subprocess, traceback
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import h5py
from scipy.optimize import linear_sum_assignment

try:
    import torch
except Exception as e:
    raise RuntimeError(f"PyTorch import failed: {e}")

PHASEC_ROOT = Path(os.environ.get("PHASEC_ROOT", ".")).resolve()
RELEASE_ROOT = Path(os.environ.get("FROST_RELEASE_ROOT", ".")).resolve()
ROOT = PHASEC_ROOT
REPO = PHASEC_ROOT / "official_repos" / "SubspaceNet"
STEP2B = PHASEC_ROOT / "results_subspacenet_step2b"
OUT = RELEASE_ROOT / "results" / "external" / "subspacenet_run"
RAW = RELEASE_ROOT / "results" / "external" / "predictions"
VAL = RELEASE_ROOT / "results" / "external" / "validated"
LOG = RELEASE_ROOT / "logs" / "subspacenet_aligned"
for p in [OUT, RAW, VAL, LOG]:
    p.mkdir(parents=True, exist_ok=True)

CFG = {
    "batch_size": int(os.environ.get("SUBSPACE_STEP3B_BATCH", "32")),
    "device": os.environ.get("SUBSPACE_STEP3B_DEVICE", "cpu"),
    "seed": int(os.environ.get("SUBSPACE_STEP3B_SEED", "20260709")),
    "penalty_deg": float(os.environ.get("SUBSPACE_STEP3B_CARDINALITY_PENALTY", "25.0")),
    "alpha_grid": [
        float(x)
        for x in os.environ.get(
            "SUBSPACE_STEP3B_ALPHA_GRID", "0,0.01,0.03,0.05,0.08,0.1,0.15,0.2,0.3,0.5"
        ).split(",")
    ],
}
np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])

REPORT: Dict = {
    "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    "status": "started",
    "config": CFG,
    "errors": [],
    "warnings": [],
}


def write_report(status: str):
    REPORT["status"] = status
    (OUT / "subspacenet_step3b_unknownK_report.json").write_text(
        json.dumps(REPORT, indent=2, default=str)
    )
    lines = ["# SubspaceNet Step 3B unknown-K wrapper report\n", f"Status: `{status}`\n"]
    for k in [
        "repo_commit",
        "chosen_selector",
        "best_alpha",
        "val_rmse_mdl",
        "val_rmse_residual",
        "test_rmse",
        "test_k_acc",
        "n_test",
    ]:
        if k in REPORT:
            lines.append(f"- {k}: `{REPORT[k]}`")
    if REPORT.get("selected_variants"):
        lines.append("\n## Selected Step2B variants per K")
        for k, v in REPORT["selected_variants"].items():
            lines.append(
                f"- K={k}: tau={v['tau']}, diff={v['diff_method']}, val_rmse={v['val_rmse']}, test_oracle_rmse={v['test_rmse']}"
            )
    if REPORT.get("errors"):
        lines.append("\n## Errors")
        for e in REPORT["errors"]:
            lines.append(f"- `{e}`")
    if REPORT.get("warnings"):
        lines.append("\n## Warnings")
        for w in REPORT["warnings"]:
            lines.append(f"- `{w}`")
    lines.append("\n## Fairness")
    lines.append(
        "The final test CSV uses neither true K nor scenario labels. Validation labels are used only to calibrate the K-selector."
    )
    (OUT / "subspacenet_step3b_unknownK_report.md").write_text("\n".join(lines))


try:
    if not REPO.exists():
        raise FileNotFoundError(f"SubspaceNet repo missing: {REPO}")
    sys.path.insert(0, str(REPO))
    from src.data_handler import create_autocorrelation_tensor
    from src.models import SubspaceNet

    cp = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True, capture_output=True
    )
    REPORT["repo_commit"] = cp.stdout.strip() if cp.returncode == 0 else "unknown"
    REPORT["torch_version"] = torch.__version__
except Exception as e:
    REPORT["errors"].append("official_import_failed: " + repr(e))
    REPORT["traceback"] = traceback.format_exc()
    write_report("failed_import")
    sys.exit(0)


# ----------------------- data utilities -----------------------
def make_sample_id(scenario: str, i: int) -> str:
    return f"{scenario}#{i:04d}"


def load_split(split: str) -> List[Dict]:
    path = (
        (PHASEC_ROOT / "data" / "trainval_fixture.h5")
        if split in ["train", "val"]
        else Path(
            os.environ.get(
                "SUBSPACENET_TEST_FIXTURE", RELEASE_ROOT / "data" / "test_fixture_mc100_aligned.h5"
            )
        )
    )
    samples = []
    with h5py.File(path, "r") as f:
        base = f[split] if split in ["train", "val"] else f["scenarios"]
        for scenario in sorted(base.keys()):
            g = base[scenario]
            Yr = np.asarray(g["Y_real"])
            Yi = np.asarray(g["Y_imag"])
            th = np.asarray(g["theta_deg_padded"])
            mask = np.asarray(g["theta_mask"])
            for i in range(Yr.shape[0]):
                m = mask[i].astype(bool)
                samples.append(
                    {
                        "sample_id": make_sample_id(scenario, i),
                        "scenario": scenario,
                        "split": split,
                        "Y": Yr[i].astype(np.float32) + 1j * Yi[i].astype(np.float32),
                        "theta_deg": th[i][m].astype(float),
                        "K": int(m.sum()),
                    }
                )
    return samples


def steering_matrix(theta_deg: np.ndarray, M: int) -> np.ndarray:
    theta = np.deg2rad(np.asarray(theta_deg, dtype=float))
    m = np.arange(M)[:, None]
    # ULA half-wavelength broadside convention.
    return np.exp(1j * np.pi * m * np.sin(theta)[None, :])


def normalized_ls_residual(Y: np.ndarray, theta_deg: np.ndarray) -> float:
    if len(theta_deg) == 0:
        return 1.0
    A = steering_matrix(np.asarray(theta_deg), Y.shape[0])
    try:
        S = np.linalg.pinv(A) @ Y
        R = Y - A @ S
        return float(np.linalg.norm(R, "fro") / (np.linalg.norm(Y, "fro") + 1e-12))
    except Exception:
        return 1.0


def mdl_k(Y: np.ndarray, kmax: int = 3) -> int:
    M, T = Y.shape
    R = (Y @ Y.conj().T) / max(T, 1)
    vals = np.sort(np.real(np.linalg.eigvalsh(R)))[::-1]
    vals = np.maximum(vals, 1e-12)
    scores = []
    for k in range(0, min(kmax, M - 1) + 1):
        noise = vals[k:]
        m = len(noise)
        if m <= 0:
            scores.append(np.inf)
            continue
        gm = np.exp(np.mean(np.log(noise)))
        am = np.mean(noise)
        ratio = max(gm / am, 1e-12)
        score = -T * m * np.log(ratio) + 0.5 * k * (2 * M - k) * np.log(max(T, 2))
        scores.append(score)
    kh = int(np.argmin(scores))
    return int(np.clip(kh, 1, kmax))


def set_rmse_deg(pred, true, penalty=None):
    if penalty is None:
        penalty = CFG["penalty_deg"]
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    m, n = len(pred), len(true)
    if m == 0 and n == 0:
        return 0.0
    if m == 0 or n == 0:
        return penalty * abs(m - n)
    C = np.abs(pred[:, None] - true[None, :])
    row, col = linear_sum_assignment(C)
    sq = float(np.sum((pred[row] - true[col]) ** 2))
    if m != n:
        sq += (penalty**2) * abs(m - n)
    return math.sqrt(sq / max(m, n))


def rx_tau(Y: np.ndarray, tau: int) -> torch.Tensor:
    X = torch.tensor(Y, dtype=torch.complex64)
    return create_autocorrelation_tensor(X, tau=tau).to(torch.float32)


# ----------------------- load best Step2B variants -----------------------
try:
    var_path = STEP2B / "subspacenet_step2b_variant_summary.csv"
    if not var_path.exists():
        raise FileNotFoundError(f"missing {var_path}")
    var = pd.read_csv(var_path)
    selected = {}
    for K in [1, 2, 3]:
        sub = var[(var["K"] == K) & (var["status"] == "ok")]
        if sub.empty:
            raise RuntimeError(f"no ok variant for K={K}")
        row = sub.loc[sub["val_rmse"].idxmin()].to_dict()
        ckpt = (
            ROOT
            / "results_subspacenet_step2b"
            / "checkpoints"
            / f"SubspaceNet_K{K}_tau{int(row['tau'])}_{row['diff_method']}.pt"
        )
        if not ckpt.exists():
            raise FileNotFoundError(f"checkpoint not found for K={K}: {ckpt}")
        row["checkpoint_local"] = str(ckpt)
        selected[K] = row
    REPORT["selected_variants"] = {
        str(k): {
            kk: (float(vv) if isinstance(vv, (np.floating, float)) else vv)
            for kk, vv in row.items()
            if kk in ["tau", "diff_method", "val_rmse", "test_rmse", "checkpoint_local"]
        }
        for k, row in selected.items()
    }
except Exception as e:
    REPORT["errors"].append("select_variants_failed: " + repr(e))
    REPORT["traceback"] = traceback.format_exc()
    write_report("failed_select_variants")
    sys.exit(0)


# ----------------------- load models -----------------------
def load_model(K: int):
    row = selected[K]
    tau = int(row["tau"])
    diff = str(row["diff_method"])
    model = SubspaceNet(tau=tau, M=K, diff_method=diff).to(CFG["device"])
    ckpt = torch.load(row["checkpoint_local"], map_location=CFG["device"])
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model, tau, diff


try:
    models = {K: load_model(K) for K in [1, 2, 3]}
except Exception as e:
    REPORT["errors"].append("load_models_failed: " + repr(e))
    REPORT["traceback"] = traceback.format_exc()
    write_report("failed_load_models")
    sys.exit(0)


def predict_one(sample: Dict, K: int) -> Dict:
    model, tau, diff = models[K]
    with torch.no_grad():
        x = rx_tau(sample["Y"], tau).unsqueeze(0).to(CFG["device"])
        out = model(x)[0]
        out = torch.nan_to_num(out, nan=0.0, posinf=np.pi / 2, neginf=-np.pi / 2)
        pred = np.rad2deg(out.detach().cpu().numpy()[0])
        pred = pred[np.isfinite(pred)]
        if len(pred) < K:
            pred = np.pad(pred, (0, K - len(pred)), constant_values=0.0)
        pred = np.clip(pred[:K], -90, 90)
    return {
        "K": K,
        "theta": pred,
        "residual": normalized_ls_residual(sample["Y"], pred),
        "tau": tau,
        "diff_method": diff,
    }


def build_candidates(samples: List[Dict]) -> Dict[str, List[Dict]]:
    cand = {}
    for idx, s in enumerate(samples):
        cs = []
        for K in [1, 2, 3]:
            cs.append(predict_one(s, K))
        cand[s["sample_id"]] = cs
        if (idx + 1) % 200 == 0:
            print(f"predicted {idx+1}/{len(samples)} samples", flush=True)
    return cand


def choose_by_mdl(sample, candidates):
    k = mdl_k(sample["Y"], 3)
    return next(c for c in candidates if c["K"] == k)


def choose_by_residual(candidates, alpha: float):
    return min(candidates, key=lambda c: c["residual"] + alpha * c["K"])


def eval_selector(samples, cand, mode: str, alpha: float = 0.0):
    rows = []
    rmses = []
    kacc = []
    for s in samples:
        cs = cand[s["sample_id"]]
        if mode == "mdl":
            choice = choose_by_mdl(s, cs)
        else:
            choice = choose_by_residual(cs, alpha)
        rm = set_rmse_deg(choice["theta"], s["theta_deg"])
        rmses.append(rm)
        kacc.append(float(choice["K"] == s["K"]))
        rows.append(
            {
                "sample_id": s["sample_id"],
                "scenario": s["scenario"],
                "true_K": s["K"],
                "k_hat": choice["K"],
                "theta_true_deg_json": json.dumps([float(x) for x in s["theta_deg"]]),
                "theta_hat_deg_json": json.dumps([float(x) for x in choice["theta"]]),
                "set_rmse": float(rm),
                "k_acc": float(choice["K"] == s["K"]),
                "selector_mode": mode,
                "selector_alpha": alpha,
                "candidate_residual": float(choice["residual"]),
                "tau": choice["tau"],
                "diff_method": choice["diff_method"],
            }
        )
    return float(np.mean(rmses)), float(np.mean(kacc)), pd.DataFrame(rows)


try:
    val_samples = load_split("val")
    test_samples = load_split("test")
    print("Building validation candidates...")
    val_cand = build_candidates(val_samples)
    print("Building test candidates...")
    test_cand = build_candidates(test_samples)
    val_mdl_rmse, val_mdl_kacc, _ = eval_selector(val_samples, val_cand, "mdl")
    best = {"mode": "mdl", "alpha": None, "val_rmse": val_mdl_rmse, "val_k_acc": val_mdl_kacc}
    alpha_rows = []
    for a in CFG["alpha_grid"]:
        rm, ka, _ = eval_selector(val_samples, val_cand, "residual", a)
        alpha_rows.append({"selector": "residual", "alpha": a, "val_rmse": rm, "val_k_acc": ka})
        if rm < best["val_rmse"]:
            best = {"mode": "residual", "alpha": a, "val_rmse": rm, "val_k_acc": ka}
    sel_df = pd.DataFrame(
        [{"selector": "mdl", "alpha": np.nan, "val_rmse": val_mdl_rmse, "val_k_acc": val_mdl_kacc}]
        + alpha_rows
    )
    sel_df.to_csv(OUT / "subspacenet_step3b_selector_validation.csv", index=False)
    test_rmse, test_kacc, test_rows = eval_selector(
        test_samples, test_cand, best["mode"], best["alpha"] or 0.0
    )
    REPORT["chosen_selector"] = best["mode"]
    REPORT["best_alpha"] = best["alpha"]
    REPORT["selector_val_rmse"] = best["val_rmse"]
    REPORT["selector_val_k_acc"] = best["val_k_acc"]
    REPORT["test_rmse"] = test_rmse
    REPORT["test_k_acc"] = test_kacc
    REPORT["n_test"] = len(test_rows)
    # final prediction CSV
    pred = test_rows.copy()
    pred["method"] = "SubspaceNet + residual selector"
    pred["implementation_id"] = pred.apply(
        lambda r: f"official_{REPORT['repo_commit']}_Step3B_unknownK_{r['selector_mode']}_K{r['k_hat']}_tau{r['tau']}_{r['diff_method']}",
        axis=1,
    )
    pred["runtime_ms"] = 0.0
    pred["uses_true_K"] = False
    pred["uses_scenario_label"] = False
    pred["notes"] = (
        "SubspaceNet official Step3B unknown-K wrapper; K selected without true K or scenario labels."
    )
    cols = [
        "sample_id",
        "method",
        "k_hat",
        "theta_hat_deg_json",
        "runtime_ms",
        "implementation_id",
        "uses_true_K",
        "uses_scenario_label",
        "notes",
    ]
    pred[cols].to_csv(RAW / "SubspaceNet.csv", index=False)
    pred[cols].to_csv(VAL / "SubspaceNet.csv", index=False)
    # scores summary
    test_rows.to_csv(OUT / "SubspaceNet_Step3B_unknownK_sample_scores.csv", index=False)
    overall = pd.DataFrame(
        [
            {
                "method": "SubspaceNet + residual selector",
                "mean_rmse": test_rmse,
                "k_acc": test_kacc,
                "n": len(test_rows),
                "chosen_selector": best["mode"],
                "alpha": best["alpha"],
            }
        ]
    )
    overall.to_csv(OUT / "SubspaceNet_Step3B_unknownK_overall_summary.csv", index=False)
    scenario = (
        test_rows.groupby("scenario")
        .agg(mean_rmse=("set_rmse", "mean"), k_acc=("k_acc", "mean"), n=("sample_id", "count"))
        .reset_index()
    )
    scenario.to_csv(OUT / "SubspaceNet_Step3B_unknownK_scenario_summary.csv", index=False)
    # compare with FROST reference if available
    frost = RELEASE_ROOT / "reference" / "frost_doa_frozen_sample_scores.csv"
    if frost.exists():
        f = pd.read_csv(frost)
        merged = (
            test_rows[["sample_id", "scenario", "set_rmse"]]
            .rename(columns={"set_rmse": "subspacenet"})
            .merge(
                f[["sample_id", "set_rmse_deg"]].rename(columns={"set_rmse_deg": "frost_doa"}),
                on="sample_id",
            )
        )
        merged["delta_subspacenet_minus_frost"] = merged["subspacenet"] - merged["frost_doa"]
        merged["subspacenet_wins"] = merged["subspacenet"] < merged["frost_doa"]
        merged.to_csv(OUT / "SubspaceNet_Step3B_paired_vs_FROST_DOA.csv", index=False)
        REPORT["paired_mean_delta_vs_frost"] = float(merged["delta_subspacenet_minus_frost"].mean())
        REPORT["subspacenet_win_rate_vs_frost"] = float(merged["subspacenet_wins"].mean())
except Exception as e:
    REPORT["errors"].append("run_failed: " + repr(e))
    REPORT["traceback"] = traceback.format_exc()
    write_report("failed_run")
    sys.exit(0)

write_report("pass_unknownK_wrapper")
print(json.dumps(REPORT, indent=2, default=str))
