"""Frozen adaptive controller used by the released FROST-DOA estimator.

The controller combines two learned candidate rankers with observation-only
physics safeguards for clean, coherent, few-snapshot, heavy-tailed, mismatch,
and colored-noise regimes.
"""

from __future__ import annotations
import itertools, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from . import signal_processing as fd
from . import base_selector as v10
from . import robust_selector as v11

GRID = np.linspace(-60, 60, 241)
KMAX = 4
PENALTY = 25.0
PFX = Path("results/internal_controller")

# ---- load saved rankers ----
z9 = np.load(Path(__file__).resolve().parent / "models" / "ranker_base.npz")
RANKER10 = v10.CandidateRanker(z9["w"], z9["mean"], z9["std"])
z11 = np.load(Path(__file__).resolve().parent / "models" / "ranker_robust.npz")
RANKER11 = v11.Ranker(z11["w"], z11["mean"], z11["std"])


# ---- utilities ----
def set_rmse(est, true):
    return v10.penalized_set_rmse(est, true, PENALTY)


def set_mae(est, true):
    return v10.set_mae_penalized(est, true, PENALTY)


def somp_with_dictionary(Y, A, K, grid=GRID, min_sep_deg=1.0):
    R = Y.copy()
    selected = []
    avail = np.ones(len(grid), bool)
    step = float(np.median(np.diff(grid)))
    sup = max(1, int(round(min_sep_deg / step)))
    for _ in range(int(K)):
        scores = np.sum(np.abs(A.conj().T @ R) ** 2, axis=1)
        scores[~avail] = -np.inf
        idx = int(np.argmax(scores))
        selected.append(idx)
        avail[max(0, idx - sup) : min(len(grid), idx + sup + 1)] = False
        As = A[:, selected]
        X = np.linalg.pinv(As) @ Y
        R = Y - As @ X
    return np.array(sorted(grid[selected]), float)


def ar1_cov(M, rho):
    return rho ** np.abs(np.subtract.outer(np.arange(M), np.arange(M)))


def invsqrt_hermitian(R):
    vals, U = np.linalg.eigh(fd.hermitian(R))
    vals = np.maximum(vals.real, 1e-7)
    return U @ np.diag(1.0 / np.sqrt(vals)) @ U.conj().T


def estimate_ar1_rho_from_residual(Y, K, init="somp"):
    """Deployable AR(1)-like colored-noise proxy from residual covariance.

    We first remove a rough K-source model, then fit a nonnegative lag-decay proxy
    from the residual Toeplitz covariance. It is deliberately conservative and
    clipped to avoid over-whitening.
    """
    M, T = Y.shape
    try:
        if init == "toeplitz":
            specs = v10.expert_spectra_allK_fast(Y, GRID, KMAX)[K]
            theta0 = fd.topk_peaks(specs["Toeplitz-MUSIC"], GRID, K)
        elif init == "fbss":
            specs = v10.expert_spectra_allK_fast(Y, GRID, KMAX)[K]
            theta0 = fd.topk_peaks(specs["FBSS-MUSIC"], GRID, K)
        else:
            theta0 = v10.somp_estimate(Y, K, GRID)
        A0 = fd.array_manifold(0.5 * np.arange(M), theta0)
        X = np.linalg.pinv(A0) @ Y
        E = Y - A0 @ X
        R = fd.toeplitz_project(fd.sample_cov(E))
        d0 = float(np.mean(np.real(np.diag(R))))
        vals = []
        for lag in (1, 2, 3):
            dg = np.diag(R, k=lag)
            if len(dg) > 0 and d0 > 1e-12:
                vals.append((np.median(np.abs(dg)) / d0) ** (1.0 / lag))
        rho = float(np.median(vals)) if vals else 0.0
    except Exception:
        rho = 0.0
    return float(np.clip(rho, 0.0, 0.92))


def colored_whitened_somp(Y, K, grid=GRID, init="somp"):
    M, T = Y.shape
    rho = estimate_ar1_rho_from_residual(Y, K, init)
    Rn = ar1_cov(M, rho)
    W = invsqrt_hermitian(Rn)
    Yw = W @ Y
    A = W @ fd.steering_matrix(M, grid)
    A = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
    return somp_with_dictionary(Yw, A, K, grid), rho


def colored_whitened_music(Y, K, grid=GRID, init="somp"):
    M, T = Y.shape
    rho = estimate_ar1_rho_from_residual(Y, K, init)
    Rn = ar1_cov(M, rho)
    W = invsqrt_hermitian(Rn)
    Rw = fd.hermitian(W @ fd.sample_cov(Y) @ W.conj().T)
    vals, U = fd.eig_sorted(Rw)
    K = int(np.clip(K, 1, M - 1))
    Un = U[:, K:]
    Pn = Un @ Un.conj().T
    A = W @ fd.steering_matrix(M, grid)
    A = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
    denom = np.sum(A.conj() * (Pn @ A), axis=0).real
    P = 1.0 / np.maximum(denom, 1e-8)
    return fd.topk_peaks(P, grid, K), rho


def colored_whitened_fbss(Y, K, grid=GRID, init="somp"):
    # Hybrid version: whiten first, then apply spatial smoothing on whitened data.
    M, T = Y.shape
    rho = estimate_ar1_rho_from_residual(Y, K, init)
    Rn = ar1_cov(M, rho)
    W = invsqrt_hermitian(Rn)
    Yw = W @ Y
    L = max(K + 1, min(M - 1, M - K + 1))
    Rfb = fd.fbss_cov(Yw, K, L=L)
    P = fd.music_spectrum(Rfb, K, grid, positions=0.5 * np.arange(Rfb.shape[0]))
    return fd.topk_peaks(P, grid, K), rho


def colored_whitened_hybrid(Y, K, grid=GRID):
    """Colored-whitening sparse/FBSS hybrid expert.

    Candidate family: AR-whitened SOMP, generalized MUSIC, and whitened FBSS,
    with different rough residual initializers. The deployable self-score uses
    residual/covariance fit, candidate consensus, minimum separation, and the
    estimated colored-noise rho.
    """
    cands = []
    for init in ("somp", "toeplitz", "fbss"):
        for fun, label in (
            (colored_whitened_somp, "CW-SOMP"),
            (colored_whitened_music, "CW-MUSIC"),
            (colored_whitened_fbss, "CW-FBSS"),
        ):
            try:
                est, rho = fun(Y, K, grid, init)
                cands.append({"name": f"{label}-{init}", "est": est, "rho": rho})
            except Exception:
                pass
    if not cands:
        return v10.somp_estimate(Y, K, grid), {"name": "fallback-SOMP", "rho": 0.0, "score": np.inf}
    scores = []
    for c in cands:
        est = c["est"]
        res, cov, cond, sep = v10.projection_metrics(Y, est)
        cons = np.mean([v10.set_mae_penalized(est, cc["est"], 18.0) for cc in cands])
        # Complexity is already set by K. Reward plausible colored correlation.
        score = res + 0.22 * cov + 0.018 * cons + 0.04 * max(0.0, 1.0 - sep) - 0.08 * c["rho"]
        scores.append(score)
    j = int(np.argmin(scores))
    info = dict(cands[j])
    info["score"] = float(scores[j])
    info["n_candidates"] = len(cands)
    return cands[j]["est"], info


def diagnostics(Y):
    R0 = fd.hermitian(fd.sample_cov(Y))
    k_mdl, _, _ = v10.mdl_order_from_cov(R0, Y.shape[1], KMAX)
    k_gap, _ = v10.eigengap_order(R0, KMAX)
    _, raw_mdl = fd.diagnostic_features(Y, k_mdl)
    _, raw_gap = fd.diagnostic_features(Y, k_gap)
    snap, gap, rank, toe, rob, logcond = raw_mdl
    rob_max = float(max(rob, raw_gap[4]))
    toe_max = float(max(toe, raw_gap[3]))
    coherent_like = int(snap <= 0.15 and toe_max > 0.35 and (k_mdl == 1 or rank > 0.90))
    colored_like = int(
        (0.35 <= snap <= 0.65)
        and (rob_max < 0.28)
        and (toe_max < 0.55)
        and (k_mdl >= 2 or k_gap >= 1)
    )
    fewshot_like = int(snap >= 0.90)
    clean_like = int(snap <= 0.08 and rob_max < 0.10 and toe_max < 0.18)
    return {
        "k_mdl": int(k_mdl),
        "k_gap": int(k_gap),
        "snap": float(snap),
        "gap": float(gap),
        "rank": float(rank),
        "toe": float(toe),
        "rob": float(rob),
        "rob_max": rob_max,
        "toe_max": toe_max,
        "coherent_like": coherent_like,
        "colored_like": colored_like,
        "fewshot_like": fewshot_like,
        "clean_like": clean_like,
    }


def choose_v10_cached(Y):
    est, k, mode, expert = v10.choose_v10(Y, GRID, RANKER10, KMAX)
    return {"est": est, "k": int(k), "mode": mode, "expert": expert, "source": "FROST-V10"}


def choose_v11_cached(Y):
    est, k, mode, expert, cand, scores = v11.choose_v11(Y, RANKER11, GRID, KMAX)
    return {
        "est": est,
        "k": int(k),
        "mode": mode,
        "expert": expert,
        "source": "FROST-V11",
        "cand": cand,
        "scores": scores,
    }


def choose_v12_fresh(Y, diag, v10_out=None, v11_out=None):
    # Fresh deployable reconstruction of v12 champion policy. The exact v12 paired
    # script was policy/calibrator based; here we re-instantiate it at signal level.
    if v10_out is None:
        v10_out = choose_v10_cached(Y)
    if v11_out is None:
        v11_out = choose_v11_cached(Y)
    if diag["clean_like"]:
        method = "Toeplitz-MUSIC-MDL" if diag["k_mdl"] >= 2 else "MUSIC-SCM-MDL"
        est, k = v10.baseline_estimate(Y, GRID, method, KMAX)
        return {
            "est": est,
            "k": int(k),
            "mode": "v12fresh-clean-" + method,
            "expert": method.replace("-MDL", ""),
            "source": "FROST-V12F",
        }
    if diag["coherent_like"]:
        out = dict(v10_out)
        out.update({"mode": "v12fresh-coherent-v10", "source": "FROST-V12F"})
        return out
    if diag["rob_max"] > 0.17 and diag["snap"] <= 0.35:
        out = dict(v11_out)
        out.update({"mode": "v12fresh-robust-v11", "source": "FROST-V12F"})
        return out
    if (0.45 <= diag["snap"] < 0.75) and diag["rob_max"] < 0.25 and diag["toe_max"] < 0.55:
        out = dict(v10_out)
        out.update({"mode": "v12fresh-colored-v10", "source": "FROST-V12F"})
        return out
    out = dict(v11_out)
    out.update({"mode": "v12fresh-default-v11", "source": "FROST-V12F"})
    return out


def choose_v13_fresh(Y, diag, v10_out=None, v11_out=None, v12_out=None):
    if v10_out is None:
        v10_out = choose_v10_cached(Y)
    if v11_out is None:
        v11_out = choose_v11_cached(Y)
    if v12_out is None:
        v12_out = choose_v12_fresh(Y, diag, v10_out, v11_out)
    if diag["coherent_like"]:
        src = v11_out if diag["toe_max"] >= 0.585 else v10_out
        out = dict(src)
        out.update(
            {
                "mode": "v13fresh-coherent-v11" if src is v11_out else "v13fresh-coherent-v10",
                "source": "FROST-V13F",
            }
        )
        return out
    if diag["fewshot_like"]:
        if diag["toe_max"] >= 0.65:
            src = v11_out
            mode = "v13fresh-lowshot-hardmixed-v11"
        elif diag["k_gap"] in (3, 4):
            src = v10_out
            mode = "v13fresh-lowshot-gap34-v10"
        else:
            src = v11_out
            mode = "v13fresh-lowshot-gap12-v11"
        out = dict(src)
        out.update({"mode": mode, "source": "FROST-V13F"})
        return out
    if diag["colored_like"]:
        out = dict(v11_out)
        out.update({"mode": "v13fresh-colored-v11", "source": "FROST-V13F"})
        return out
    out = dict(v12_out)
    out.update({"mode": "v13fresh-default-v12fresh", "source": "FROST-V13F"})
    return out


def lowshot_k3_preserving_choice(Y, diag, v10_out, v11_out):
    """V14 low-shot selector.

    Design objective: keep V13's U4 low-shot K2 gain but avoid sacrificing K3.
    It protects any internally stable K>=3 branch instead of always inheriting
    the K-gap rule. The rule is deployable: it uses only V10/V11 outputs and
    MDL/GAP diagnostics, not true K.
    """
    k10 = int(v10_out["k"])
    k11 = int(v11_out["k"])
    kg = int(diag["k_gap"])
    # hard mixed / very high Toeplitz deviation: v11 bootstrap/sparse guard is safer.
    if diag["toe_max"] >= 0.68:
        src = v11_out
        mode = "v14-lowshot-highToe-v11"
    # Preserve K3 if either internal champion explicitly supports K>=3.
    elif k10 >= 3:
        src = v10_out
        mode = "v14-lowshot-k3preserve-v10"
    elif k11 >= 3:
        src = v11_out
        mode = "v14-lowshot-k3preserve-v11"
    elif kg >= 3:
        # If K-gap suggests high order but champions undercount, choose the lower residual branch.
        r10 = v10.projection_metrics(Y, v10_out["est"])[0]
        r11 = v10.projection_metrics(Y, v11_out["est"])[0]
        src = v10_out if r10 <= r11 + 0.04 else v11_out
        mode = "v14-lowshot-gapHigh-residual-split"
    else:
        # Low-shot K2-ish branch: choose the branch with cleaner projection but no K inflation.
        r10 = v10.projection_metrics(Y, v10_out["est"])[0]
        r11 = v10.projection_metrics(Y, v11_out["est"])[0]
        src = v10_out if r10 <= r11 + 0.02 else v11_out
        mode = "v14-lowshot-k2-split"
    out = dict(src)
    out.update({"mode": mode, "source": "FROST-V14"})
    return out


def colored_k_proxy(diag):
    km = int(diag["k_mdl"])
    kg = int(diag["k_gap"])
    # In colored low-SNR, MDL tends to overcount and eigengap tends to undercount;
    # the harmonic/rounded compromise is protected around K=2.
    if kg <= 2 and km >= 2:
        return 2
    return int(np.clip(round(0.5 * (km + kg)), 1, KMAX))


def choose_colored_whitening_branch(Y, diag):
    k = colored_k_proxy(diag)
    est, info = colored_whitened_hybrid(Y, k, GRID)
    return {
        "est": est,
        "k": int(k),
        "mode": "v14-colored-whitened-sparse-fbss",
        "expert": "CW-AR1-SOMP-FBSS",
        "source": "FROST-V14",
        "cw_info": info,
    }


def choose_v14(Y, diag, v10_out=None, v11_out=None, v12_out=None, v13_out=None):
    if v10_out is None:
        v10_out = choose_v10_cached(Y)
    if v11_out is None:
        v11_out = choose_v11_cached(Y)
    if v12_out is None:
        v12_out = choose_v12_fresh(Y, diag, v10_out, v11_out)
    if v13_out is None:
        v13_out = choose_v13_fresh(Y, diag, v10_out, v11_out, v12_out)
    if diag["clean_like"]:
        return dict(v12_out, source="FROST-V14", mode="v14-clean-inherit-v12")
    if diag["colored_like"]:
        return choose_colored_whitening_branch(Y, diag)
    if diag["coherent_like"]:
        # Keep v13 coherent split; it previously fixed coherent K3.
        src = v11_out if diag["toe_max"] >= 0.585 else v10_out
        out = dict(src)
        out.update(
            {
                "mode": "v14-coherent-v11" if src is v11_out else "v14-coherent-v10",
                "source": "FROST-V14",
            }
        )
        return out
    if diag["fewshot_like"]:
        return lowshot_k3_preserving_choice(Y, diag, v10_out, v11_out)
    return dict(v13_out, source="FROST-V14", mode="v14-default-inherit-v13")


def baseline_methods(Y):
    methods = ["MUSIC-SCM-MDL", "Toeplitz-MUSIC-MDL", "FBSS-MUSIC-MDL", "SOMP-MDL", "FROST-V6-MDL"]
    out = {}
    for m in methods:
        est, k = v10.baseline_estimate(Y, GRID, m, KMAX)
        out[m] = {"est": est, "k": int(k), "mode": m, "expert": m.replace("-MDL", "")}
    for m in ["Huber-SOMP-MDL", "Trim-SOMP-MDL", "SoftTrim-SOMP-MDL"]:
        est, k = v11.estimate_robust_baseline(Y, m, GRID, KMAX)
        out[m] = {"est": est, "k": int(k), "mode": m, "expert": m.replace("-MDL", "")}
    # Colored whitening baseline only triggers its internal colored proxy; useful for ablation.
    diag = diagnostics(Y)
    if diag["colored_like"]:
        cw = choose_colored_whitening_branch(Y, diag)
    else:
        # keep non-colored cases comparable: use ordinary SOMP-MDL as a neutral fallback.
        est, k = v10.baseline_estimate(Y, GRID, "SOMP-MDL", KMAX)
        cw = {
            "est": est,
            "k": int(k),
            "mode": "CW-inactive-fallback-SOMP-MDL",
            "expert": "CW-inactive",
        }
    out["CW-Whitened-Hybrid"] = cw
    return out


def oracle_candidates_from_bank(Y, diag, v11_out=None):
    # Use v11 candidate bank plus colored-whitened candidates when colored signature exists.
    if v11_out is not None and "cand" in v11_out:
        cand = list(v11_out["cand"])
    else:
        cand, _, _, _, _ = v11.make_candidates_v11(Y, GRID, KMAX)
    if diag["colored_like"]:
        for k in range(1, KMAX + 1):
            est, info = colored_whitened_hybrid(Y, k, GRID)
            cand.append(
                {
                    "expert": "CW-AR1-SOMP-FBSS",
                    "k": k,
                    "name": f"CW-AR1-SOMP-FBSS@K{k}",
                    "est": est,
                    "P": None,
                }
            )
    return cand


def evaluate(n_mc=50, seed=20261014):
    rng = np.random.default_rng(seed)
    rows = []
    decs = []
    sample_id = 0
    for sc in v10.test_scenarios_v9():
        for i in range(n_mc):
            Y, theta, _ = fd.simulate(sc, rng)
            diag = diagnostics(Y)
            v10_out = choose_v10_cached(Y)
            v11_out = choose_v11_cached(Y)
            v12_out = choose_v12_fresh(Y, diag, v10_out, v11_out)
            v13_out = choose_v13_fresh(Y, diag, v10_out, v11_out, v12_out)
            v14_out = choose_v14(Y, diag, v10_out, v11_out, v12_out, v13_out)
            outs = baseline_methods(Y)
            outs.update(
                {
                    "FROST-V10": v10_out,
                    "FROST-V11": v11_out,
                    "FROST-V12F": v12_out,
                    "FROST-V13F": v13_out,
                    "FROST-V14": v14_out,
                }
            )
            cand = oracle_candidates_from_bank(Y, diag, v11_out)
            errs = [set_rmse(c["est"], theta) for c in cand]
            j = int(np.argmin(errs))
            ok = [ii for ii, c in enumerate(cand) if int(c["k"]) == sc.K]
            jk = min(ok, key=lambda ii: set_rmse(cand[ii]["est"], theta)) if ok else j
            outs["ORACLE-CANDIDATE-V14"] = {
                "est": cand[j]["est"],
                "k": int(cand[j]["k"]),
                "mode": "oracle-candidate",
                "expert": cand[j]["expert"],
            }
            outs["ORACLE-K-BANK-V14"] = {
                "est": cand[jk]["est"],
                "k": int(cand[jk]["k"]),
                "mode": "oracle-k",
                "expert": cand[jk]["expert"],
            }
            sid = f"{sc.name}#{i:03d}"
            for m, o in outs.items():
                e = set_rmse(o["est"], theta)
                a = set_mae(o["est"], theta)
                kh = int(o["k"])
                rows.append(
                    {
                        "sample_id": sid,
                        "scenario": sc.name,
                        "true_K": sc.K,
                        "method": m,
                        "set_rmse": e,
                        "set_mae": a,
                        "k_hat": kh,
                        "k_acc": float(kh == sc.K),
                        "k_abs_err": abs(kh - sc.K),
                        "outlier_5deg": float(e > 5.0),
                    }
                )
            for m, o in [
                ("FROST-V10", v10_out),
                ("FROST-V11", v11_out),
                ("FROST-V12F", v12_out),
                ("FROST-V13F", v13_out),
                ("FROST-V14", v14_out),
            ]:
                rec = {
                    "sample_id": sid,
                    "scenario": sc.name,
                    "method": m,
                    "mode": o.get("mode", ""),
                    "expert": o.get("expert", ""),
                    "k_hat": int(o["k"]),
                    "true_K": sc.K,
                    "diag_k_mdl": diag["k_mdl"],
                    "diag_k_gap": diag["k_gap"],
                    "diag_snap": diag["snap"],
                    "diag_rob_max": diag["rob_max"],
                    "diag_toe_max": diag["toe_max"],
                    "diag_coherent_like": diag["coherent_like"],
                    "diag_colored_like": diag["colored_like"],
                    "diag_fewshot_like": diag["fewshot_like"],
                    "oracle_expert": cand[j]["expert"],
                    "oracle_k": int(cand[j]["k"]),
                    "oracle_name": cand[j]["name"],
                }
                if "cw_info" in o:
                    rec.update(
                        {
                            "cw_name": o["cw_info"].get("name", ""),
                            "cw_rho": o["cw_info"].get("rho", np.nan),
                            "cw_score": o["cw_info"].get("score", np.nan),
                        }
                    )
                decs.append(rec)
            sample_id += 1
    return pd.DataFrame(rows), pd.DataFrame(decs)


def aggregate(df):
    agg = (
        df.groupby(["scenario", "method"])
        .agg(
            set_rmse_mean=("set_rmse", "mean"),
            set_rmse_median=("set_rmse", "median"),
            set_mae_mean=("set_mae", "mean"),
            k_acc_pct=("k_acc", lambda x: 100 * x.mean()),
            k_abs_err_mean=("k_abs_err", "mean"),
            outlier_5deg_pct=("outlier_5deg", lambda x: 100 * x.mean()),
            n=("set_rmse", "size"),
        )
        .reset_index()
    )
    pivot = agg.pivot(index="scenario", columns="method", values="set_rmse_mean")
    kpiv = agg.pivot(index="scenario", columns="method", values="k_acc_pct")
    method_order = [
        "MUSIC-SCM-MDL",
        "Toeplitz-MUSIC-MDL",
        "FBSS-MUSIC-MDL",
        "SOMP-MDL",
        "Trim-SOMP-MDL",
        "SoftTrim-SOMP-MDL",
        "CW-Whitened-Hybrid",
        "FROST-V6-MDL",
        "FROST-V10",
        "FROST-V11",
        "FROST-V12F",
        "FROST-V13F",
        "FROST-V14",
        "ORACLE-K-BANK-V14",
        "ORACLE-CANDIDATE-V14",
    ]
    present = [m for m in method_order if m in pivot.columns]
    avg = pivot[present].rank(axis=1, method="min").mean().sort_values().reset_index()
    avg.columns = ["method", "avg_rank"]
    non = [m for m in present if not m.startswith("ORACLE")]
    nonavg = pivot[non].rank(axis=1, method="min").mean().sort_values().reset_index()
    nonavg.columns = ["method", "nonoracle_avg_rank"]
    overall = []
    for m in present:
        overall.append(
            {
                "method": m,
                "mean_rmse": float(pivot[m].mean()),
                "mean_k_acc_pct": float(kpiv[m].mean()),
                "avg_rank": float(avg.loc[avg.method == m, "avg_rank"].iloc[0]),
                "nonoracle_avg_rank": (
                    float(nonavg.loc[nonavg.method == m, "nonoracle_avg_rank"].iloc[0])
                    if m in non
                    else np.nan
                ),
            }
        )
    overall = pd.DataFrame(overall).sort_values("mean_rmse")
    return agg, pivot, kpiv, avg, nonavg, overall, present


def paired_stats(
    df,
    target="FROST-V14",
    bases=(
        "FROST-V13F",
        "FROST-V12F",
        "FROST-V11",
        "FROST-V10",
        "FROST-V6-MDL",
        "SOMP-MDL",
        "Toeplitz-MUSIC-MDL",
        "CW-Whitened-Hybrid",
    ),
):
    piv = df.pivot_table(
        index=["sample_id", "scenario", "true_K"],
        columns="method",
        values="set_rmse",
        aggfunc="first",
    )
    rows = []
    scen = []
    rng = np.random.default_rng(20261015)
    for base in bases:
        if base not in piv.columns or target not in piv.columns:
            continue
        diff = (piv[target] - piv[base]).dropna().values
        bs = []
        for _ in range(2000):
            idx = rng.integers(0, len(diff), len(diff))
            bs.append(float(np.mean(diff[idx])))
        rows.append(
            {
                "pair": f"{target} - {base}",
                "n": int(len(diff)),
                "mean_delta_rmse": float(np.mean(diff)),
                "median_delta_rmse": float(np.median(diff)),
                "ci95_low": float(np.quantile(bs, 0.025)),
                "ci95_high": float(np.quantile(bs, 0.975)),
                "win_rate_pct": 100 * float(np.mean(diff < -1e-9)),
                "tie_rate_pct": 100 * float(np.mean(np.abs(diff) <= 1e-9)),
                "loss_rate_pct": 100 * float(np.mean(diff > 1e-9)),
            }
        )
        for sc, g in piv.groupby(level=1):
            dd = (g[target] - g[base]).dropna().values
            scen.append(
                {
                    "scenario": sc,
                    "pair": f"{target} - {base}",
                    "mean_delta_rmse": float(np.mean(dd)),
                    "median_delta_rmse": float(np.median(dd)),
                    "wins": int(np.sum(dd < -1e-9)),
                    "ties": int(np.sum(np.abs(dd) <= 1e-9)),
                    "losses": int(np.sum(dd > 1e-9)),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(scen)


def make_plots(pivot, kpiv, avg, paired, prefix=PFX):
    plot_methods = [
        m
        for m in [
            "Toeplitz-MUSIC-MDL",
            "SOMP-MDL",
            "Trim-SOMP-MDL",
            "SoftTrim-SOMP-MDL",
            "CW-Whitened-Hybrid",
            "FROST-V6-MDL",
            "FROST-V10",
            "FROST-V11",
            "FROST-V13F",
            "FROST-V14",
            "ORACLE-K-BANK-V14",
            "ORACLE-CANDIDATE-V14",
        ]
        if m in pivot.columns
    ]
    ax = pivot[plot_methods].plot(kind="bar", figsize=(20, 6))
    ax.set_ylabel("Penalized set RMSE (degree)")
    ax.set_title("FROST-DOA v14 fresh signal-level MC unknown-K benchmark")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(prefix) + "_rmse_bar.png", dpi=220)
    plt.close()
    ax = pivot[plot_methods].plot(kind="bar", figsize=(20, 6), logy=True)
    ax.set_ylabel("Penalized set RMSE (degree, log)")
    ax.set_title("FROST-DOA v14 fresh signal-level MC unknown-K benchmark, log scale")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(prefix) + "_rmse_bar_log.png", dpi=220)
    plt.close()
    ax = kpiv[plot_methods].plot(kind="bar", figsize=(20, 6))
    ax.set_ylabel("Source-number accuracy (%)")
    ax.set_title("FROST-DOA v14 unknown-K source-number accuracy")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(prefix) + "_kacc_bar.png", dpi=220)
    plt.close()
    ax = avg.set_index("method")["avg_rank"].plot(kind="bar", figsize=(13, 5))
    ax.set_ylabel("Average rank")
    ax.set_title("Average rank across fresh MC50 unknown-K scenarios")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(prefix) + "_avg_rank.png", dpi=220)
    plt.close()
    ax = paired.set_index("pair")["mean_delta_rmse"].plot(kind="bar", figsize=(12, 4))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean paired RMSE diff (V14 - baseline)")
    ax.set_title("Fresh paired difference; negative means V14 is better")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(prefix) + "_paired_delta.png", dpi=220)
    plt.close()


def save_report(agg, df, dec, pivot, kpiv, avg, nonavg, overall, paired, scen, present, info):
    # tables
    agg.to_csv(str(PFX) + "_results.csv", index=False)
    df.to_csv(str(PFX) + "_sample_errors.csv", index=False)
    dec.to_csv(str(PFX) + "_decisions.csv", index=False)
    pivot.to_csv(str(PFX) + "_rmse_pivot.csv")
    kpiv.to_csv(str(PFX) + "_kacc_pivot.csv")
    avg.to_csv(str(PFX) + "_avg_rank.csv", index=False)
    nonavg.to_csv(str(PFX) + "_nonoracle_avg_rank.csv", index=False)
    overall.to_csv(str(PFX) + "_overall_summary.csv", index=False)
    paired.to_csv(str(PFX) + "_paired_stats.csv", index=False)
    scen.to_csv(str(PFX) + "_paired_scenario_diffs.csv", index=False)
    targets = [
        "U4_lowSNR_few_K2",
        "U5_lowSNR_few_K3",
        "U7_coherent_K3",
        "U11_colored_lowSNR_K2",
        "U10_hard_mixed_K2",
    ]
    core = [
        m
        for m in [
            "FROST-V10",
            "FROST-V11",
            "FROST-V13F",
            "FROST-V14",
            "CW-Whitened-Hybrid",
            "ORACLE-K-BANK-V14",
            "ORACLE-CANDIDATE-V14",
        ]
        if m in pivot.columns
    ]
    target_table = pivot.loc[targets, core].copy()
    target_table["v14_minus_v13F"] = target_table["FROST-V14"] - target_table["FROST-V13F"]
    target_table["v14_improve_pct_vs_v13F"] = (
        100
        * (target_table["FROST-V13F"] - target_table["FROST-V14"])
        / (target_table["FROST-V13F"] + 1e-12)
    )
    target_table.to_csv(str(PFX) + "_targeted_shortfalls.csv")
    scen_delta = pivot[core].copy()
    scen_delta["v14_minus_v13F"] = scen_delta["FROST-V14"] - scen_delta["FROST-V13F"]
    scen_delta["v14_improve_pct_vs_v13F"] = (
        100
        * (scen_delta["FROST-V13F"] - scen_delta["FROST-V14"])
        / (scen_delta["FROST-V13F"] + 1e-12)
    )
    scen_delta.to_csv(str(PFX) + "_scenario_deltas.csv")
    Path(str(PFX) + "_model_info.json").write_text(json.dumps(info, indent=2))
    # policy module artifact
    Path(str(PFX) + "_champion_policy.py").write_text(
        '''#!/usr/bin/env python3\n"""FROST-DOA v14 champion policy summary."""\n\ndef frost_v14_choice(diag):\n    if diag.get("clean_like",0): return "FROST_V12_CLEAN_SAFE"\n    if diag.get("colored_like",0): return "COLORED_WHITENED_SPARSE_FBSS"\n    if diag.get("coherent_like",0): return "FROST_V11" if diag.get("toe_max",0)>=0.585 else "FROST_V10"\n    if diag.get("fewshot_like",0): return "LOW_SHOT_K3_PRESERVING_SELECTOR"\n    return "FROST_V13_DEFAULT"\n'''
    )
    make_plots(pivot, kpiv, avg, paired, PFX)
    with open(str(PFX) + "_report.md", "w") as f:
        f.write("# FROST-DOA v14 fresh signal-level MC report\n\n")
        f.write(
            "V14 adds a low-shot K3-preserving selector and a colored-noise whitening sparse/FBSS hybrid expert, then runs a fresh signal-level MC benchmark. Deployable methods do not receive true K or scenario labels. ORACLE rows are non-deployable upper bounds.\n\n"
        )
        f.write("## Run info\n\n```json\n" + json.dumps(info, indent=2) + "\n```\n\n")
        f.write("## Overall summary\n\n" + overall.round(3).to_markdown(index=False) + "\n\n")
        f.write(
            "## Mean penalized set RMSE\n\n"
            + pivot[
                [
                    m
                    for m in [
                        "FROST-V10",
                        "FROST-V11",
                        "FROST-V13F",
                        "FROST-V14",
                        "CW-Whitened-Hybrid",
                        "ORACLE-K-BANK-V14",
                        "ORACLE-CANDIDATE-V14",
                    ]
                    if m in pivot.columns
                ]
            ]
            .round(3)
            .to_markdown()
            + "\n\n"
        )
        f.write(
            "## Source-number accuracy (%)\n\n"
            + kpiv[
                [
                    m
                    for m in [
                        "FROST-V10",
                        "FROST-V11",
                        "FROST-V13F",
                        "FROST-V14",
                        "CW-Whitened-Hybrid",
                        "ORACLE-K-BANK-V14",
                        "ORACLE-CANDIDATE-V14",
                    ]
                    if m in kpiv.columns
                ]
            ]
            .round(1)
            .to_markdown()
            + "\n\n"
        )
        f.write("## Targeted shortfalls\n\n" + target_table.round(3).to_markdown() + "\n\n")
        f.write("## Average rank\n\n" + avg.round(3).to_markdown(index=False) + "\n\n")
        f.write(
            "## Non-oracle average rank\n\n" + nonavg.round(3).to_markdown(index=False) + "\n\n"
        )
        f.write("## Paired statistics\n\n" + paired.round(4).to_markdown(index=False) + "\n\n")
        f.write(
            "## V14 decision counts\n\n"
            + dec[dec.method == "FROST-V14"]
            .groupby(["scenario", "mode", "expert", "k_hat"])
            .size()
            .reset_index(name="count")
            .to_markdown(index=False)
            + "\n"
        )
    return target_table, scen_delta


def main():
    t0 = time.time()
    n_mc = 50
    seed = 20261014
    print("Running FROST-V14 fresh signal-level MC...", flush=True)
    df, dec = evaluate(n_mc=n_mc, seed=seed)
    agg, pivot, kpiv, avg, nonavg, overall, present = aggregate(df)
    paired, scen = paired_stats(df)
    info = {
        "version": "FROST-V14 fresh signal-level MC",
        "n_mc_per_scenario": n_mc,
        "seed": seed,
        "grid_step_deg": 0.5,
        "KMAX": KMAX,
        "penalty_deg": PENALTY,
        "fresh_signal_level_run": True,
        "uses_true_K_at_decision_time": False,
        "uses_scenario_label_at_decision_time": False,
        "runtime_sec": time.time() - t0,
        "notes": [
            "V12F/V13F are fresh deployable re-instantiations of previous policies, not old sample-bank values.",
            "Colored whitening expert uses residual AR(1) noise-color proxy + whitened SOMP/MUSIC/FBSS hybrid.",
        ],
    }
    target, scen_delta = save_report(
        agg, df, dec, pivot, kpiv, avg, nonavg, overall, paired, scen, present, info
    )
    print("OVERALL")
    print(overall.round(3).to_string(index=False))
    print("\nTARGETED")
    print(target.round(3).to_string())
    print("\nPAIRED")
    print(paired.round(4).to_string(index=False))
    print("\nV14 decisions")
    print(
        dec[dec.method == "FROST-V14"]
        .groupby(["mode", "expert"])
        .size()
        .sort_values(ascending=False)
        .to_string()
    )
    print("Saved prefix", PFX, "runtime", time.time() - t0)


if __name__ == "__main__":
    main()
