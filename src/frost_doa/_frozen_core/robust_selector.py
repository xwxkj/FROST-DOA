"""Frozen robust candidate bank and 237-feature calibration ranker.

This module adds Huber-, hard-trimmed-, and softly-trimmed SOMP candidates,
together with deterministic snapshot-bootstrap stability scoring.
"""

from __future__ import annotations
import itertools, json, os, time
import numpy as np
import pandas as pd
from . import signal_processing as fd
from . import base_selector as v10

GRID = np.linspace(-60, 60, 241)
KMAX = 4
PENALTY = 25.0

_perm_cache = {}


def mae_set_fast(a, b, penalty=18.0):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = len(a)
    k = len(b)
    if m == 0 and k == 0:
        return 0.0
    if m == 0 or k == 0:
        return float(penalty)
    if m <= k:
        key = (k, m)
        if key not in _perm_cache:
            _perm_cache[key] = np.array(list(itertools.permutations(range(k), m)), int)
        P = _perm_cache[key]
        costs = np.sum(np.abs(a[None, :] - b[P]), axis=1) + (k - m) * penalty
        return float(np.min(costs) / max(m, k))
    key = (m, k)
    if key not in _perm_cache:
        _perm_cache[key] = np.array(list(itertools.permutations(range(m), k)), int)
    P = _perm_cache[key]
    costs = np.sum(np.abs(a[P] - b[None, :]), axis=1) + (m - k) * penalty
    return float(np.min(costs) / max(m, k))


def robust_weights_energy(Y, mode="huber", keep_frac=0.75):
    e = np.mean(np.abs(Y) ** 2, axis=0)
    med = np.median(e) + 1e-12
    if mode == "huber":
        c = 2.5 * med
        w = np.minimum(1.0, np.sqrt(c / (e + 1e-12)))
    elif mode == "trim":
        thr = np.quantile(e, keep_frac)
        w = (e <= thr).astype(float)
        if np.sum(w) < max(1, Y.shape[1] // 3):
            w = np.ones(Y.shape[1])
    elif mode == "softtrim":
        q = np.quantile(e, keep_frac)
        w = 1.0 / (1.0 + np.exp((e - q) / (0.25 * q + 1e-12)))
    else:
        w = np.ones(Y.shape[1])
    return w


def robust_somp_estimate(Y, K, grid=GRID, mode="huber", min_sep_deg=1.0):
    if Y.shape[1] <= 2:
        return v10.somp_estimate(Y, K, grid, min_sep_deg)
    if mode in ("huber", "trim", "softtrim"):
        w = robust_weights_energy(Y, mode=mode)
        return v10.somp_estimate(Y * w[None, :], K, grid, min_sep_deg)
    if mode == "tyler":
        M, T = Y.shape
        Rty = fd.tyler_cov(Y, n_iter=4)
        load = 1e-3 * np.trace(Rty).real / M
        try:
            RiY = np.linalg.solve(Rty + load * np.eye(M), Y)
        except Exception:
            RiY = np.linalg.pinv(Rty + load * np.eye(M)) @ Y
        d = np.sum(Y.conj() * RiY, axis=0).real
        med = np.median(d) + 1e-12
        w = np.minimum(1.0, np.sqrt(2.5 * med / (d + 1e-12)))
        return v10.somp_estimate(Y * w[None, :], K, grid, min_sep_deg)
    return v10.somp_estimate(Y, K, grid, min_sep_deg)


BASE_EXPERTS_V11 = v10.BASE_EXPERTS + ["Huber-SOMP", "Trim-SOMP", "SoftTrim-SOMP"]
ROBUST_MAP = {
    "Huber-SOMP": "huber",
    "Trim-SOMP": "trim",
    "SoftTrim-SOMP": "softtrim",
}
SPARSE_EXPERTS = {"SOMP", "Huber-SOMP", "Trim-SOMP", "SoftTrim-SOMP"}
ROBUST_EXPERTS = {"Huber-SOMP", "Trim-SOMP", "SoftTrim-SOMP"}


def make_candidates_v11(Y, grid=GRID, kmax=KMAX):
    cand, specs, beta, raw, mode = v10.make_candidates_unknownK(Y, grid, kmax)
    for k in range(1, kmax + 1):
        for ex, mo in ROBUST_MAP.items():
            cand.append(
                {
                    "expert": ex,
                    "k": k,
                    "name": f"{ex}@K{k}",
                    "est": robust_somp_estimate(Y, k, grid, mo),
                    "P": None,
                }
            )
    return cand, specs, beta, raw, mode


def candidate_feature_matrix_v11(Y, grid, cand, specs_by_k, beta_by_k, raw_by_k, kmax=KMAX):
    kg = v10.k_global_features(Y, grid, kmax)
    k_mdl = int(round(kg[0] * kmax))
    k_gap = int(round(kg[1] * kmax))
    C = len(cand)
    Dmat = np.zeros((C, C), float)
    for i in range(C):
        ai = cand[i]["est"]
        for j in range(i + 1, C):
            d = mae_set_fast(ai, cand[j]["est"])
            Dmat[i, j] = Dmat[j, i] = d
    proj = [v10.projection_metrics(Y, c["est"]) for c in cand]
    e = np.mean(np.abs(Y) ** 2, axis=0)
    eg = np.array(
        [
            np.mean(e),
            np.std(e) / (np.mean(e) + 1e-12),
            np.max(e) / (np.median(e) + 1e-12),
            np.quantile(e, 0.9) / (np.median(e) + 1e-12),
        ]
    )
    feats = []
    for idx, c in enumerate(cand):
        k = c["k"]
        expert = c["expert"]
        theta = c["est"]
        beta = beta_by_k[k]
        raw = raw_by_k[k]
        snap, gap, rank, toe, rob, logcond = raw
        res, cov, cond, sep = proj[idx]
        mask = np.ones(C, bool)
        mask[idx] = False
        ds_all = Dmat[idx, mask]
        maskk = np.array([o["k"] == k and ii != idx for ii, o in enumerate(cand)])
        ds_k = Dmat[idx, maskk]
        cons_all = np.array([np.mean(ds_all), np.min(ds_all), np.median(ds_all), np.max(ds_all)])
        cons_k = (
            np.array([np.mean(ds_k), np.min(ds_k), np.median(ds_k), np.max(ds_k)])
            if len(ds_k) > 0
            else np.zeros(4)
        )
        specs = specs_by_k[k]
        sf = []
        for sn in [expert, "Toeplitz-MUSIC", "FBSS-MUSIC", "MUSIC-SCM", "Tyler-MUSIC", "Fuzzy-LSE"]:
            vals = v10.interp_p(specs.get(sn, None), grid, theta)
            sf += [float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))]
        kf = np.array(
            [
                k / kmax,
                k / Y.shape[0],
                abs(k - k_mdl) / kmax,
                abs(k - k_gap) / kmax,
                1.0 if k == k_mdl else 0.0,
                1.0 if k == k_gap else 0.0,
                fd.sigmoid((0.45 - res) / 0.08),
                fd.sigmoid((res - 0.65) / 0.10),
                fd.sigmoid((sep - 1.2) / 0.5),
                fd.sigmoid((2.5 - cond) / 0.7),
            ]
        )
        mem = np.array(
            [
                fd.sigmoid((0.55 - res) / 0.10),
                fd.sigmoid((4.0 - cons_all[2]) / 1.5),
                fd.sigmoid((4.0 - cons_k[2]) / 1.5),
                fd.sigmoid((sep - 1.2) / 0.5),
                fd.sigmoid((np.mean(sf[0::3]) - 0.5) / 0.15),
                fd.sigmoid((0.75 - snap) / 0.25),
                fd.sigmoid((snap - 0.75) / 0.25),
                fd.sigmoid((0.25 - rob) / 0.06),
                fd.sigmoid((rob - 0.25) / 0.06),
                fd.sigmoid((rank - 0.42) / 0.10),
            ]
        )
        ohe = np.zeros(len(BASE_EXPERTS_V11))
        ohe[BASE_EXPERTS_V11.index(expert)] = 1.0
        ohk = np.zeros(kmax)
        ohk[k - 1] = 1.0
        ef = np.array(
            [
                expert in SPARSE_EXPERTS,
                expert in ROBUST_EXPERTS or expert == "Tyler-MUSIC",
                expert in fd.EXPERTS,
                expert.startswith("Fuzzy") or expert == "FROST-V6",
            ],
            dtype=float,
        )
        feats.append(
            np.concatenate(
                [
                    beta,
                    raw,
                    kg,
                    eg,
                    np.array([res, cov, cond, sep]),
                    cons_all / 20,
                    cons_k / 20,
                    np.array(sf),
                    kf,
                    mem,
                    ef,
                    ohe,
                    ohk,
                    np.outer(ohe, beta).ravel(),
                    np.outer(ohk, beta).ravel(),
                ]
            )
        )
    return np.vstack(feats)


class Ranker:
    def __init__(self, w, mean, std):
        self.w = w
        self.mean = mean
        self.std = std

    def score(self, X):
        return np.tensordot((X - self.mean) / (self.std + 1e-8), self.w, axes=([-1], [0]))


def train_ranker_v11(grid=GRID, n_train=360, seed=20260725, epochs=220, kmax=KMAX):
    rng = np.random.default_rng(seed)
    X = []
    y = []
    counts = {}
    oracle = []
    for i in range(n_train):
        sc = v10.make_train_scenario_v9(rng)
        Y, theta, _ = fd.simulate(sc, rng)
        cand, specs, beta, raw, mode = make_candidates_v11(Y, grid, kmax)
        errs = np.array([v10.penalized_set_rmse(c["est"], theta, PENALTY) for c in cand])
        best = int(np.argmin(errs))
        oracle.append(float(errs[best]))
        counts[cand[best]["name"]] = counts.get(cand[best]["name"], 0) + 1
        X.append(candidate_feature_matrix_v11(Y, grid, cand, specs, beta, raw, kmax))
        y.append(best)
    X = np.stack(X)
    y = np.array(y)
    N, C, D = X.shape
    mean = X.reshape(-1, D).mean(0)
    std = X.reshape(-1, D).std(0) + 1e-6
    Xn = (X - mean) / std
    w = np.zeros(D)
    m = np.zeros(D)
    v = np.zeros(D)
    lr = 0.035
    b1 = 0.9
    b2 = 0.999
    for t in range(1, epochs + 1):
        s = np.einsum("ncd,d->nc", Xn, w)
        s -= s.max(1, keepdims=True)
        P = np.exp(s)
        P /= P.sum(1, keepdims=True)
        Y1 = np.zeros_like(P)
        Y1[np.arange(N), y] = 1
        grad = np.einsum("nc,ncd->d", P - Y1, Xn) / N + 1e-4 * w
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * (grad * grad)
        w -= lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + 1e-8)
    pred = np.argmax(np.einsum("ncd,d->nc", Xn, w), 1)
    info = {
        "n_train": n_train,
        "epochs": epochs,
        "train_acc_best_candidate": float(np.mean(pred == y)),
        "oracle_train_penalized_rmse": float(np.mean(oracle)),
        "feature_dim": int(D),
        "candidates_per_sample": int(C),
        "label_counts_top20": dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:20]),
    }
    return Ranker(w, mean, std), info


def estimate_sparse_by_expert(Y, expert, k, grid=GRID):
    if expert == "SOMP":
        return v10.somp_estimate(Y, k, grid)
    if expert == "Huber-SOMP":
        return robust_somp_estimate(Y, k, grid, "huber")
    if expert == "Trim-SOMP":
        return robust_somp_estimate(Y, k, grid, "trim")
    if expert == "SoftTrim-SOMP":
        return robust_somp_estimate(Y, k, grid, "softtrim")
    return v10.somp_estimate(Y, k, grid)


def bootstrap_stability_sparse(Y, cand, idxs, grid=GRID, B=3, penalty=18.0):
    # Deterministic snapshot bootstrap. Lower is more stable.
    M, T = Y.shape
    seed = int((np.abs(np.sum(np.real(Y))) + np.abs(np.sum(np.imag(Y)))) * 1e6) % (2**32 - 1)
    rng = np.random.default_rng(seed)
    out = {}
    if T <= 2:
        return {i: 0.0 for i in idxs}
    for i in idxs:
        c = cand[i]
        ds = []
        for _ in range(B):
            sel = rng.integers(0, T, size=T)
            Yb = Y[:, sel]
            eb = estimate_sparse_by_expert(Yb, c["expert"], c["k"], grid)
            ds.append(mae_set_fast(c["est"], eb, penalty=penalty))
        out[i] = float(np.mean(ds) + 0.35 * np.std(ds))
    return out


def safe_physics_estimate(Y, grid=GRID, kmax=KMAX):
    R0 = fd.hermitian(fd.sample_cov(Y))
    k_mdl, _, _ = v10.mdl_order_from_cov(R0, Y.shape[1], kmax)
    k_gap, _ = v10.eigengap_order(R0, kmax)
    beta, raw = fd.diagnostic_features(Y, k_mdl)
    snap, gap, rank, toe, rob, logcond = raw
    if snap <= 0.08 and toe < 0.18 and rob < 0.10:
        if k_mdl >= 3:
            est, k = v10.baseline_estimate(Y, grid, "Toeplitz-MUSIC-MDL", kmax)
            return est, k, "clean-toeplitz-mdl", "Toeplitz-MUSIC"
        else:
            est, k = v10.baseline_estimate(Y, grid, "MUSIC-SCM-MDL", kmax)
            return est, k, "clean-scm-mdl", "MUSIC-SCM"
    est, k = v10.baseline_estimate(Y, grid, "FROST-V6-MDL", kmax)
    return est, k, "safe-v6-mdl", "FROST-V6"


def choose_v11_core(Y, ranker, grid=GRID, kmax=KMAX):
    cand, specs, beta, raw, mode = make_candidates_v11(Y, grid, kmax)
    X = candidate_feature_matrix_v11(Y, grid, cand, specs, beta, raw, kmax)
    scores = ranker.score(X)
    order = np.argsort(scores)[::-1]
    return cand[int(order[0])], cand, scores, specs, beta, raw


def choose_v11(Y, ranker, grid=GRID, kmax=KMAX):
    top, cand, scores, specs, beta_by_k, raw_by_k = choose_v11_core(Y, ranker, grid, kmax)
    R0 = fd.hermitian(fd.sample_cov(Y))
    k_mdl, _, _ = v10.mdl_order_from_cov(R0, Y.shape[1], kmax)
    k_gap, _ = v10.eigengap_order(R0, kmax)
    _, raw_mdl = fd.diagnostic_features(Y, k_mdl)
    _, raw_gap = fd.diagnostic_features(Y, k_gap)
    snap, gap, rank, toe, rob, logcond = raw_mdl
    rob_max = max(rob, raw_gap[4])
    toe_max = max(toe, raw_gap[3])
    coherent_like = snap <= 0.15 and toe_max > 0.35 and (k_mdl == 1 or rank > 0.90)
    highrisk = snap >= 0.45 or rob_max > 0.20 or coherent_like
    if highrisk:
        order = np.argsort(scores)[::-1]
        sparse_idx = [i for i, c in enumerate(cand) if c["expert"] in SPARSE_EXPERTS]
        chosen = top
        mode = "v11-ranker"
        if sparse_idx and (snap >= 0.45 or rob_max > 0.15):
            margin = 0.65 if snap >= 0.45 else 0.45
            pool = [i for i in sparse_idx if scores[i] >= scores[order[0]] - margin]
            if len(pool) == 0:
                pool = sorted(sparse_idx, key=lambda i: scores[i], reverse=True)[:8]
            else:
                pool = sorted(pool, key=lambda i: scores[i], reverse=True)[:10]
            boot = bootstrap_stability_sparse(Y, cand, pool, grid, B=3)
            sv = np.array([scores[i] for i in pool])
            bv = np.array([boot[i] for i in pool])
            sz = (sv - np.mean(sv)) / (np.std(sv) + 1e-8)
            bz = (bv - np.min(bv)) / (np.std(bv) + 1e-8)
            finals = []
            for ii, i in enumerate(pool):
                c = cand[i]
                k_bonus = (
                    0.15 * (c["k"] == k_mdl)
                    + 0.10 * (c["k"] == k_gap)
                    + 0.05 * (c["k"] >= min(k_mdl, k_gap))
                )
                robust_bonus = 0.12 * (c["expert"] in ROBUST_EXPERTS and rob_max > 0.18)
                finals.append(sz[ii] - 0.55 * bz[ii] + k_bonus + robust_bonus)
            best = pool[int(np.argmax(finals))]
            if scores[best] >= scores[order[0]] - margin - 0.10:
                chosen = cand[best]
                mode = "v11-bootstrap-robust-sparse"
        return chosen["est"], chosen["k"], mode, chosen["expert"], cand, scores

    # Fast v10-style physics safeguard using already computed candidate bank.
    def get(expert, k):
        for c in cand:
            if c["expert"] == expert and c["k"] == k:
                return c
        return None

    if snap <= 0.08 and toe < 0.18 and rob < 0.10:
        expert = "Toeplitz-MUSIC" if k_mdl >= 3 else "MUSIC-SCM"
        c = get(expert, k_mdl)
        return c["est"], c["k"], "v10-physics-safe:clean-" + expert, expert, cand, scores
    c = get("FROST-V6", k_mdl)
    return c["est"], c["k"], "v10-physics-safe:safe-v6-mdl", "FROST-V6", cand, scores


def estimate_robust_baseline(Y, method, grid=GRID, kmax=KMAX):
    R0 = fd.hermitian(fd.sample_cov(Y))
    k, _, _ = v10.mdl_order_from_cov(R0, Y.shape[1], kmax)
    if method == "Huber-SOMP-MDL":
        est = robust_somp_estimate(Y, k, grid, "huber")
    elif method == "Trim-SOMP-MDL":
        est = robust_somp_estimate(Y, k, grid, "trim")
    elif method == "SoftTrim-SOMP-MDL":
        est = robust_somp_estimate(Y, k, grid, "softtrim")
    else:
        raise ValueError(method)
    return est, k


def test_scenarios():
    return v10.test_scenarios_v9()


def evaluate(ranker, grid=GRID, n_mc=50, seed=20260727, kmax=KMAX):
    rng = np.random.default_rng(seed)
    rows = []
    decisions = []
    base_methods = [
        "MUSIC-SCM-MDL",
        "Toeplitz-MUSIC-MDL",
        "FBSS-MUSIC-MDL",
        "SOMP-MDL",
        "FROST-V6-MDL",
        "Huber-SOMP-MDL",
        "Trim-SOMP-MDL",
        "SoftTrim-SOMP-MDL",
    ]
    expert_for_method = {
        "MUSIC-SCM-MDL": "MUSIC-SCM",
        "Toeplitz-MUSIC-MDL": "Toeplitz-MUSIC",
        "FBSS-MUSIC-MDL": "FBSS-MUSIC",
        "SOMP-MDL": "SOMP",
        "FROST-V6-MDL": "FROST-V6",
        "Huber-SOMP-MDL": "Huber-SOMP",
        "Trim-SOMP-MDL": "Trim-SOMP",
        "SoftTrim-SOMP-MDL": "SoftTrim-SOMP",
    }
    for sc in test_scenarios():
        for i in range(n_mc):
            Y, theta, _ = fd.simulate(sc, rng)
            est, k, mode, expert, cand, scores = choose_v11(Y, ranker, grid, kmax)
            R0 = fd.hermitian(fd.sample_cov(Y))
            k_mdl, _, _ = v10.mdl_order_from_cov(R0, Y.shape[1], kmax)
            ests = {"FROST-V11": est}
            ks = {"FROST-V11": k}
            for m in base_methods:
                ex = expert_for_method[m]
                cc = next(c for c in cand if c["expert"] == ex and c["k"] == k_mdl)
                ests[m] = cc["est"]
                ks[m] = cc["k"]
            errs_c = [v10.penalized_set_rmse(c["est"], theta, PENALTY) for c in cand]
            j = int(np.argmin(errs_c))
            ests["ORACLE-CANDIDATE-V11"] = cand[j]["est"]
            ks["ORACLE-CANDIDATE-V11"] = cand[j]["k"]
            ok_idxs = [ii for ii, c in enumerate(cand) if c["k"] == sc.K]
            jj = min(
                ok_idxs, key=lambda ii: v10.penalized_set_rmse(cand[ii]["est"], theta, PENALTY)
            )
            ests["ORACLE-K-BANK-V11"] = cand[jj]["est"]
            ks["ORACLE-K-BANK-V11"] = cand[jj]["k"]
            for m in ests:
                err = v10.penalized_set_rmse(ests[m], theta, PENALTY)
                mae = v10.set_mae_penalized(ests[m], theta, PENALTY)
                rows.append(
                    {
                        "scenario": sc.name,
                        "method": m,
                        "set_rmse": err,
                        "set_mae": mae,
                        "k_acc": float(ks[m] == sc.K),
                        "k_abs_err": abs(ks[m] - sc.K),
                        "outlier_5deg": float(err > 5),
                    }
                )
            decisions.append(
                {
                    "scenario": sc.name,
                    "mode": mode,
                    "expert": expert,
                    "k": k,
                    "oracle_name": cand[j]["name"],
                    "oracle_k": cand[j]["k"],
                    "oracle_expert": cand[j]["expert"],
                }
            )
    df = pd.DataFrame(rows)
    dec = pd.DataFrame(decisions)
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
    return agg, df, dec


def save_outputs(agg, df, dec, info, ranker, prefix="/mnt/data/frost_doa_v11_unknownK"):
    agg.to_csv(prefix + "_results.csv", index=False)
    df.to_csv(prefix + "_sample_errors.csv", index=False)
    dec.to_csv(prefix + "_decisions.csv", index=False)
    pivot = agg.pivot(index="scenario", columns="method", values="set_rmse_mean")
    kpiv = agg.pivot(index="scenario", columns="method", values="k_acc_pct")
    pivot.to_csv(prefix + "_rmse_pivot.csv")
    kpiv.to_csv(prefix + "_kacc_pivot.csv")
    major = [
        m
        for m in [
            "MUSIC-SCM-MDL",
            "Toeplitz-MUSIC-MDL",
            "FBSS-MUSIC-MDL",
            "SOMP-MDL",
            "Huber-SOMP-MDL",
            "Trim-SOMP-MDL",
            "SoftTrim-SOMP-MDL",
            "FROST-V6-MDL",
            "FROST-V11",
            "ORACLE-K-BANK-V11",
            "ORACLE-CANDIDATE-V11",
        ]
        if m in pivot.columns
    ]
    avg = pivot[major].rank(axis=1, method="min").mean().sort_values().reset_index()
    avg.columns = ["method", "avg_rank"]
    avg.to_csv(prefix + "_avg_rank.csv", index=False)
    np.savez(prefix + "_ranker_model.npz", w=ranker.w, mean=ranker.mean, std=ranker.std)
    with open(prefix + "_model_info.json", "w") as f:
        json.dump(info, f, indent=2)
    try:
        import matplotlib.pyplot as plt

        pm = [
            m
            for m in [
                "MUSIC-SCM-MDL",
                "Toeplitz-MUSIC-MDL",
                "FBSS-MUSIC-MDL",
                "SOMP-MDL",
                "Trim-SOMP-MDL",
                "SoftTrim-SOMP-MDL",
                "FROST-V6-MDL",
                "FROST-V11",
                "ORACLE-K-BANK-V11",
                "ORACLE-CANDIDATE-V11",
            ]
            if m in pivot.columns
        ]
        ax = pivot[pm].plot(kind="bar", figsize=(17, 6))
        ax.set_ylabel("Penalized set RMSE (degree)")
        ax.set_title("FROST-DOA v11 unknown-K benchmark")
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(prefix + "_rmse_bar.png", dpi=220)
        plt.close()
        ax = pivot[pm].plot(kind="bar", figsize=(17, 6), logy=True)
        ax.set_ylabel("Penalized set RMSE (degree, log)")
        ax.set_title("FROST-DOA v11 unknown-K benchmark, log scale")
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(prefix + "_rmse_bar_log.png", dpi=220)
        plt.close()
        ax = kpiv[pm].plot(kind="bar", figsize=(17, 6))
        ax.set_ylabel("Source-number accuracy (%)")
        ax.set_title("Unknown-K source enumeration accuracy")
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(prefix + "_kacc_bar.png", dpi=220)
        plt.close()
        ax = avg.set_index("method")["avg_rank"].plot(kind="bar", figsize=(11, 5))
        ax.set_ylabel("Average rank")
        ax.set_title("Average rank across unknown-K scenarios")
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(prefix + "_avg_rank.png", dpi=220)
        plt.close()
    except Exception as e:
        print("plot skipped", repr(e))
    with open(prefix + "_report.md", "w") as f:
        f.write("# FROST-DOA v11 unknown-K benchmark report\n\n")
        f.write(
            "v11 adds robust sparse experts (Huber-SOMP, Trim-SOMP, SoftTrim-SOMP, Tyler-weighted SOMP) and a neuro-fuzzy champion selector. Deployment methods do not receive true K.\n\n"
        )
        f.write(
            "Penalized set RMSE uses a 25 degree cardinality penalty. ORACLE rows are nondeployable upper bounds.\n\n"
        )
        f.write("## Training info\n\n```json\n" + json.dumps(info, indent=2) + "\n```\n\n")
        f.write("## Mean penalized set RMSE\n\n" + pivot[major].round(3).to_markdown() + "\n\n")
        f.write("## Source-number accuracy (%)\n\n" + kpiv[major].round(1).to_markdown() + "\n\n")
        f.write("## Average rank\n\n" + avg.round(3).to_markdown(index=False) + "\n\n")
        f.write(
            "## Decision counts\n\n"
            + dec.groupby(["scenario", "mode", "expert", "k"])
            .size()
            .reset_index(name="count")
            .to_markdown(index=False)
            + "\n"
        )
    return pivot, kpiv, avg


def main():
    t0 = time.time()
    ranker, info = train_ranker_v11(n_train=260, epochs=140)
    info["train_runtime_sec"] = time.time() - t0
    print("TRAIN_INFO", json.dumps(info), flush=True)
    t1 = time.time()
    agg, df, dec = evaluate(ranker, n_mc=8, seed=20260727)
    info["eval_runtime_sec"] = time.time() - t1
    pivot, kpiv, avg = save_outputs(agg, df, dec, info, ranker)
    print("RMSE_MEAN")
    print(
        pivot[
            [
                "FROST-V11",
                "SOMP-MDL",
                "Trim-SOMP-MDL",
                "SoftTrim-SOMP-MDL",
                "Toeplitz-MUSIC-MDL",
                "ORACLE-K-BANK-V11",
                "ORACLE-CANDIDATE-V11",
            ]
        ]
        .round(3)
        .to_string()
    )
    print("AVERAGE_RANK")
    print(avg.round(3).to_string(index=False))
    print("KACC_MEAN")
    print(
        kpiv[["FROST-V11", "SOMP-MDL", "Trim-SOMP-MDL", "SoftTrim-SOMP-MDL", "Toeplitz-MUSIC-MDL"]]
        .mean()
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
