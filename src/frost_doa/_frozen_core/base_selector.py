"""Frozen base candidate bank and 202-feature calibration ranker.

The public release retains this module verbatim in numerical behavior while
using package-relative imports and packaged ranker parameters.
"""

from __future__ import annotations
import itertools, json, os, time
import numpy as np
import pandas as pd
from . import signal_processing as fd

BASE_EXPERTS = fd.EXPERTS + ["SOMP", "FROST-V6", "Fuzzy-LSE", "Fuzzy-Arith"]


def expert_spectra_allK_fast(Y, grid, kmax=4):
    M, T = Y.shape
    R0 = fd.hermitian(fd.sample_cov(Y))
    Rsh = fd.shrinkage_cov(R0, T=T)
    Rtoe = fd.toeplitz_project(R0)
    Rty = fd.tyler_cov(Y, n_iter=4)
    out = {}
    for k in range(1, kmax + 1):
        specs = {}
        specs["MUSIC-SCM"] = fd.music_spectrum(R0, k, grid, positions=0.5 * np.arange(M))
        specs["Shrink-MUSIC"] = fd.music_spectrum(Rsh, k, grid, positions=0.5 * np.arange(M))
        specs["Toeplitz-MUSIC"] = fd.music_spectrum(Rtoe, k, grid, positions=0.5 * np.arange(M))
        L = max(k + 1, min(M - 1, M - k + 1))
        Rfb = fd.fbss_cov(Y, k, L=L)
        specs["FBSS-MUSIC"] = fd.music_spectrum(
            Rfb, k, grid, positions=0.5 * np.arange(Rfb.shape[0])
        )
        specs["Tyler-MUSIC"] = fd.music_spectrum(Rty, k, grid, positions=0.5 * np.arange(M))
        out[k] = specs
    return out


def somp_estimate(Y, K, grid, min_sep_deg=1.0):
    M, T = Y.shape
    A = fd.steering_matrix(M, grid)
    R = Y.copy()
    selected = []
    available = np.ones(len(grid), bool)
    step = float(np.median(np.diff(grid)))
    sup = max(1, int(round(min_sep_deg / step)))
    for _ in range(int(K)):
        scores = np.sum(np.abs(A.conj().T @ R) ** 2, axis=1)
        scores[~available] = -np.inf
        idx = int(np.argmax(scores))
        selected.append(idx)
        lo = max(0, idx - sup)
        hi = min(len(grid), idx + sup + 1)
        available[lo:hi] = False
        As = A[:, selected]
        X = np.linalg.pinv(As) @ Y
        R = Y - As @ X
    return np.array(sorted(grid[selected]), float)


def mdl_order_from_cov(R, T, kmax=4, force_min_1=True):
    vals, _ = fd.eig_sorted(R)
    vals = np.maximum(vals.real, 1e-12)
    M = len(vals)
    maxk = int(min(kmax, M - 1))
    ks = range(1 if force_min_1 else 0, maxk + 1)
    scores = []
    for k in ks:
        noise = vals[k:]
        if len(noise) == 0:
            scores.append(np.inf)
            continue
        am = np.mean(noise)
        gm = np.exp(np.mean(np.log(noise)))
        ratio = max(gm / (am + 1e-12), 1e-12)
        scores.append(
            float(-T * (M - k) * np.log(ratio) + 0.5 * k * (2 * M - k) * np.log(max(T, 2)))
        )
    scores = np.array(scores, float)
    return int(list(ks)[int(np.argmin(scores))]), scores, list(ks)


def eigengap_order(R, kmax=4):
    vals, _ = fd.eig_sorted(R)
    vals = np.maximum(vals.real, 1e-12)
    ratios = vals[:-1] / vals[1:]
    return int(np.argmax(ratios[:kmax]) + 1), ratios[:kmax]


def penalized_set_rmse(est, true, penalty_deg=25.0):
    est = np.asarray(est, float)
    true = np.asarray(true, float)
    m = len(est)
    k = len(true)
    if m == 0 and k == 0:
        return 0.0
    if m == 0 or k == 0:
        return float(penalty_deg)
    best = 1e18
    if m <= k:
        for idx in itertools.permutations(range(k), m):
            best = min(best, np.sum((est - true[list(idx)]) ** 2) + (k - m) * penalty_deg**2)
    else:
        for idx in itertools.permutations(range(m), k):
            best = min(best, np.sum((est[list(idx)] - true) ** 2) + (m - k) * penalty_deg**2)
    return float(np.sqrt(best / max(m, k)))


def set_mae_penalized(est, true, penalty_deg=25.0):
    est = np.asarray(est, float)
    true = np.asarray(true, float)
    m = len(est)
    k = len(true)
    if m == 0 and k == 0:
        return 0.0
    if m == 0 or k == 0:
        return float(penalty_deg)
    best = 1e18
    if m <= k:
        for idx in itertools.permutations(range(k), m):
            best = min(best, np.sum(np.abs(est - true[list(idx)])) + (k - m) * penalty_deg)
    else:
        for idx in itertools.permutations(range(m), k):
            best = min(best, np.sum(np.abs(est[list(idx)] - true)) + (m - k) * penalty_deg)
    return float(best / max(m, k))


def dist_between_sets_var(a, b, penalty=18.0):
    return set_mae_penalized(a, b, penalty_deg=penalty)


def v6_estimate(Y, specs, beta, raw, grid, K):
    snap, gap, rank, toe, rob, logcond = raw
    if snap >= 0.75:
        return somp_estimate(Y, K, grid), "few-SOMP"
    if rob > 0.25:
        return somp_estimate(Y, K, grid), "robust-SOMP"
    if (rank < 0.42) or (toe > 0.16) or (rob > 0.15):
        return fd.topk_peaks(specs["Toeplitz-MUSIC"], grid, K), "toe-guard"
    P = fd.fuse_spectra(specs, fd.handcrafted_weights(beta), mode="arith")
    return fd.topk_peaks(P, grid, K), "clean-fuzzy"


def make_candidates_unknownK(Y, grid, kmax=4):
    cand = []
    specs_by_k = {}
    beta_by_k = {}
    raw_by_k = {}
    mode_by_k = {}
    pre = expert_spectra_allK_fast(Y, grid, kmax)
    for k in range(1, kmax + 1):
        specs = pre[k]
        beta, raw = fd.diagnostic_features(Y, k)
        specs_by_k[k] = dict(specs)
        beta_by_k[k] = beta
        raw_by_k[k] = raw
        for name in fd.EXPERTS:
            cand.append(
                {
                    "expert": name,
                    "k": k,
                    "name": f"{name}@K{k}",
                    "est": fd.topk_peaks(specs[name], grid, k),
                    "P": specs[name],
                }
            )
        cand.append(
            {
                "expert": "SOMP",
                "k": k,
                "name": f"SOMP@K{k}",
                "est": somp_estimate(Y, k, grid),
                "P": None,
            }
        )
        est, mode = v6_estimate(Y, specs, beta, raw, grid, k)
        mode_by_k[k] = mode
        cand.append({"expert": "FROST-V6", "k": k, "name": f"FROST-V6@K{k}", "est": est, "P": None})
        w = fd.frost_protected_weights(beta, fd.handcrafted_weights(beta))
        Pl = fd.fuse_spectra(specs, w, mode="lse", temperature=6.0)
        Pa = fd.fuse_spectra(specs, w, mode="arith")
        specs_by_k[k]["Fuzzy-LSE"] = Pl
        specs_by_k[k]["Fuzzy-Arith"] = Pa
        cand.append(
            {
                "expert": "Fuzzy-LSE",
                "k": k,
                "name": f"Fuzzy-LSE@K{k}",
                "est": fd.topk_peaks(Pl, grid, k),
                "P": Pl,
            }
        )
        cand.append(
            {
                "expert": "Fuzzy-Arith",
                "k": k,
                "name": f"Fuzzy-Arith@K{k}",
                "est": fd.topk_peaks(Pa, grid, k),
                "P": Pa,
            }
        )
    return cand, specs_by_k, beta_by_k, raw_by_k, mode_by_k


def projection_metrics(Y, theta):
    theta = np.asarray(theta, float)
    K = len(theta)
    M, T = Y.shape
    if K == 0:
        return 1.0, 1.0, 0.0, 120.0
    A = fd.array_manifold(0.5 * np.arange(M), theta)
    X = np.linalg.pinv(A) @ Y
    res = float(np.linalg.norm(Y - A @ X, "fro") ** 2 / (np.linalg.norm(Y, "fro") ** 2 + 1e-12))
    R0 = fd.sample_cov(Y)
    S = fd.sample_cov(X)
    Rfit = A @ S @ A.conj().T + res * np.trace(R0).real / M * np.eye(M)
    cov = float(np.linalg.norm(R0 - Rfit, "fro") / (np.linalg.norm(R0, "fro") + 1e-12))
    cond = float(np.log1p(np.linalg.cond(A.conj().T @ A + 1e-8 * np.eye(K))))
    sep = float(np.min(np.diff(np.sort(theta)))) if K > 1 else 120.0
    return res, cov, cond, sep


def interp_p(P, grid, theta):
    if P is None or len(theta) == 0:
        return np.zeros(max(1, len(theta)))
    return np.interp(theta, grid, fd.normalize_spectrum(P), left=0, right=0)


def k_global_features(Y, grid, kmax=4):
    R0 = fd.hermitian(fd.sample_cov(Y))
    M, T = Y.shape
    k_mdl, mdl_scores, _ = mdl_order_from_cov(R0, T, kmax)
    k_gap, gr = eigengap_order(R0, kmax)
    vals, _ = fd.eig_sorted(R0)
    vals = np.maximum(vals.real, 1e-12)
    eig = vals[: min(M, 8)] / (vals[0] + 1e-12)
    eig = np.pad(eig, (0, max(0, 8 - len(eig))))[:8]
    gr = np.pad(np.array(gr, float), (0, max(0, kmax - len(gr))))[:kmax]
    md = (mdl_scores - np.min(mdl_scores)) / (np.std(mdl_scores) + 1e-8)
    md = np.pad(md, (0, max(0, kmax - len(md))))[:kmax]
    return np.concatenate(
        [
            np.array([k_mdl / kmax, k_gap / kmax, T / M, M / T, np.log1p(vals[0] / vals[-1])]),
            eig,
            np.log1p(gr),
            md,
        ]
    )


def candidate_feature_matrix_unknownK(Y, grid, cand, specs_by_k, beta_by_k, raw_by_k, kmax=4):
    kg = k_global_features(Y, grid, kmax)
    k_mdl = int(round(kg[0] * kmax))
    k_gap = int(round(kg[1] * kmax))
    C = len(cand)
    Dmat = np.zeros((C, C), float)
    for i in range(C):
        for j in range(i + 1, C):
            d = dist_between_sets_var(cand[i]["est"], cand[j]["est"])
            Dmat[i, j] = Dmat[j, i] = d
    proj = [projection_metrics(Y, c["est"]) for c in cand]
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
            vals = interp_p(specs.get(sn, None), grid, theta)
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
        ohe = np.zeros(len(BASE_EXPERTS))
        ohe[BASE_EXPERTS.index(expert)] = 1.0
        ohk = np.zeros(kmax)
        ohk[k - 1] = 1.0
        feats.append(
            np.concatenate(
                [
                    beta,
                    raw,
                    kg,
                    np.array([res, cov, cond, sep]),
                    cons_all / 20,
                    cons_k / 20,
                    np.array(sf),
                    kf,
                    mem,
                    ohe,
                    ohk,
                    np.outer(ohe, beta).ravel(),
                    np.outer(ohk, beta).ravel(),
                ]
            )
        )
    return np.vstack(feats)


class CandidateRanker:
    def __init__(self, w, mean, std, names=None):
        self.w = w
        self.mean = mean
        self.std = std
        self.names = names or []

    def score(self, X):
        return np.tensordot((X - self.mean) / (self.std + 1e-8), self.w, axes=([-1], [0]))


def make_train_scenario_v9(rng):
    M = 8
    K = int(rng.choice([1, 1, 2, 2, 2, 3, 3, 4]))
    r = int(rng.integers(0, 8))
    if r == 0:
        return fd.Scenario(
            "train_clean",
            M,
            K,
            int(rng.choice([64, 128])),
            float(rng.uniform(0, 12)),
            float(rng.uniform(0, 0.25)),
            "gaussian",
            False,
            float(rng.choice([4, 5, 8])),
        )
    if r == 1:
        return fd.Scenario(
            "train_lowfew",
            M,
            K,
            int(rng.choice([4, 8, 16])),
            float(rng.uniform(-18, -4)),
            float(rng.uniform(0, 0.4)),
            "gaussian",
            False,
            float(rng.choice([3, 5, 8])),
        )
    if r == 2:
        return fd.Scenario(
            "train_coh",
            M,
            K,
            int(rng.choice([32, 64, 128])),
            float(rng.uniform(-5, 8)),
            float(rng.uniform(0.85, 0.995)),
            "gaussian",
            False,
            float(rng.choice([3, 5, 8])),
        )
    if r == 3:
        return fd.Scenario(
            "train_mismatch",
            M,
            K,
            int(rng.choice([16, 32, 64, 128])),
            float(rng.uniform(-5, 8)),
            float(rng.uniform(0, 0.5)),
            "gaussian",
            True,
            float(rng.choice([3, 5, 8])),
        )
    if r == 4:
        return fd.Scenario(
            "train_tail",
            M,
            K,
            int(rng.choice([16, 32, 64, 128])),
            float(rng.uniform(-5, 8)),
            float(rng.uniform(0, 0.4)),
            str(rng.choice(["student", "impulsive"])),
            False,
            float(rng.choice([3, 5, 8])),
        )
    if r == 5:
        return fd.Scenario(
            "train_hard",
            M,
            K,
            int(rng.choice([4, 8, 16])),
            float(rng.uniform(-15, -3)),
            float(rng.uniform(0.75, 0.995)),
            str(rng.choice(["student", "impulsive", "colored"])),
            True,
            float(rng.choice([2, 3, 5])),
        )
    if r == 6:
        return fd.Scenario(
            "train_colored",
            M,
            K,
            int(rng.choice([16, 32, 64])),
            float(rng.uniform(-10, 5)),
            float(rng.uniform(0, 0.7)),
            "colored",
            bool(rng.random() < 0.4),
            float(rng.choice([3, 5, 8])),
        )
    return fd.Scenario(
        "train_close",
        M,
        K,
        int(rng.choice([32, 64, 128])),
        float(rng.uniform(-5, 8)),
        float(rng.uniform(0, 0.6)),
        "gaussian",
        bool(rng.random() < 0.2),
        float(rng.choice([1.5, 2.0, 3.0])),
    )


def train_v9_ranker(grid, n_train=300, seed=20260724, epochs=180, kmax=4):
    rng = np.random.default_rng(seed)
    X = []
    y = []
    oracle = []
    counts = {}
    for i in range(n_train):
        sc = make_train_scenario_v9(rng)
        Y, theta, _ = fd.simulate(sc, rng)
        cand, specs, beta, raw, mode = make_candidates_unknownK(Y, grid, kmax)
        errs = np.array([penalized_set_rmse(c["est"], theta) for c in cand])
        best = int(np.argmin(errs))
        oracle.append(float(errs[best]))
        counts[cand[best]["name"]] = counts.get(cand[best]["name"], 0) + 1
        X.append(candidate_feature_matrix_unknownK(Y, grid, cand, specs, beta, raw, kmax))
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
        grad = np.einsum("nc,ncd->d", P - Y1, Xn) / N + 1.5e-4 * w
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * (grad * grad)
        w -= lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + 1e-8)
        if t in (300, 500):
            lr *= 0.5
    pred = np.argmax(np.einsum("ncd,d->nc", Xn, w), 1)
    info = {
        "n_train": n_train,
        "epochs": epochs,
        "train_acc_best_candidate": float(np.mean(pred == y)),
        "oracle_train_penalized_rmse": float(np.mean(oracle)),
        "feature_dim": int(D),
        "candidates_per_sample": int(C),
        "label_counts_top15": dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:15]),
    }
    return CandidateRanker(w, mean, std), info


def choose_v9(Y, grid, ranker, kmax=4):
    cand, specs, beta, raw, mode = make_candidates_unknownK(Y, grid, kmax)
    X = candidate_feature_matrix_unknownK(Y, grid, cand, specs, beta, raw, kmax)
    scores = ranker.score(X)
    order = np.argsort(scores)[::-1]
    top = cand[int(order[0])]
    gap = float(scores[order[0]] - scores[order[1]])
    R0 = fd.hermitian(fd.sample_cov(Y))
    k_mdl, _, _ = mdl_order_from_cov(R0, Y.shape[1], kmax)
    k_gap, _ = eigengap_order(R0, kmax)
    chosen = top
    m = "ranker"
    if gap < 0.12 and abs(top["k"] - k_mdl) >= 2:
        idxs = [i for i, c in enumerate(cand) if c["k"] == k_mdl]
        bi = max(idxs, key=lambda i: scores[i])
        chosen = cand[bi]
        m = "mdl-guard"
    snap = Y.shape[0] / max(Y.shape[1], 1)
    if snap >= 0.75 and gap < 0.30:
        candidate_ks = sorted(set([top["k"], k_mdl, k_gap, min(kmax, k_mdl + 1)]))
        idxs = [i for i, c in enumerate(cand) if c["expert"] == "SOMP" and c["k"] in candidate_ks]
        if idxs:
            bi = max(idxs, key=lambda i: scores[i])
            chosen = cand[bi]
            m = "few-somp-kguard"
    return chosen, cand, scores, m, k_mdl, k_gap, gap


def baseline_estimate(Y, grid, method, kmax=4):
    R0 = fd.hermitian(fd.sample_cov(Y))
    if method.endswith("-GAP"):
        k, _ = eigengap_order(R0, kmax)
        base = method.replace("-GAP", "")
    else:
        k, _, _ = mdl_order_from_cov(R0, Y.shape[1], kmax)
        base = method.replace("-MDL", "")
    specs = expert_spectra_allK_fast(Y, grid, kmax)[k]
    if base in fd.EXPERTS:
        est = fd.topk_peaks(specs[base], grid, k)
    elif base == "SOMP":
        est = somp_estimate(Y, k, grid)
    elif base == "FROST-V6":
        beta, raw = fd.diagnostic_features(Y, k)
        est, _ = v6_estimate(Y, specs, beta, raw, grid, k)
    else:
        raise ValueError(method)
    return est, k


def test_scenarios_v9():
    return [
        fd.Scenario("U1_clean_K1", 8, 1, 128, 5, 0, "gaussian", False, 5),
        fd.Scenario("U2_clean_K2", 8, 2, 128, 5, 0, "gaussian", False, 5),
        fd.Scenario("U3_clean_K3", 8, 3, 128, 5, 0, "gaussian", False, 5),
        fd.Scenario("U4_lowSNR_few_K2", 8, 2, 8, -10, 0, "gaussian", False, 5),
        fd.Scenario("U5_lowSNR_few_K3", 8, 3, 8, -8, 0, "gaussian", False, 5),
        fd.Scenario("U6_coherent_K2", 8, 2, 64, 0, 0.98, "gaussian", False, 5),
        fd.Scenario("U7_coherent_K3", 8, 3, 64, 0, 0.95, "gaussian", False, 5),
        fd.Scenario("U8_mismatch_K2", 8, 2, 32, 0, 0.2, "gaussian", True, 5),
        fd.Scenario("U9_heavytail_K3", 8, 3, 32, 0, 0, "student", False, 5),
        fd.Scenario("U10_hard_mixed_K2", 8, 2, 8, -8, 0.95, "student", True, 3),
        fd.Scenario("U11_colored_lowSNR_K2", 8, 2, 16, -8, 0.5, "colored", False, 4),
        fd.Scenario("U12_close_K2", 8, 2, 64, 0, 0.3, "gaussian", False, 2),
    ]


def save_outputs(agg, df, dec, info, prefix="/mnt/data/frost_doa_v9_unknownK"):
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
            "FROST-V6-MDL",
            "MUSIC-SCM-GAP",
            "Toeplitz-MUSIC-GAP",
            "SOMP-GAP",
            "FROST-V9",
            "FROST-V10",
            "ORACLE-K-BANK",
            "ORACLE-CANDIDATE",
        ]
        if m in pivot.columns
    ]
    avg = pivot[major].rank(axis=1, method="min").mean().sort_values().reset_index()
    avg.columns = ["method", "avg_rank"]
    avg.to_csv(prefix + "_avg_rank.csv", index=False)
    with open(prefix + "_model_info.json", "w") as f:
        json.dump(info, f, indent=2)
    dec.groupby(["scenario", "v9_expert", "v9_k"]).size().reset_index(name="count").to_csv(
        prefix + "_decision_counts_long.csv", index=False
    )
    try:
        import matplotlib.pyplot as plt

        pm = [
            m
            for m in [
                "MUSIC-SCM-MDL",
                "Toeplitz-MUSIC-MDL",
                "FBSS-MUSIC-MDL",
                "SOMP-MDL",
                "FROST-V6-MDL",
                "SOMP-GAP",
                "FROST-V9",
                "FROST-V10",
                "ORACLE-K-BANK",
                "ORACLE-CANDIDATE",
            ]
            if m in pivot.columns
        ]
        ax = pivot[pm].plot(kind="bar", figsize=(16, 6))
        ax.set_ylabel("Penalized set RMSE (degree)")
        ax.set_title("FROST-DOA v9 unknown-K benchmark")
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(prefix + "_rmse_bar.png", dpi=220)
        plt.close()
        ax = pivot[pm].plot(kind="bar", figsize=(16, 6), logy=True)
        ax.set_ylabel("Penalized set RMSE (degree, log)")
        ax.set_title("FROST-DOA v9 unknown-K benchmark, log scale")
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(prefix + "_rmse_bar_log.png", dpi=220)
        plt.close()
        ax = kpiv[pm].plot(kind="bar", figsize=(16, 6))
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
        print("plot skipped", e)
    with open(prefix + "_report.md", "w") as f:
        f.write(
            "# FROST-DOA v9 unknown-K benchmark report\n\nSynthetic ULA benchmark; M=8, K not given to deployable methods, candidate K=1..4, grid step 0.5 deg.\n\n"
        )
        f.write(
            "Penalized set RMSE uses a 25 deg cardinality penalty for missed/false sources. ORACLE baselines are nondeployable.\n\n"
        )
        f.write("## Training info\n\n```json\n" + json.dumps(info, indent=2) + "\n```\n\n")
        f.write(
            "## Mean penalized set RMSE\n\n"
            + pivot[major].round(3).to_markdown()
            + "\n\n## K accuracy (%)\n\n"
            + kpiv[major].round(1).to_markdown()
            + "\n\n## Average rank\n\n"
            + avg.round(3).to_markdown(index=False)
            + "\n"
        )
    return pivot, kpiv, avg


def choose_v10(Y, grid, ranker, kmax=4):
    """FROST-V10: v9 unknown-K ranker with champion safety policy for clean/close/mismatch regimes."""
    chosen, cand, scores, mode, k_mdl, k_gap, score_gap = choose_v9(Y, grid, ranker, kmax)
    R0 = fd.hermitian(fd.sample_cov(Y))
    beta, raw = fd.diagnostic_features(Y, k_mdl)
    _, raw_gap = fd.diagnostic_features(Y, k_gap)
    snap, gap, rank, toe, rob, logcond = raw
    rob_max = max(rob, raw_gap[4])
    toe_max = max(toe, raw_gap[3])
    # High-risk regimes where V9 ranker is needed: few snapshots, colored/robust deviation, coherent rank collapse.
    coherent_like = snap <= 0.15 and toe_max > 0.35 and (k_mdl == 1 or rank > 0.90)
    if snap >= 0.45 or rob_max > 0.20 or coherent_like:
        return chosen["est"], chosen["k"], f"v9:{mode}", chosen["expert"]
    # Clean high-support regime: classical/Toeplitz source enumeration is sharper.
    if snap <= 0.08 and toe < 0.18 and rob < 0.10:
        if k_mdl >= 3:
            est, k = baseline_estimate(Y, grid, "Toeplitz-MUSIC-MDL", kmax)
            return est, k, "clean-toeplitz-mdl", "Toeplitz-MUSIC"
        else:
            est, k = baseline_estimate(Y, grid, "MUSIC-SCM-MDL", kmax)
            return est, k, "clean-scm-mdl", "MUSIC-SCM"
    # Moderate mismatch / close sources: V6 Toeplitz guard is safer than the aggressive ranker.
    est, k = baseline_estimate(Y, grid, "FROST-V6-MDL", kmax)
    return est, k, "safe-v6-mdl", "FROST-V6"
