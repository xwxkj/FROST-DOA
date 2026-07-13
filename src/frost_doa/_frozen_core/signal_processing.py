"""Numerically frozen signal-processing primitives used by FROST-DOA.

This module preserves the implementation used for the published benchmark.
Public users should normally import :class:`frost_doa.FROSTDOAEstimator`.
"""

import numpy as np
from dataclasses import dataclass
import itertools, math, json, time, os
from typing import Dict, Tuple, List

try:
    import torch
except Exception:
    torch = None

# -----------------------------
# Utilities
# -----------------------------


def sigmoid(x):
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))


def steering_matrix(M: int, grid_deg: np.ndarray, positions=None):
    if positions is None:
        positions = 0.5 * np.arange(M)  # in wavelengths
    theta = np.deg2rad(grid_deg)
    phase = 2 * np.pi * np.asarray(positions)[:, None] * np.sin(theta)[None, :]
    return np.exp(1j * phase) / np.sqrt(M)


def array_manifold(positions: np.ndarray, thetas_deg: np.ndarray):
    theta = np.deg2rad(thetas_deg)
    phase = 2 * np.pi * positions[:, None] * np.sin(theta)[None, :]
    return np.exp(1j * phase) / np.sqrt(len(positions))


def sample_cov(Y):
    return (Y @ Y.conj().T) / max(1, Y.shape[1])


def hermitian(R):
    return 0.5 * (R + R.conj().T)


def toeplitz_project(R):
    M = R.shape[0]
    out = np.zeros_like(R, dtype=np.complex128)
    for k in range(-(M - 1), M):
        vals = []
        for i in range(M):
            j = i - k
            if 0 <= j < M:
                vals.append(R[i, j])
        avg = np.mean(vals) if vals else 0
        for i in range(M):
            j = i - k
            if 0 <= j < M:
                out[i, j] = avg
    return hermitian(out)


def shrinkage_cov(R, T=None, alpha=None):
    M = R.shape[0]
    if alpha is None:
        if T is None:
            alpha = 0.25
        else:
            alpha = float(np.clip(M / (T + M), 0.05, 0.75))
    return hermitian((1 - alpha) * R + alpha * (np.trace(R).real / M) * np.eye(M))


def fbss_cov(Y, K=2, L=None):
    M, T = Y.shape
    if L is None:
        # enough subarrays for smoothing but preserve aperture
        L = max(K + 1, min(M - 1, M - K + 1))
    L = int(np.clip(L, K + 1, M))
    P = M - L + 1
    R = np.zeros((L, L), dtype=np.complex128)
    for p in range(P):
        Ys = Y[p : p + L, :]
        R += sample_cov(Ys)
    R /= P
    J = np.fliplr(np.eye(L))
    Rfb = 0.5 * (R + J @ R.conj() @ J)
    return hermitian(Rfb)


def tyler_cov(Y, n_iter=10, loading=1e-3):
    M, T = Y.shape
    R = sample_cov(Y) + loading * np.trace(sample_cov(Y)).real / M * np.eye(M)
    R = hermitian(R)
    for _ in range(n_iter):
        try:
            RiY = np.linalg.solve(R + loading * np.eye(M), Y)
        except np.linalg.LinAlgError:
            RiY = np.linalg.pinv(R + loading * np.eye(M)) @ Y
        denom = np.sum(Y.conj() * RiY, axis=0).real
        denom = np.maximum(denom, 1e-8)
        Rn = M * (Y / denom[None, :]) @ Y.conj().T / max(T, 1)
        Rn = hermitian(Rn)
        tr = np.trace(Rn).real
        if tr <= 0 or not np.isfinite(tr):
            break
        R = Rn * (M / tr)
        R = hermitian(R + loading * np.eye(M))
    # rescale to SCM trace for consistent spectra
    Rs = sample_cov(Y)
    tr_s = np.trace(Rs).real
    tr_r = np.trace(R).real
    if tr_r > 0:
        R = R * (tr_s / tr_r)
    return hermitian(R)


def eig_sorted(R):
    vals, vecs = np.linalg.eigh(hermitian(R))
    idx = np.argsort(vals)[::-1]
    return vals[idx].real, vecs[:, idx]


def music_spectrum(R, K, grid_deg, positions=None, diagonal_loading=1e-8):
    M = R.shape[0]
    if positions is None:
        positions = 0.5 * np.arange(M)
    vals, vecs = eig_sorted(R)
    K_eff = int(np.clip(K, 0, M - 1))
    Un = vecs[:, K_eff:]
    Pn = Un @ Un.conj().T
    A = steering_matrix(M, grid_deg, positions)
    denom = np.sum(A.conj() * (Pn @ A), axis=0).real
    denom = np.maximum(denom, diagonal_loading)
    P = 1.0 / denom
    return P.real


def normalize_spectrum(P):
    P = np.asarray(P, dtype=float)
    P = np.maximum(P, 1e-12)
    # robust log normalization: make peak scale comparable
    return P / (np.max(P) + 1e-12)


def topk_peaks(P, grid_deg, K, min_sep_deg=1.0):
    P = np.asarray(P).copy()
    grid = np.asarray(grid_deg)
    step = float(np.median(np.diff(grid)))
    suppress = max(1, int(round(min_sep_deg / step)))
    estimates = []
    Pc = P.copy()
    for _ in range(K):
        idx = int(np.argmax(Pc))
        # parabolic interpolation around grid peak on log spectrum
        theta = grid[idx]
        if 0 < idx < len(grid) - 1:
            y0, y1, y2 = np.log(np.maximum(P[idx - 1 : idx + 2], 1e-12))
            denom = y0 - 2 * y1 + y2
            if abs(denom) > 1e-9:
                delta = 0.5 * (y0 - y2) / denom
                delta = float(np.clip(delta, -1.0, 1.0))
                theta = grid[idx] + delta * step
        estimates.append(float(theta))
        lo = max(0, idx - suppress)
        hi = min(len(Pc), idx + suppress + 1)
        Pc[lo:hi] = -np.inf
    return np.array(sorted(estimates))


def rmse_match(est, true):
    est = np.asarray(est)
    true = np.asarray(true)
    K = len(true)
    best = 1e9
    for perm in itertools.permutations(range(K)):
        diff = est[list(perm)] - true
        val = np.mean(diff**2)
        if val < best:
            best = val
    return float(np.sqrt(best))


def mae_match(est, true):
    est = np.asarray(est)
    true = np.asarray(true)
    K = len(true)
    best = 1e9
    for perm in itertools.permutations(range(K)):
        val = np.mean(np.abs(est[list(perm)] - true))
        if val < best:
            best = val
    return float(best)


# -----------------------------
# Data generation
# -----------------------------


@dataclass
class Scenario:
    name: str
    M: int = 8
    K: int = 2
    T: int = 32
    snr_db: float = 0.0
    coherent_rho: float = 0.0
    noise: str = "gaussian"  # gaussian, student, impulsive, colored
    mismatch: bool = False
    min_sep: float = 4.0
    angle_range: Tuple[float, float] = (-60.0, 60.0)


def random_doas(K, rng, angle_range=(-60, 60), min_sep=4.0):
    lo, hi = angle_range
    for _ in range(10000):
        theta = np.sort(rng.uniform(lo, hi, K))
        if K == 1 or np.min(np.diff(theta)) >= min_sep:
            return theta
    # fallback evenly spaced
    return np.linspace(lo + 10, hi - 10, K)


def generate_sources(K, T, rng, rho=0.0):
    if K == 1 or rho <= 1e-6:
        S = (rng.standard_normal((K, T)) + 1j * rng.standard_normal((K, T))) / np.sqrt(2)
    else:
        base = (rng.standard_normal((1, T)) + 1j * rng.standard_normal((1, T))) / np.sqrt(2)
        S = []
        for k in range(K):
            e = (rng.standard_normal((1, T)) + 1j * rng.standard_normal((1, T))) / np.sqrt(2)
            # alternating phases avoid exact duplicates while controlling correlation magnitude
            phase = np.exp(1j * 2 * np.pi * k / K)
            sk = rho * phase * base + np.sqrt(max(0, 1 - rho**2)) * e
            S.append(sk[0])
        S = np.vstack(S)
    # normalize per source
    S = S / (np.sqrt(np.mean(np.abs(S) ** 2, axis=1, keepdims=True)) + 1e-12)
    return S


def generate_noise(M, T, rng, kind="gaussian", noise_power=1.0):
    if kind == "gaussian":
        N = (rng.standard_normal((M, T)) + 1j * rng.standard_normal((M, T))) / np.sqrt(2)
    elif kind == "colored":
        rho = 0.7
        C = rho ** np.abs(np.subtract.outer(np.arange(M), np.arange(M)))
        vals, vecs = np.linalg.eigh(C)
        Cs = vecs @ np.diag(np.sqrt(np.maximum(vals, 0))) @ vecs.T
        W = (rng.standard_normal((M, T)) + 1j * rng.standard_normal((M, T))) / np.sqrt(2)
        N = Cs @ W
    elif kind == "student":
        df = 3.0
        W = (rng.standard_normal((M, T)) + 1j * rng.standard_normal((M, T))) / np.sqrt(2)
        scale = np.sqrt(df / np.maximum(rng.chisquare(df, size=(1, T)), 1e-12))
        N = W * scale
    elif kind == "impulsive":
        W = (rng.standard_normal((M, T)) + 1j * rng.standard_normal((M, T))) / np.sqrt(2)
        impulses = rng.random((1, T)) < 0.08
        amp = np.where(impulses, 12.0, 1.0)
        N = W * amp
    else:
        raise ValueError(kind)
    N = N / (np.sqrt(np.mean(np.abs(N) ** 2)) + 1e-12) * np.sqrt(noise_power)
    return N


def simulate(scen: Scenario, rng):
    M, K, T = scen.M, scen.K, scen.T
    theta = random_doas(K, rng, scen.angle_range, scen.min_sep)
    positions_nom = 0.5 * np.arange(M)
    positions = positions_nom.copy()
    gains = np.ones(M, dtype=np.complex128)
    if scen.mismatch:
        # moderate calibration/mutual mismatch
        pos_sigma = rng.uniform(0.0, 0.035)  # wavelength
        positions = positions + rng.normal(0, pos_sigma, size=M)
        gain_db = rng.normal(0, rng.uniform(0.0, 1.2), size=M)
        phase_deg = rng.normal(0, rng.uniform(0.0, 10.0), size=M)
        gains = (10 ** (gain_db / 20)) * np.exp(1j * np.deg2rad(phase_deg))
    A = array_manifold(positions, theta)
    A = gains[:, None] * A
    S = generate_sources(K, T, rng, rho=scen.coherent_rho)
    X = A @ S
    sig_power = np.mean(np.abs(X) ** 2)
    noise_power = sig_power / (10 ** (scen.snr_db / 10))
    N = generate_noise(M, T, rng, kind=scen.noise, noise_power=noise_power)
    Y = X + N
    return Y, theta, positions_nom


# -----------------------------
# Expert spectra & fuzzy gate
# -----------------------------
EXPERTS = ["MUSIC-SCM", "Shrink-MUSIC", "Toeplitz-MUSIC", "FBSS-MUSIC", "Tyler-MUSIC"]


def expert_spectra(Y, K, grid_deg):
    M, T = Y.shape
    R0 = hermitian(sample_cov(Y))
    spectra = {}
    spectra["MUSIC-SCM"] = music_spectrum(R0, K, grid_deg, positions=0.5 * np.arange(M))
    spectra["Shrink-MUSIC"] = music_spectrum(
        shrinkage_cov(R0, T=T), K, grid_deg, positions=0.5 * np.arange(M)
    )
    spectra["Toeplitz-MUSIC"] = music_spectrum(
        toeplitz_project(R0), K, grid_deg, positions=0.5 * np.arange(M)
    )
    L = max(K + 1, min(M - 1, M - K + 1))
    Rfb = fbss_cov(Y, K, L=L)
    spectra["FBSS-MUSIC"] = music_spectrum(
        Rfb, K, grid_deg, positions=0.5 * np.arange(Rfb.shape[0])
    )
    Rty = tyler_cov(Y, n_iter=4)
    spectra["Tyler-MUSIC"] = music_spectrum(Rty, K, grid_deg, positions=0.5 * np.arange(M))
    return spectra


def diagnostic_features(Y, K):
    M, T = Y.shape
    R0 = hermitian(sample_cov(Y))
    vals, _ = eig_sorted(R0)
    eps = 1e-12
    gap = (
        np.log((vals[K - 1] + eps) / (vals[K] + eps))
        if K < M
        else np.log((vals[-2] + eps) / (vals[-1] + eps))
    )
    rank_spread = (vals[K - 1] + eps) / (vals[0] + eps)
    Rtoe = toeplitz_project(R0)
    toe_dev = np.linalg.norm(R0 - Rtoe, "fro") / (np.linalg.norm(R0, "fro") + eps)
    Rty = tyler_cov(Y, n_iter=3)
    rob_diff = np.linalg.norm(R0 - Rty, "fro") / (np.linalg.norm(R0, "fro") + eps)
    cond = (vals[0] + eps) / (vals[-1] + eps)
    snap_ratio = M / max(T, 1)
    # fuzzy memberships, bounded in [0,1]
    few = sigmoid((snap_ratio - 0.75) / 0.30)
    very_few = sigmoid((snap_ratio - 1.20) / 0.25)
    low_snr = sigmoid((0.75 - gap) / 0.35)
    high_snr = 1 - low_snr
    # small kth eigenvalue relative to first often indicates high coherence/rank collapse
    coh = sigmoid((0.10 - rank_spread) / 0.04)
    ng = sigmoid((rob_diff - 0.45) / 0.18)
    mismatch = sigmoid((toe_dev - 0.45) / 0.15)
    close_or_uncertain = sigmoid((1.0 - gap) / 0.35)
    # Rule activations, interpretable; normalized later
    beta = np.array(
        [
            high_snr * (1 - few) * (1 - coh) * (1 - ng),  # R1: clean / high support
            low_snr * (1 - very_few),  # R2: low SNR -> shrink/toeplitz
            few,  # R3: few snapshots
            coh,  # R4: coherent/rank collapsed
            ng,  # R5: non-Gaussian/heavy-tail
            mismatch,  # R6: calibration mismatch/finite-sample deviation
            close_or_uncertain,  # R7: close/uncertain eigengap
            1.0,  # R8: safety prior
        ],
        dtype=float,
    )
    beta = np.maximum(beta, 1e-6)
    beta = beta / (np.sum(beta) + 1e-12)
    raw = np.array([snap_ratio, gap, rank_spread, toe_dev, rob_diff, np.log(cond + 1)], dtype=float)
    return beta, raw


def handcrafted_weights(beta):
    # maps rules to experts; rows=rules, cols=experts
    C = np.array(
        [
            [3.0, 0.2, 0.1, 0.0, 0.0],  # clean -> SCM
            [0.1, 2.3, 1.0, 0.1, 0.1],  # low SNR -> shrink/toe
            [0.1, 2.2, 1.4, 0.2, 0.1],  # few snapshots -> shrink/toe
            [0.0, 0.4, 1.0, 3.0, 0.0],  # coherent -> FBSS/toe
            [0.0, 0.4, 0.2, 0.0, 3.0],  # non-Gaussian -> Tyler
            [0.1, 1.0, 0.8, 0.2, 0.5],  # mismatch -> conservative fusion
            [0.2, 0.8, 0.7, 0.7, 0.4],  # uncertain -> diverse fusion
            [0.6, 0.6, 0.5, 0.4, 0.4],  # prior
        ]
    )
    logits = beta @ C
    exps = np.exp(logits - np.max(logits))
    return exps / np.sum(exps)


def fuse_spectra(spectra: Dict[str, np.ndarray], weights, mode="geom", temperature=6.0):
    Ps = []
    for name in EXPERTS:
        Ps.append(normalize_spectrum(spectra[name]))
    Ps = np.vstack(Ps)
    w = np.asarray(weights, dtype=float)
    w = np.maximum(w, 1e-8)
    w = w / np.sum(w)
    if mode == "geom":
        logP = np.sum(w[:, None] * np.log(np.maximum(Ps, 1e-12)), axis=0)
        P = np.exp(logP - np.max(logP))
    elif mode == "arith":
        P = np.sum(w[:, None] * Ps, axis=0)
    elif mode == "lse":
        # winner-preserving weighted generalized mean; avoids bad experts suppressing good peaks
        tau = temperature
        P = (np.sum(w[:, None] * (np.maximum(Ps, 1e-12) ** tau), axis=0)) ** (1.0 / tau)
    elif mode == "max":
        P = np.max(w[:, None] * Ps, axis=0)
    else:
        raise ValueError(mode)
    return normalize_spectrum(P)


def frost_protected_weights(beta, learned_w=None):
    # Rule-protected fuzzy gate: keep learned consequent, but impose Toeplitz/FBSS/Tyler safeguards.
    # beta indices: clean, low_snr, few, coherent, nonGaussian, mismatch, uncertain, prior
    if learned_w is None:
        w = handcrafted_weights(beta)
    else:
        w = np.array(learned_w, dtype=float).copy()
    # strong interpretable safeguards discovered in validation
    w[2] += 0.65 * (beta[3] + beta[5] + beta[6])  # Toeplitz for coherent/mismatch/uncertain
    w[3] += 0.35 * beta[3]  # FBSS for coherent
    w[4] += 0.40 * beta[4]  # Tyler for non-Gaussian
    w[1] += 0.35 * (beta[1] + beta[2])  # shrinkage for low SNR/few snapshots
    # Do not let SCM dominate when uncertainty is high.
    w[0] *= 1.0 - 0.55 * (beta[1] + beta[2] + beta[3] + beta[4] + beta[5] + beta[6])
    w = np.maximum(w, 1e-8)
    w = w / np.sum(w)
    return w


def estimate_method(Y, K, grid, method, gate=None):
    specs = expert_spectra(Y, K, grid)
    if method in EXPERTS:
        P = specs[method]
        w = None
    elif method == "FROST-H":
        beta, _ = diagnostic_features(Y, K)
        w = handcrafted_weights(beta)
        P = fuse_spectra(specs, w)
    elif method == "FROST-L":
        beta, _ = diagnostic_features(Y, K)
        w = gate.predict_weights(beta[None, :])[0]
        P = fuse_spectra(specs, w)
    elif method == "ORACLE-EXPERT":
        raise ValueError("oracle needs truth")
    else:
        raise ValueError(method)
    return topk_peaks(P, grid, K, min_sep_deg=1.0), P, w


# -----------------------------
# Neuro-fuzzy consequent learner
# -----------------------------
class FuzzyGate:
    def __init__(self, W=None, b=None):
        self.W = W
        self.b = b

    def predict_weights(self, beta_np):
        logits = beta_np @ self.W + self.b
        logits = logits - np.max(logits, axis=1, keepdims=True)
        e = np.exp(logits)
        return e / np.sum(e, axis=1, keepdims=True)


def train_gate(Xbeta, ylabels, epochs=500, lr=0.05, seed=0):
    # Fast CPU neuro-fuzzy consequent learner using multinomial logistic regression.
    # It learns consequent expert weights from fuzzy rule activations.
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=500, C=3.0, random_state=seed)
    clf.fit(Xbeta, ylabels)
    R = Xbeta.shape[1]
    E = len(EXPERTS)
    W = np.zeros((R, E), dtype=float)
    b = np.full(E, -6.0, dtype=float)
    for idx, cls in enumerate(clf.classes_):
        W[:, int(cls)] = clf.coef_[idx]
        b[int(cls)] = clf.intercept_[idx]
    return FuzzyGate(W, b)


# -----------------------------
# Dataset construction and evaluation
# -----------------------------


def make_random_scenario(rng):
    M = 8
    K = 2
    T = int(rng.choice([4, 8, 16, 32, 64, 128]))
    snr = float(rng.uniform(-18, 12))
    # mixture of coherent and non-coherent cases
    if rng.random() < 0.35:
        rho = float(rng.uniform(0.85, 0.995))
    else:
        rho = float(rng.uniform(0.0, 0.4))
    noise = str(rng.choice(["gaussian", "gaussian", "colored", "student", "impulsive"]))
    mismatch = bool(rng.random() < 0.35)
    min_sep = float(rng.choice([2.0, 3.0, 5.0, 8.0]))
    return Scenario(
        name="trainmix",
        M=M,
        K=K,
        T=T,
        snr_db=snr,
        coherent_rho=rho,
        noise=noise,
        mismatch=mismatch,
        min_sep=min_sep,
    )


def build_gate_training(n=1200, grid=None, seed=123):
    rng = np.random.default_rng(seed)
    Xbeta = []
    y = []
    rmses = []
    for i in range(n):
        scen = make_random_scenario(rng)
        Y, theta, _ = simulate(scen, rng)
        beta, _ = diagnostic_features(Y, scen.K)
        specs = expert_spectra(Y, scen.K, grid)
        errs = []
        for name in EXPERTS:
            est = topk_peaks(specs[name], grid, scen.K, min_sep_deg=1.0)
            errs.append(rmse_match(est, theta))
        Xbeta.append(beta)
        y.append(int(np.argmin(errs)))
        rmses.append(errs)
    return np.vstack(Xbeta), np.array(y), np.array(rmses)


def evaluate_scenario(
    scen: Scenario, methods: List[str], grid, n=200, seed=0, gate=None, verbose=False
):
    rng = np.random.default_rng(seed)
    rows = []
    per_sample = []
    for i in range(n):
        Y, theta, _ = simulate(scen, rng)
        specs = expert_spectra(Y, scen.K, grid)
        # precompute expert estimates
        ests = {}
        Pdict = {}
        for name in EXPERTS:
            Pdict[name] = specs[name]
            ests[name] = topk_peaks(specs[name], grid, scen.K, min_sep_deg=1.0)
        # FROSTs
        beta, _ = diagnostic_features(Y, scen.K)
        weightsH = handcrafted_weights(beta)
        P_H = fuse_spectra(specs, weightsH)
        ests["FROST-H"] = topk_peaks(P_H, grid, scen.K, min_sep_deg=1.0)
        if gate is not None:
            weightsL = gate.predict_weights(beta[None, :])[0]
            P_L = fuse_spectra(specs, weightsL)
            ests["FROST-L"] = topk_peaks(P_L, grid, scen.K, min_sep_deg=1.0)
        # Oracle expert chooses lowest RMSE among base experts (not deployable)
        errs_ex = [rmse_match(ests[name], theta) for name in EXPERTS]
        best_idx = int(np.argmin(errs_ex))
        ests["ORACLE-EXPERT"] = ests[EXPERTS[best_idx]]
        for m in methods:
            if m not in ests:
                continue
            e = rmse_match(ests[m], theta)
            a = mae_match(ests[m], theta)
            rows.append((m, e, a))
        per_sample.append(
            {
                "theta": theta.tolist(),
                "best_expert": EXPERTS[best_idx],
                "best_rmse": errs_ex[best_idx],
            }
        )
    # aggregate
    out = []
    for m in methods:
        vals = np.array([r[1] for r in rows if r[0] == m])
        maes = np.array([r[2] for r in rows if r[0] == m])
        if len(vals) == 0:
            continue
        out.append(
            {
                "scenario": scen.name,
                "method": m,
                "rmse_mean": float(np.mean(vals)),
                "rmse_median": float(np.median(vals)),
                "mae_mean": float(np.mean(maes)),
                "outlier_5deg_pct": float(np.mean(vals > 5.0) * 100),
                "n": int(len(vals)),
            }
        )
    return out


def main():
    t0 = time.time()
    grid = np.arange(-60, 60.0001, 0.5)
    print("Grid points", len(grid), flush=True)
    print("Building fuzzy-gate training data...", flush=True)
    Xbeta, y, rmses = build_gate_training(n=260, grid=grid, seed=20260707)
    print(
        "Expert label distribution",
        {EXPERTS[i]: int(np.sum(y == i)) for i in range(len(EXPERTS))},
        flush=True,
    )
    gate = train_gate(Xbeta, y, epochs=180, lr=0.08, seed=2026)
    pred = np.argmax(gate.predict_weights(Xbeta), axis=1)
    acc = float(np.mean(pred == y))
    print("Gate train expert-label acc", acc, flush=True)
    print("Gate W", np.round(gate.W, 3), "b", np.round(gate.b, 3), flush=True)

    scenarios = [
        Scenario(
            "A_clean_high_support",
            M=8,
            K=2,
            T=128,
            snr_db=5,
            coherent_rho=0.0,
            noise="gaussian",
            mismatch=False,
            min_sep=5,
        ),
        Scenario(
            "B_lowSNR_few_snapshots",
            M=8,
            K=2,
            T=8,
            snr_db=-10,
            coherent_rho=0.0,
            noise="gaussian",
            mismatch=False,
            min_sep=5,
        ),
        Scenario(
            "C_coherent_sources",
            M=8,
            K=2,
            T=64,
            snr_db=0,
            coherent_rho=0.98,
            noise="gaussian",
            mismatch=False,
            min_sep=5,
        ),
        Scenario(
            "D_array_mismatch",
            M=8,
            K=2,
            T=32,
            snr_db=0,
            coherent_rho=0.2,
            noise="gaussian",
            mismatch=True,
            min_sep=5,
        ),
        Scenario(
            "E_heavy_tail_noise",
            M=8,
            K=2,
            T=32,
            snr_db=0,
            coherent_rho=0.0,
            noise="student",
            mismatch=False,
            min_sep=5,
        ),
        Scenario(
            "F_hard_mixed",
            M=8,
            K=2,
            T=8,
            snr_db=-8,
            coherent_rho=0.95,
            noise="student",
            mismatch=True,
            min_sep=3,
        ),
    ]
    methods = EXPERTS + ["FROST-H", "FROST-L", "ORACLE-EXPERT"]
    allres = []
    for si, scen in enumerate(scenarios):
        print("Evaluating", scen.name, flush=True)
        res = evaluate_scenario(scen, methods, grid, n=80, seed=1000 + si, gate=gate)
        allres.extend(res)
        # print compact best
        best = min(res, key=lambda x: x["rmse_mean"])
        print(" best", best, flush=True)
    import pandas as pd

    df = pd.DataFrame(allres)
    out_csv = "/mnt/data/frost_doa_prelim_results.csv"
    df.to_csv(out_csv, index=False)
    # create pivot tables
    piv = df.pivot(index="scenario", columns="method", values="rmse_mean")
    piv.to_csv("/mnt/data/frost_doa_prelim_rmse_pivot.csv")
    # save gate info
    info = {
        "experts": EXPERTS,
        "grid_step": 0.5,
        "gate_train_acc": acc,
        "W": gate.W.tolist(),
        "b": gate.b.tolist(),
        "runtime_sec": time.time() - t0,
    }
    with open("/mnt/data/frost_gate_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print("\nRMSE mean pivot:")
    print(piv.round(3).to_string())
    print("\nSaved", out_csv, "runtime", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
