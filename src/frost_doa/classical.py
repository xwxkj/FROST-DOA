"""Classical baseline implementations using the same array and scoring conventions."""

from __future__ import annotations

from typing import Dict
import numpy as np

from ._frozen_core import base_selector
from ._frozen_core import signal_processing as sp

GRID_DEG = np.linspace(-60.0, 60.0, 241)


def estimate_source_number_mdl(Y: np.ndarray, max_sources: int = 4) -> int:
    """Wax–Kailath MDL estimate, constrained to at least one source."""
    R = sp.hermitian(sp.sample_cov(Y))
    p_hat, _, _ = base_selector.mdl_order_from_cov(R, Y.shape[1], max_sources)
    return int(p_hat)


def music(Y: np.ndarray, p_hat: int, covariance: str = "sample") -> np.ndarray:
    """MUSIC, Toeplitz-MUSIC, or FBSS-MUSIC with local peak refinement."""
    M, T = Y.shape
    if covariance == "sample":
        R = sp.hermitian(sp.sample_cov(Y))
        positions = 0.5 * np.arange(M)
    elif covariance == "toeplitz":
        R = sp.toeplitz_project(sp.hermitian(sp.sample_cov(Y)))
        positions = 0.5 * np.arange(M)
    elif covariance == "fbss":
        L = max(p_hat + 1, min(M - 1, M - p_hat + 1))
        R = sp.fbss_cov(Y, p_hat, L=L)
        positions = 0.5 * np.arange(R.shape[0])
    else:
        raise ValueError(f"Unsupported covariance mode: {covariance}")
    spectrum = sp.music_spectrum(R, p_hat, GRID_DEG, positions=positions)
    return sp.topk_peaks(spectrum, GRID_DEG, p_hat, min_sep_deg=1.0)


def root_music(Y: np.ndarray, p_hat: int) -> np.ndarray:
    """Root-MUSIC for a half-wavelength ULA, with MUSIC fallback if roots fail."""
    M = Y.shape[0]
    R = sp.hermitian(sp.sample_cov(Y))
    _, vectors = sp.eig_sorted(R)
    p_eff = int(np.clip(p_hat, 1, M - 1))
    Un = vectors[:, p_eff:]
    Pn = Un @ Un.conj().T
    coefficients = [np.sum(np.diag(Pn, k=lag)) for lag in range(-(M - 1), M)]
    polynomial = np.asarray(coefficients[::-1], dtype=np.complex128)
    if np.all(np.abs(polynomial) < 1e-14):
        return music(Y, p_hat, "sample")
    roots = np.roots(polynomial)
    roots = roots[np.isfinite(roots)]
    inside = roots[np.abs(roots) < 1.0]
    if len(inside) < p_eff:
        inside = roots
    selected = inside[np.argsort(np.abs(np.abs(inside) - 1.0))[:p_eff]]
    angles = []
    for root in selected:
        sine = np.angle(root) / np.pi
        if -1.0 <= sine <= 1.0:
            theta = float(np.rad2deg(np.arcsin(np.clip(sine, -1.0, 1.0))))
            if -60.0 <= theta <= 60.0:
                angles.append(theta)
    return (
        np.asarray(sorted(angles[:p_hat]), dtype=float)
        if len(angles) >= p_hat
        else music(Y, p_hat, "sample")
    )


def esprit(Y: np.ndarray, p_hat: int) -> np.ndarray:
    """Least-squares ESPRIT for a half-wavelength ULA."""
    M = Y.shape[0]
    R = sp.hermitian(sp.sample_cov(Y))
    _, vectors = sp.eig_sorted(R)
    p_eff = int(np.clip(p_hat, 1, M - 1))
    Es = vectors[:, :p_eff]
    try:
        Phi = np.linalg.pinv(Es[:-1, :], rcond=1e-8) @ Es[1:, :]
        eigenvalues = np.linalg.eigvals(Phi)
    except np.linalg.LinAlgError:
        return music(Y, p_hat, "sample")
    selected = eigenvalues[np.argsort(np.abs(np.abs(eigenvalues) - 1.0))[:p_eff]]
    angles = []
    for value in selected:
        sine = np.angle(value) / np.pi
        if -1.0 <= sine <= 1.0:
            theta = float(np.rad2deg(np.arcsin(np.clip(sine, -1.0, 1.0))))
            if -60.0 <= theta <= 60.0:
                angles.append(theta)
    return (
        np.asarray(sorted(angles[:p_hat]), dtype=float)
        if len(angles) >= p_hat
        else music(Y, p_hat, "sample")
    )


def somp(Y: np.ndarray, p_hat: int) -> np.ndarray:
    """Simultaneous orthogonal matching pursuit on the common 0.5° grid."""
    return base_selector.somp_estimate(Y, p_hat, GRID_DEG, min_sep_deg=1.0)


def sbl(Y: np.ndarray, p_hat: int, iterations: int = 20) -> np.ndarray:
    """Deterministic multi-snapshot EM-style sparse Bayesian learning baseline."""
    M = Y.shape[0]
    A = sp.steering_matrix(M, GRID_DEG)
    R = sp.hermitian(sp.sample_cov(Y))
    eigenvalues, _ = sp.eig_sorted(R)
    sigma2 = float(max(np.median(eigenvalues[max(1, p_hat) :]), 1e-4 * np.mean(eigenvalues)))
    gamma = np.mean(np.abs(A.conj().T @ Y) ** 2, axis=1)
    gamma = gamma / (np.max(gamma) + 1e-12) * max(np.trace(R).real / M, 1e-6)
    gamma = np.maximum(gamma, 1e-8)
    identity = np.eye(M)
    for _ in range(iterations):
        covariance = (A * gamma[None, :]) @ A.conj().T + sigma2 * identity
        covariance = sp.hermitian(covariance)
        try:
            inverse_A = np.linalg.solve(covariance, A)
            inverse_Y = np.linalg.solve(covariance, Y)
        except np.linalg.LinAlgError:
            inverse = np.linalg.pinv(covariance, rcond=1e-8)
            inverse_A = inverse @ A
            inverse_Y = inverse @ Y
        posterior_mean = gamma[:, None] * (A.conj().T @ inverse_Y)
        posterior_variance = gamma - gamma**2 * np.sum(A.conj() * inverse_A, axis=0).real
        posterior_variance = np.maximum(posterior_variance, 0.0)
        update = np.mean(np.abs(posterior_mean) ** 2, axis=1) + posterior_variance
        gamma = 0.85 * gamma + 0.15 * np.maximum(update, 1e-10)
        signal_power = np.sum(gamma)
        total_power = np.trace(R).real
        sigma2 = float(np.clip((total_power - signal_power) / M, 1e-6, total_power / M))
    return sp.topk_peaks(gamma, GRID_DEG, p_hat, min_sep_deg=1.0)


METHODS: Dict[str, callable] = {
    "MUSIC + MDL": lambda Y, p: music(Y, p, "sample"),
    "Root-MUSIC + MDL": root_music,
    "ESPRIT + MDL": esprit,
    "FBSS-MUSIC + MDL": lambda Y, p: music(Y, p, "fbss"),
    "Toeplitz-MUSIC + MDL": lambda Y, p: music(Y, p, "toeplitz"),
    "SOMP + MDL": somp,
    "SBL + MDL": sbl,
}
