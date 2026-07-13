"""Aligned fixture generation and loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

from ._frozen_core import base_selector
from ._frozen_core import signal_processing


@dataclass(frozen=True)
class FixtureSample:
    sample_id: str
    scenario: str
    snapshots: np.ndarray
    true_doas_deg: np.ndarray


def generate_aligned_mc100_fixture(
    output_path: str | Path,
    trials_per_scenario: int = 100,
    seed_base: int = 20261030,
) -> Path:
    """Generate the exact signal fixture used by the frozen FROST-DOA results.

    Each scenario uses an independent RNG stream with seed
    ``seed_base + 1009 * scenario_index``.  This detail is essential: the older
    Phase-C fixture in the development archive used different samples and must
    not be paired with the frozen FROST-DOA scores.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scenarios = base_selector.test_scenarios_v9()
    with h5py.File(output_path, "w") as h5:
        h5.attrs["seed_base"] = int(seed_base)
        h5.attrs["trials_per_scenario"] = int(trials_per_scenario)
        h5.attrs["sample_id_format"] = "{scenario}#{index:04d}"
        root = h5.create_group("scenarios")
        for idx, scenario in enumerate(scenarios):
            rng = np.random.default_rng(seed_base + 1009 * idx)
            ys, thetas = [], []
            for _ in range(trials_per_scenario):
                y, theta, _ = signal_processing.simulate(scenario, rng)
                ys.append(y)
                thetas.append(theta)
            y_stack = np.stack(ys)
            padded = np.full((trials_per_scenario, 4), np.nan, dtype=np.float64)
            mask = np.zeros((trials_per_scenario, 4), dtype=np.int8)
            for i, theta in enumerate(thetas):
                padded[i, : len(theta)] = theta
                mask[i, : len(theta)] = 1
            g = root.create_group(scenario.name)
            g.create_dataset("Y_real", data=y_stack.real.astype(np.float64), compression="gzip")
            g.create_dataset("Y_imag", data=y_stack.imag.astype(np.float64), compression="gzip")
            g.create_dataset("theta_deg_padded", data=padded)
            g.create_dataset("theta_mask", data=mask)
            g.attrs["num_sensors"] = int(scenario.M)
            g.attrs["num_sources"] = int(scenario.K)
            g.attrs["num_snapshots"] = int(scenario.T)
            g.attrs["snr_db"] = float(scenario.snr_db)
            g.attrs["correlation"] = float(scenario.coherent_rho)
            g.attrs["noise"] = str(scenario.noise)
            g.attrs["mismatch"] = bool(scenario.mismatch)
            g.attrs["minimum_separation_deg"] = float(scenario.min_sep)
    return output_path


def iter_fixture(path: str | Path) -> Iterator[FixtureSample]:
    """Iterate through an aligned fixture in deterministic scenario order."""
    with h5py.File(path, "r") as h5:
        for scenario in sorted(h5["scenarios"].keys()):
            g = h5["scenarios"][scenario]
            for i in range(g["Y_real"].shape[0]):
                y = np.asarray(g["Y_real"][i]) + 1j * np.asarray(g["Y_imag"][i])
                mask = np.asarray(g["theta_mask"][i]).astype(bool)
                theta = np.asarray(g["theta_deg_padded"][i])[mask].astype(float)
                yield FixtureSample(
                    sample_id=f"{scenario}#{i:04d}",
                    scenario=scenario,
                    snapshots=y,
                    true_doas_deg=theta,
                )
