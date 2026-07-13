#!/usr/bin/env python3
"""Create publication-ready figures with Times New Roman typography.

The figures are written as 600-dpi PNG plus vector PDF/SVG. Colors are chosen
for print readability and color-vision accessibility; hatches and markers keep
methods distinguishable when printed in grayscale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_COLORS = {
    "FROST-DOA": "#005BBB",
    "DA-MUSIC + MDL": "#7B2CBF",
    "SubspaceNet + residual selector": "#F28E2B",
    "MUSIC + MDL": "#9AA3AD",
    "Root-MUSIC + MDL": "#6C757D",
    "ESPRIT + MDL": "#7D8597",
    "FBSS-MUSIC + MDL": "#ADB5BD",
    "Toeplitz-MUSIC + MDL": "#5C677D",
    "SOMP + MDL": "#8D99AE",
    "SBL + MDL": "#B0A8B9",
}
METHOD_HATCHES = {
    "FROST-DOA": "",
    "DA-MUSIC + MDL": "//",
    "SubspaceNet + residual selector": "\\\\",
}
SCENARIO_LABELS = {
    "U1_clean_K1": "Clean, P=1",
    "U2_clean_K2": "Clean, P=2",
    "U3_clean_K3": "Clean, P=3",
    "U4_lowSNR_few_K2": "Low SNR/few, P=2",
    "U5_lowSNR_few_K3": "Low SNR/few, P=3",
    "U6_coherent_K2": "Coherent, P=2",
    "U7_coherent_K3": "Coherent, P=3",
    "U8_mismatch_K2": "Array mismatch, P=2",
    "U9_heavytail_K3": "Heavy-tailed, P=3",
    "U10_hard_mixed_K2": "Hard mixed, P=2",
    "U11_colored_lowSNR_K2": "Colored low SNR, P=2",
    "U12_close_K2": "Closely spaced, P=2",
}


def configure_style() -> str:
    available = {font.name for font in fm.fontManager.ttflist}
    family = "Times New Roman" if "Times New Roman" in available else "Times"
    if family not in available:
        family = "Liberation Serif"
    mpl.rcParams.update(
        {
            "font.family": family,
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "grid.color": "#D9DEE5",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return family


def color(method: str) -> str:
    return METHOD_COLORS.get(method, "#8A96A3")


def save_all(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in {
        "png": {"dpi": 600},
        "pdf": {},
        "svg": {},
    }.items():
        fig.savefig(output_dir / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def overall_rmse(overall: pd.DataFrame, output_dir: Path) -> None:
    data = overall.sort_values("mean_rmse_deg", ascending=True)
    fig, ax = plt.subplots(figsize=(7.15, 4.4))
    bars = ax.barh(
        data.method,
        data.mean_rmse_deg,
        color=[color(m) for m in data.method],
        edgecolor="#30343B",
        linewidth=0.55,
    )
    for bar, method in zip(bars, data.method):
        bar.set_hatch(METHOD_HATCHES.get(method, ""))
    for bar, value in zip(bars, data.mean_rmse_deg):
        ax.text(
            value + 0.25,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            fontsize=8.5,
        )
    ax.set_xlabel("Mean penalized set RMSE (degrees)")
    ax.set_ylabel("")
    ax.grid(axis="x")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, max(data.mean_rmse_deg) * 1.18)
    fig.tight_layout()
    save_all(fig, output_dir, "fig_overall_rmse")


def source_accuracy(overall: pd.DataFrame, output_dir: Path) -> None:
    data = overall.sort_values("source_number_accuracy_pct", ascending=True)
    fig, ax = plt.subplots(figsize=(7.15, 4.4))
    bars = ax.barh(
        data.method,
        data.source_number_accuracy_pct,
        color=[color(m) for m in data.method],
        edgecolor="#30343B",
        linewidth=0.55,
    )
    for bar, method in zip(bars, data.method):
        bar.set_hatch(METHOD_HATCHES.get(method, ""))
    for bar, value in zip(bars, data.source_number_accuracy_pct):
        ax.text(
            value + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            fontsize=8.5,
        )
    ax.set_xlabel("Source-number accuracy (%)")
    ax.set_xlim(0, 105)
    ax.grid(axis="x")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_all(fig, output_dir, "fig_source_number_accuracy")


def scenario_comparison(scenario: pd.DataFrame, output_dir: Path) -> None:
    requested = [
        "FROST-DOA",
        "DA-MUSIC + MDL",
        "SubspaceNet + residual selector",
        "Root-MUSIC + MDL",
    ]
    methods = [m for m in requested if m in set(scenario.method)]
    ordered_scenarios = list(SCENARIO_LABELS)
    pivot = scenario.pivot(index="scenario", columns="method", values="mean_rmse_deg").reindex(
        ordered_scenarios
    )
    x = np.arange(len(pivot))
    width = 0.78 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=(10.2, 4.5))
    for j, method in enumerate(methods):
        offset = (j - (len(methods) - 1) / 2) * width
        ax.bar(
            x + offset,
            pivot[method],
            width=width,
            label=method,
            color=color(method),
            edgecolor="#30343B",
            linewidth=0.45,
            hatch=METHOD_HATCHES.get(method, ""),
        )
    ax.set_xticks(x, [SCENARIO_LABELS[s] for s in ordered_scenarios], rotation=28, ha="right")
    ax.set_ylabel("Mean penalized set RMSE (degrees)")
    ax.grid(axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=2, frameon=False, loc="upper left")
    fig.tight_layout()
    save_all(fig, output_dir, "fig_scenario_rmse")


def paired_win_rates(paired: pd.DataFrame, output_dir: Path) -> None:
    data = paired.sort_values("comparison_win_rate_pct", ascending=True)
    fig, ax = plt.subplots(figsize=(7.15, 4.2))
    y = np.arange(len(data))
    ax.barh(y, data.frost_win_rate_pct, color="#005BBB", label="FROST-DOA wins")
    ax.barh(
        y,
        data.comparison_win_rate_pct,
        left=data.frost_win_rate_pct,
        color="#D8DEE7",
        label="Comparison method wins",
    )
    ax.barh(
        y,
        data.tie_rate_pct,
        left=data.frost_win_rate_pct + data.comparison_win_rate_pct,
        color="#FFFFFF",
        edgecolor="#30343B",
        linewidth=0.45,
        label="Ties",
    )
    ax.set_yticks(y, data.comparison_method)
    ax.set_xlabel("Paired sample fraction (%)")
    ax.set_xlim(0, 100)
    ax.grid(axis="x")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout()
    save_all(fig, output_dir, "fig_paired_win_rates")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("figures/terminal_run"))
    args = parser.parse_args()
    family = configure_style()
    print(f"[figures] font family: {family}")
    overall = pd.read_csv(args.results_dir / "overall_summary.csv")
    scenario = pd.read_csv(args.results_dir / "scenario_summary.csv")
    paired = pd.read_csv(args.results_dir / "paired_vs_frost_doa.csv")
    overall_rmse(overall, args.output_dir)
    source_accuracy(overall, args.output_dir)
    scenario_comparison(scenario, args.output_dir)
    paired_win_rates(paired, args.output_dir)
    print(f"[figures] wrote PNG/PDF/SVG files to {args.output_dir}")


if __name__ == "__main__":
    main()
