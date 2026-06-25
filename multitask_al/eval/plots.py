"""Plot active learning metrics vs query steps."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Human-readable labels for the three IMDB targets.
TARGET_PLOT_CONFIG = [
    ("target_imdb_score_f1_macro", "IMDb Score"),
    ("target_content_rating_f1_macro", "Content Rating"),
    ("target_gross_f1_macro", "Gross"),
]

ACCURACY_COLUMNS = [
    ("target_imdb_score_accuracy", "IMDb Score"),
    ("target_content_rating_accuracy", "Content Rating"),
    ("target_gross_accuracy", "Gross"),
]


def plot_f1_curves(metrics: pd.DataFrame, output_dir: Path | str) -> list[Path]:
    """
    Save F1 macro vs query step plots.

    Writes:
      f1_by_target.png  — one subplot per target (paper-style)
      f1_combined.png   — all three targets on one axes
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = metrics["step"]

    written: list[Path] = []

    # Three-panel figure (one target per panel).
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, (col, title) in zip(axes, TARGET_PLOT_CONFIG):
        ax.plot(steps, metrics[col], marker="o", markersize=3, linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Query Step")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Macro F1")
    fig.suptitle("IMDB Baseline Active Learning — Macro F1", y=1.02)
    fig.tight_layout()
    path_panels = output_dir / "f1_by_target.png"
    fig.savefig(path_panels, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(path_panels)

    # Single axes with all targets overlaid.
    fig, ax = plt.subplots(figsize=(8, 5))
    for col, title in TARGET_PLOT_CONFIG:
        ax.plot(steps, metrics[col], marker="o", markersize=3, linewidth=1.5, label=title)
    ax.set_xlabel("Query Step")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("IMDB Baseline Active Learning — Macro F1")
    fig.tight_layout()
    path_combined = output_dir / "f1_combined.png"
    fig.savefig(path_combined, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(path_combined)

    return written


def plot_accuracy_curves(metrics: pd.DataFrame, output_dir: Path | str) -> list[Path]:
    """Save accuracy vs query step (same layout as F1 plots)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = metrics["step"]
    written: list[Path] = []

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, (col, title) in zip(axes, ACCURACY_COLUMNS):
        ax.plot(steps, metrics[col], marker="o", markersize=3, linewidth=1.5, color="C2")
        ax.set_title(title)
        ax.set_xlabel("Query Step")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Accuracy")
    fig.suptitle("IMDB Baseline Active Learning — Accuracy", y=1.02)
    fig.tight_layout()
    path = output_dir / "accuracy_by_target.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    return written


def plot_all_metrics(metrics: pd.DataFrame, output_dir: Path | str) -> list[Path]:
    """Generate all standard AL curve plots."""
    paths = plot_f1_curves(metrics, output_dir)
    paths.extend(plot_accuracy_curves(metrics, output_dir))
    return paths
