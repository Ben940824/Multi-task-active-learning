"""Plot active learning metrics vs query steps."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _f1_plot_config(
    metrics: pd.DataFrame,
    target_labels: dict[str, str] | None = None,
) -> list[tuple[str, str]]:
    """Build (column_name, display_title) pairs from metrics columns."""
    f1_cols = [c for c in metrics.columns if c.endswith("_f1_macro")]
    config: list[tuple[str, str]] = []
    for col in f1_cols:
        target = col[: -len("_f1_macro")]
        title = (target_labels or {}).get(target, target)
        config.append((col, title))
    return config


def _accuracy_plot_config(
    metrics: pd.DataFrame,
    target_labels: dict[str, str] | None = None,
) -> list[tuple[str, str]]:
    """Build (column_name, display_title) pairs for accuracy columns."""
    acc_cols = [c for c in metrics.columns if c.endswith("_accuracy")]
    config: list[tuple[str, str]] = []
    for col in acc_cols:
        target = col[: -len("_accuracy")]
        title = (target_labels or {}).get(target, target)
        config.append((col, title))
    return config


def plot_f1_curves(
    metrics: pd.DataFrame,
    output_dir: Path | str,
    title: str = "Baseline Active Learning",
    target_labels: dict[str, str] | None = None,
) -> list[Path]:
    """
    Save F1 macro vs query step plots.

    Writes:
      f1_by_target.png  — one subplot per target (paper-style)
      f1_combined.png   — all targets on one axes
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = metrics["step"]
    plot_config = _f1_plot_config(metrics, target_labels)

    written: list[Path] = []

    # One panel per target.
    n_targets = len(plot_config)
    fig, axes = plt.subplots(1, n_targets, figsize=(4 * n_targets, 4), sharey=True)
    if n_targets == 1:
        axes = [axes]
    for ax, (col, panel_title) in zip(axes, plot_config):
        ax.plot(steps, metrics[col], marker="o", markersize=3, linewidth=1.5)
        ax.set_title(panel_title)
        ax.set_xlabel("Query Step")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Macro F1")
    fig.suptitle(f"{title} — Macro F1", y=1.02)
    fig.tight_layout()
    path_panels = output_dir / "f1_by_target.png"
    fig.savefig(path_panels, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(path_panels)

    # Single axes with all targets overlaid.
    fig, ax = plt.subplots(figsize=(8, 5))
    for col, panel_title in plot_config:
        ax.plot(steps, metrics[col], marker="o", markersize=3, linewidth=1.5, label=panel_title)
    ax.set_xlabel("Query Step")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{title} — Macro F1")
    fig.tight_layout()
    path_combined = output_dir / "f1_combined.png"
    fig.savefig(path_combined, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(path_combined)

    return written


def plot_accuracy_curves(
    metrics: pd.DataFrame,
    output_dir: Path | str,
    title: str = "Baseline Active Learning",
    target_labels: dict[str, str] | None = None,
) -> list[Path]:
    """Save accuracy vs query step (same layout as F1 plots)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = metrics["step"]
    plot_config = _accuracy_plot_config(metrics, target_labels)
    written: list[Path] = []

    n_targets = len(plot_config)
    fig, axes = plt.subplots(1, n_targets, figsize=(4 * n_targets, 4), sharey=True)
    if n_targets == 1:
        axes = [axes]
    for ax, (col, panel_title) in zip(axes, plot_config):
        ax.plot(steps, metrics[col], marker="o", markersize=3, linewidth=1.5, color="C2")
        ax.set_title(panel_title)
        ax.set_xlabel("Query Step")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Accuracy")
    fig.suptitle(f"{title} — Accuracy", y=1.02)
    fig.tight_layout()
    path = output_dir / "accuracy_by_target.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    return written


def plot_all_metrics(
    metrics: pd.DataFrame,
    output_dir: Path | str,
    title: str = "Baseline Active Learning",
    target_labels: dict[str, str] | None = None,
) -> list[Path]:
    """Generate all standard AL curve plots."""
    paths = plot_f1_curves(metrics, output_dir, title=title, target_labels=target_labels)
    paths.extend(
        plot_accuracy_curves(metrics, output_dir, title=title, target_labels=target_labels)
    )
    return paths
