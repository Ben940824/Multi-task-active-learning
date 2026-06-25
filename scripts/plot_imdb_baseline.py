#!/usr/bin/env python3
"""
Plot IMDB baseline metrics from an existing run (no retraining).

Usage:
    .venv/bin/python scripts/plot_imdb_baseline.py
    .venv/bin/python scripts/plot_imdb_baseline.py --metrics outputs/imdb_baseline/metrics.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from multitask_al.eval.plots import plot_all_metrics  # noqa: E402

DEFAULT_METRICS = REPO_ROOT / "outputs" / "imdb_baseline" / "metrics.csv"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "imdb_baseline"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot AL curves from metrics.csv")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    import pandas as pd

    metrics = pd.read_csv(args.metrics)
    paths = plot_all_metrics(metrics, args.output_dir)
    for p in paths:
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
