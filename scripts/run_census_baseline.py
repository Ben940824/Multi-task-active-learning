#!/usr/bin/env python3
"""
Run Census single-output baseline active learning (paper Algorithm 1).

Usage (from repo root):
    .venv/bin/python scripts/split_census.py
    .venv/bin/python scripts/run_census_baseline.py

Requires:
    data/census/features.parquet (or .csv)
    data/census/targets.parquet  (or .csv)
    data/census/split.json

Outputs:
    outputs/census_baseline/metrics.csv
    outputs/census_baseline/run_config.json
    outputs/census_baseline/f1_by_target.png
    outputs/census_baseline/f1_combined.png
    outputs/census_baseline/accuracy_by_target.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from multitask_al.active_learning.loop import (  # noqa: E402
    save_run,
    run_baseline_active_learning,
)
from multitask_al.data.split import load_split  # noqa: E402
from multitask_al.preprocess.census import TARGET_COLUMNS  # noqa: E402

DEFAULT_DATA_DIR = REPO_ROOT / "data" / "census"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "census_baseline"

CENSUS_TARGET_LABELS = {
    "target_income": "Income",
    "target_marital_status": "Marital Status",
    "target_workclass": "Work Class",
}

PAPER_BASELINE_F1 = {
    "target_income": 0.711,
    "target_marital_status": 0.358,
    "target_workclass": 0.164,
}


def _load_frame(data_dir: Path, name: str):
    import pandas as pd

    parquet = data_dir / f"{name}.parquet"
    csv = data_dir / f"{name}.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing {name}.parquet/.csv in {data_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Census single-output baseline active learning"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42, help="RF random_state")
    args = parser.parse_args()

    features = _load_frame(args.data_dir, "features")
    targets = _load_frame(args.data_dir, "targets")
    split = load_split(args.data_dir)

    if len(features) != len(targets):
        raise ValueError("features and targets row counts differ")

    print(f"Loaded {len(features)} rows, split seed={split.random_state}")
    print(f"  train={len(split.train_indices)} test={len(split.test_indices)} "
          f"pool={len(split.pool_indices)}")
    print(f"Running baseline AL: {args.n_steps} query steps, no resampling")

    run = run_baseline_active_learning(
        features=features,
        targets=targets,
        split=split,
        target_columns=TARGET_COLUMNS,
        n_steps=args.n_steps,
        rf_random_state=args.seed,
    )

    out = save_run(
        run,
        args.output_dir,
        plot_title="Census Baseline Active Learning",
        target_labels=CENSUS_TARGET_LABELS,
    )
    print(f"\nWrote {out / 'metrics.csv'}")
    print(f"Wrote {out / 'run_config.json'}")
    print(f"Wrote {out / 'f1_by_target.png'}")
    print(f"Wrote {out / 'f1_combined.png'}")
    print(f"Wrote {out / 'accuracy_by_target.png'}")

    s0 = run.steps[0]
    sf = run.steps[-1]
    print("\nMacro-F1 (step 0 -> step {}):".format(sf.step))
    for target in TARGET_COLUMNS:
        s0_f1 = s0.per_target[target]["f1_macro"]
        sf_f1 = sf.per_target[target]["f1_macro"]
        paper_f1 = PAPER_BASELINE_F1[target]
        label = CENSUS_TARGET_LABELS[target]
        print(f"  {label:22s}  {s0_f1:.3f} -> {sf_f1:.3f}  (paper baseline ~{paper_f1})")


if __name__ == "__main__":
    main()
