#!/usr/bin/env python3
"""
Create a random train / test / pool split for Census active learning.

Usage (from repo root):
    .venv/bin/python scripts/split_census.py
    .venv/bin/python scripts/split_census.py --seed 0

Reads preprocessed targets from data/census/targets.parquet (or .csv).
Writes:
    data/census/split.json
    data/census/split_report.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from multitask_al.data.split import (  # noqa: E402
    DEFAULT_N_TEST,
    DEFAULT_N_TRAIN,
    DEFAULT_RANDOM_STATE,
    random_holdout_split,
    save_split,
)
from multitask_al.preprocess.census import TARGET_COLUMNS  # noqa: E402

DEFAULT_DATA_DIR = REPO_ROOT / "data" / "census"


def _load_targets(data_dir: Path):
    import pandas as pd

    parquet = data_dir / "targets.parquet"
    csv = data_dir / "targets.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(
        f"No targets file in {data_dir}. Run scripts/preprocess_census.py first."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Random Census train/test/pool split")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory with preprocessed Census files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=DEFAULT_N_TRAIN,
        help="Initial labeled training size (default: 100)",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=DEFAULT_N_TEST,
        help="Fixed test size (default: 1000)",
    )
    args = parser.parse_args()

    targets = _load_targets(args.data_dir)
    n_samples = len(targets)

    split = random_holdout_split(
        n_samples=n_samples,
        n_train=args.n_train,
        n_test=args.n_test,
        random_state=args.seed,
    )
    save_split(split, args.data_dir, targets=targets, target_columns=TARGET_COLUMNS)

    print(f"Split {n_samples} rows (seed={args.seed}, stratified=False)")
    print(f"  train: {len(split.train_indices)}")
    print(f"  test:  {len(split.test_indices)}")
    print(f"  pool:  {len(split.pool_indices)}")
    print(f"Wrote {args.data_dir / 'split.json'}")
    print(f"Wrote {args.data_dir / 'split_report.json'}")


if __name__ == "__main__":
    main()
