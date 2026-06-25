"""
Random holdout split for active learning (paper baseline).

Paper (report_multitaskAL.pdf, Baseline Implementation):
  - 100 samples for initial training
  - 1000 samples for fixed test
  - remaining samples form the unlabeled pool

No stratification: paper does not mention it. Pure random split with a fixed seed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Paper-fixed sizes (same across all datasets in the report).
DEFAULT_N_TRAIN = 100
DEFAULT_N_TEST = 1000
DEFAULT_RANDOM_STATE = 42


@dataclass
class DatasetSplit:
    """Index-based partition of a preprocessed dataset."""

    train_indices: list[int]
    test_indices: list[int]
    pool_indices: list[int]
    n_total: int
    n_train: int
    n_test: int
    random_state: int
    stratified: bool = False

    def validate(self) -> None:
        """Ensure indices are disjoint, cover all rows, and match expected sizes."""
        train = set(self.train_indices)
        test = set(self.test_indices)
        pool = set(self.pool_indices)

        if train & test or train & pool or test & pool:
            raise ValueError("train, test, and pool indices must be disjoint")

        union = train | test | pool
        if len(union) != self.n_total:
            raise ValueError(
                f"indices cover {len(union)} rows, expected n_total={self.n_total}"
            )
        if union != set(range(self.n_total)):
            raise ValueError("indices must be a permutation of range(n_total)")

        if len(train) != self.n_train:
            raise ValueError(f"expected {self.n_train} train rows, got {len(train)}")
        if len(test) != self.n_test:
            raise ValueError(f"expected {self.n_test} test rows, got {len(test)}")


def random_holdout_split(
    n_samples: int,
    n_train: int = DEFAULT_N_TRAIN,
    n_test: int = DEFAULT_N_TEST,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> DatasetSplit:
    """
    Randomly partition row indices into train, test, and pool.

    One shuffle of [0, n_samples) with ``random_state``, then:
      test  <- first n_test indices
      train <- next n_train indices
      pool  <- remainder
    """
    required = n_train + n_test
    if n_samples < required:
        raise ValueError(
            f"need at least {required} samples for {n_train} train + {n_test} test, "
            f"got {n_samples}"
        )

    rng = np.random.default_rng(random_state)
    perm = rng.permutation(n_samples)

    test_idx = perm[:n_test].tolist()
    train_idx = perm[n_test : n_test + n_train].tolist()
    pool_idx = perm[n_test + n_train :].tolist()

    split = DatasetSplit(
        train_indices=[int(i) for i in train_idx],
        test_indices=[int(i) for i in test_idx],
        pool_indices=[int(i) for i in pool_idx],
        n_total=n_samples,
        n_train=n_train,
        n_test=n_test,
        random_state=random_state,
        stratified=False,
    )
    split.validate()
    return split


def target_distribution(
    targets: pd.DataFrame,
    indices: list[int],
    target_columns: list[str],
) -> dict[str, dict[str, int]]:
    """Class counts per target within a subset of rows."""
    subset = targets.iloc[indices]
    return {
        col: subset[col].value_counts().sort_index().astype(int).to_dict()
        for col in target_columns
    }


def build_split_report(
    split: DatasetSplit,
    targets: pd.DataFrame,
    target_columns: list[str],
) -> dict[str, Any]:
    """Summary for manual review (subset sizes and label distributions)."""
    return {
        "random_state": split.random_state,
        "stratified": split.stratified,
        "n_total": split.n_total,
        "sizes": {
            "train": len(split.train_indices),
            "test": len(split.test_indices),
            "pool": len(split.pool_indices),
        },
        "target_distribution": {
            "train": target_distribution(targets, split.train_indices, target_columns),
            "test": target_distribution(targets, split.test_indices, target_columns),
            "pool": target_distribution(targets, split.pool_indices, target_columns),
        },
    }


def save_split(
    split: DatasetSplit,
    output_dir: Path | str,
    targets: pd.DataFrame | None = None,
    target_columns: list[str] | None = None,
) -> Path:
    """Persist split indices and an optional review report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = asdict(split)
    split_path = output_dir / "split.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if targets is not None and target_columns is not None:
        report = build_split_report(split, targets, target_columns)
        report_path = output_dir / "split_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return output_dir


def load_split(path: Path | str) -> DatasetSplit:
    """Load a split from split.json."""
    path = Path(path)
    split_file = path if path.name == "split.json" else path / "split.json"
    with open(split_file, encoding="utf-8") as f:
        data = json.load(f)
    split = DatasetSplit(**data)
    split.validate()
    return split
