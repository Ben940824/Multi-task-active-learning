#!/usr/bin/env python3
"""
Preprocess Mushroom raw CSV into paper-aligned features and targets.

Usage (from repo root):
    .venv/bin/python scripts/preprocess_mushroom.py

Outputs:
    data/mushroom/features.parquet / features.csv
    data/mushroom/targets.parquet   / targets.csv
    data/mushroom/combined.parquet  / combined.csv
    data/mushroom/preprocessing_config.yaml
    data/mushroom/preprocessing_report.json
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a script without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from multitask_al.preprocess.mushroom import (  # noqa: E402
    preprocess_mushroom,
    print_validation_summary,
    save_preprocessed,
)

DEFAULT_RAW = (
    REPO_ROOT
    / "raw_data"
    / "drive-download-20260618T195026Z-3-001"
    / "mushroom.csv"
)
DEFAULT_OUTPUT = REPO_ROOT / "data" / "mushroom"


def main() -> None:
    raw_path = DEFAULT_RAW
    output_dir = DEFAULT_OUTPUT

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw Mushroom CSV not found at {raw_path}. "
            "Expected raw_data/drive-download-20260618T195026Z-3-001/mushroom.csv"
        )

    print(f"Reading raw data: {raw_path}")
    result = preprocess_mushroom(raw_path)

    out = save_preprocessed(result, output_dir)
    print(f"\nWrote output to: {out}")
    print_validation_summary(result)


if __name__ == "__main__":
    main()
