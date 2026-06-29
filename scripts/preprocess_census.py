#!/usr/bin/env python3
"""
Preprocess Census (UCI Adult) raw CSV into paper-aligned features and targets.

Usage (from repo root):
    .venv/bin/python scripts/preprocess_census.py

Outputs:
    data/census/features.parquet / features.csv
    data/census/targets.parquet   / targets.csv
    data/census/combined.parquet  / combined.csv
    data/census/preprocessing_config.yaml
    data/census/preprocessing_report.json
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from multitask_al.preprocess.census import (  # noqa: E402
    preprocess_census,
    print_validation_summary,
    save_preprocessed,
)

DEFAULT_RAW = (
    REPO_ROOT
    / "raw_data"
    / "drive-download-20260618T195026Z-3-001"
    / "census.csv"
)
DEFAULT_OUTPUT = REPO_ROOT / "data" / "census"


def main() -> None:
    raw_path = DEFAULT_RAW
    output_dir = DEFAULT_OUTPUT

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw Census CSV not found at {raw_path}. "
            "Expected raw_data/drive-download-20260618T195026Z-3-001/census.csv"
        )

    print(f"Reading raw data: {raw_path}")
    result = preprocess_census(raw_path)

    out = save_preprocessed(result, output_dir)
    print(f"\nWrote output to: {out}")
    print_validation_summary(result)


if __name__ == "__main__":
    main()
