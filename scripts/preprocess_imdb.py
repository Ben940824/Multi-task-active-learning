#!/usr/bin/env python3
"""
Preprocess IMDB raw CSV into paper-aligned features and targets.

Usage (from repo root):
    .venv/bin/python scripts/preprocess_imdb.py

Outputs:
    data/imdb/features.parquet / features.csv
    data/imdb/targets.parquet   / targets.csv
    data/imdb/combined.parquet  / combined.csv
    data/imdb/preprocessing_config.yaml
    data/imdb/preprocessing_report.json
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a script without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from multitask_al.preprocess.imdb import (  # noqa: E402
    preprocess_imdb,
    print_validation_summary,
    save_preprocessed,
)

DEFAULT_RAW = (
    REPO_ROOT
    / "raw_data"
    / "drive-download-20260618T195026Z-3-001"
    / "movie_metadata.csv"
)
DEFAULT_OUTPUT = REPO_ROOT / "data" / "imdb"


def main() -> None:
    raw_path = DEFAULT_RAW
    output_dir = DEFAULT_OUTPUT

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw IMDB CSV not found at {raw_path}. "
            "Expected raw_data/drive-download-20260618T195026Z-3-001/movie_metadata.csv"
        )

    print(f"Reading raw data: {raw_path}")
    result = preprocess_imdb(raw_path)

    out = save_preprocessed(result, output_dir)
    print(f"\nWrote output to: {out}")
    print_validation_summary(result)


if __name__ == "__main__":
    main()
