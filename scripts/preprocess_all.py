#!/usr/bin/env python3
"""
Preprocess all 10 paper datasets into data/<name>/.

Usage (from repo root):
    .venv/bin/python scripts/preprocess_all.py
    .venv/bin/python scripts/preprocess_all.py --only bank telco flight

Each dataset writes:
    data/<name>/features.csv, targets.csv, combined.csv (+ parquet)
    data/<name>/preprocessing_config.yaml
    data/<name>/preprocessing_report.json

Also writes data/preprocessing_summary.json (master IR / row-count table).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

RAW_DIR = REPO_ROOT / "raw_data" / "drive-download-20260618T195026Z-3-001"
DATA_DIR = REPO_ROOT / "data"

# (output_name, raw_filename, preprocess_fn module pieces)
from multitask_al.preprocess import (  # noqa: E402
    anuran,
    bank,
    census,
    flight,
    imdb,
    mushroom,
    paris,
    shopper,
    smoking,
    telco,
)

DATASETS: dict[str, dict] = {
    "imdb": {
        "raw": RAW_DIR / "movie_metadata.csv",
        "preprocess": imdb.preprocess_imdb,
        "save": imdb.save_preprocessed,
        "summary": imdb.print_validation_summary,
    },
    "mushroom": {
        "raw": RAW_DIR / "mushroom.csv",
        "preprocess": mushroom.preprocess_mushroom,
        "save": mushroom.save_preprocessed,
        "summary": mushroom.print_validation_summary,
    },
    "census": {
        "raw": RAW_DIR / "census.csv",
        "preprocess": census.preprocess_census,
        "save": census.save_preprocessed,
        "summary": census.print_validation_summary,
    },
    "bank": {
        "raw": RAW_DIR / "bank-additional.csv",
        "preprocess": bank.preprocess_bank,
        "save": bank.save_preprocessed,
        "summary": bank.print_validation_summary,
    },
    "shopper": {
        "raw": RAW_DIR / "online_shoppers_intention.csv",
        "preprocess": shopper.preprocess_shopper,
        "save": shopper.save_preprocessed,
        "summary": shopper.print_validation_summary,
    },
    "anuran": {
        "raw": RAW_DIR / "Frogs_MFCCs.csv",
        "preprocess": anuran.preprocess_anuran,
        "save": anuran.save_preprocessed,
        "summary": anuran.print_validation_summary,
    },
    "telco": {
        "raw": RAW_DIR / "telco.csv",
        "preprocess": telco.preprocess_telco,
        "save": telco.save_preprocessed,
        "summary": telco.print_validation_summary,
    },
    "paris": {
        "raw": RAW_DIR / "ParisHousingClass.csv",
        "preprocess": paris.preprocess_paris,
        "save": paris.save_preprocessed,
        "summary": paris.print_validation_summary,
    },
    "smoking": {
        "raw": RAW_DIR / "smoking.csv",
        "preprocess": smoking.preprocess_smoking,
        "save": smoking.save_preprocessed,
        "summary": smoking.print_validation_summary,
    },
    "flight": {
        "raw": RAW_DIR / "flight.csv",
        "preprocess": flight.preprocess_flight,
        "save": flight.save_preprocessed,
        "summary": flight.print_validation_summary,
    },
}


def _ir_match(ours: dict, paper: dict) -> bool:
    for key in ours:
        if key not in paper:
            return False
        if ours[key]["IR1"] != paper[key]["IR1"]:
            return False
        if ours[key]["IR2"] != paper[key]["IR2"]:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess all paper datasets")
    parser.add_argument(
        "--only",
        nargs="*",
        choices=sorted(DATASETS.keys()),
        help="Subset of datasets to process (default: all)",
    )
    args = parser.parse_args()

    names = args.only or sorted(DATASETS.keys())
    master_summary: dict[str, dict] = {}

    for name in names:
        spec = DATASETS[name]
        raw_path = spec["raw"]
        out_dir = DATA_DIR / name

        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw file for {name}: {raw_path}")

        print(f"\n{'#' * 60}\nProcessing {name}\n{'#' * 60}")
        print(f"Reading: {raw_path}")
        result = spec["preprocess"](raw_path)
        spec["save"](result, out_dir)
        spec["summary"](result)
        print(f"Wrote: {out_dir}")

        report_path = out_dir / "preprocessing_report.json"
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)

        master_summary[name] = {
            "rows_ours": report["row_counts"].get(
                "final",
                report["row_counts"].get(
                    "after_income_normalize",
                    report["row_counts"].get("after_dropna"),
                ),
            ),
            "rows_paper": report["paper_row_count"],
            "row_delta": report["row_count_delta_vs_paper"],
            "features_ours": report["feature_count"],
            "features_paper": report["paper_feature_count"],
            "ir_match": _ir_match(
                report["imbalance_ratios"], report["paper_imbalance_ratios"]
            ),
            "imbalance_ratios": report["imbalance_ratios"],
        }

    summary_path = DATA_DIR / "preprocessing_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2)
    print(f"\nWrote master summary: {summary_path}")

    matched = sum(1 for v in master_summary.values() if v["ir_match"])
    print(f"\nIR exact match: {matched}/{len(master_summary)} datasets")


if __name__ == "__main__":
    main()
