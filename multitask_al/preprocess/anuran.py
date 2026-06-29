"""Anuran Calls (Frogs MFCCs) preprocessing (report_multitaskAL.pdf Table 1)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from multitask_al.preprocess.common import (
    build_standard_report,
    compute_imbalance_ratios,
    mark_missing_values,
    save_preprocessed_artifacts,
)

PAPER_ROW_COUNT = 7195
PAPER_TOTAL_COLUMN_COUNT = 25
PAPER_FEATURE_COUNT = 22

IDENTIFIER_COLUMNS = ["RecordID"]
RAW_TARGET_COLUMNS = ["Family", "Genus", "Species"]
TARGET_COLUMNS = ["target_family", "target_genus", "target_species"]
RAW_TO_TARGET = {
    "Family": "target_family",
    "Genus": "target_genus",
    "Species": "target_species",
}

MFCC_COLUMNS = [f"MFCCs_{i}" for i in range(1, 23)]  # MFCCs_ 1 .. MFCCs_22 in CSV
# Actual column names have a space: "MFCCs_ 1"
def _mfcc_column_names(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("MFCCs_")]


PAPER_IR: dict[str, tuple[float, float]] = {
    "target_family": (25.066, 65.0),
    "target_genus": (23.499, 61.029),
    "target_species": (16.813, 51.147),
}


@dataclass
class PreprocessConfig:
    raw_path: str
    identifier_columns: list[str] = field(default_factory=lambda: list(IDENTIFIER_COLUMNS))
    target_columns: list[str] = field(default_factory=lambda: list(TARGET_COLUMNS))
    notes: list[str] = field(
        default_factory=lambda: ["Drop RecordID (unique identifier per paper)."]
    )


@dataclass
class PreprocessResult:
    features: pd.DataFrame
    targets: pd.DataFrame
    config: PreprocessConfig
    row_counts: dict[str, int]
    imbalance_ratios: dict[str, dict[str, float]]
    feature_column_names: list[str]


def preprocess_anuran(raw_path: Path | str) -> PreprocessResult:
    raw_path = Path(raw_path)
    config = PreprocessConfig(raw_path=str(raw_path.resolve()))

    df = pd.read_csv(raw_path)
    counts: dict[str, int] = {"raw": len(df)}
    df = mark_missing_values(df)
    df = df.drop(columns=IDENTIFIER_COLUMNS, errors="ignore")
    df = df.drop_duplicates()
    counts["after_dedup"] = len(df)
    df = df.dropna()
    counts["final"] = len(df)

    if counts["final"] != PAPER_ROW_COUNT:
        raise ValueError(f"Row count {counts['final']} != paper {PAPER_ROW_COUNT}")

    mfcc_cols = _mfcc_column_names(df)
    if len(mfcc_cols) != PAPER_FEATURE_COUNT:
        raise ValueError(f"Expected {PAPER_FEATURE_COUNT} MFCC columns, got {len(mfcc_cols)}")

    targets = pd.DataFrame(index=df.index)
    for raw_col, target_col in RAW_TO_TARGET.items():
        targets[target_col] = df[raw_col].astype(str)

    features = df[mfcc_cols].astype(float)

    imbalance_ratios = {
        col: compute_imbalance_ratios(targets[col]) for col in TARGET_COLUMNS
    }

    return PreprocessResult(
        features=features,
        targets=targets,
        config=config,
        row_counts=counts,
        imbalance_ratios=imbalance_ratios,
        feature_column_names=list(features.columns),
    )


def save_preprocessed(result: PreprocessResult, output_dir: Path | str) -> Path:
    report = build_standard_report(
        result.row_counts,
        PAPER_ROW_COUNT,
        len(result.feature_column_names),
        PAPER_FEATURE_COUNT,
        PAPER_TOTAL_COLUMN_COUNT,
        TARGET_COLUMNS,
        result.targets,
        result.imbalance_ratios,
        PAPER_IR,
    )
    config_payload: dict[str, Any] = {
        **asdict(result.config),
        "feature_column_names": result.feature_column_names,
        "raw_to_target": RAW_TO_TARGET,
        "paper_targets": {
            "row_count": PAPER_ROW_COUNT,
            "total_column_count": PAPER_TOTAL_COLUMN_COUNT,
            "feature_count": PAPER_FEATURE_COUNT,
        },
    }
    return save_preprocessed_artifacts(
        result.features, result.targets, output_dir, config_payload, report
    )


def print_validation_summary(result: PreprocessResult) -> None:
    print("=" * 60)
    print("Anuran Calls preprocessing validation vs paper Table 1")
    print("=" * 60)
    print(f"Rows (paper/ours): {PAPER_ROW_COUNT} / {result.row_counts['final']}")
    print(f"Features (paper/ours): {PAPER_FEATURE_COUNT} / {len(result.feature_column_names)}")
    for target in TARGET_COLUMNS:
        ours = result.imbalance_ratios[target]
        paper = PAPER_IR[target]
        print(f"  {target}: IR1 {ours['IR1']} vs {paper[0]}, IR2 {ours['IR2']} vs {paper[1]}")
