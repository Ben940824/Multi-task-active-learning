"""Bank Marketing preprocessing (report_multitaskAL.pdf Table 1)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from multitask_al.preprocess.common import (
    build_mixed_features,
    build_standard_report,
    compute_imbalance_ratios,
    drop_rows_with_unknown_categories,
    save_preprocessed_artifacts,
    strip_object_columns,
)

PAPER_ROW_COUNT = 3090
PAPER_TOTAL_COLUMN_COUNT = 21
PAPER_FEATURE_COUNT = 18

RAW_TARGET_COLUMNS = ["y", "loan", "housing"]
TARGET_COLUMNS = ["target_campaign", "target_personal_loan", "target_housing_loan"]
RAW_TO_TARGET = {
    "y": "target_campaign",
    "loan": "target_personal_loan",
    "housing": "target_housing_loan",
}

NUMERIC_FEATURE_COLUMNS = [
    "age",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]

CATEGORICAL_FEATURE_COLUMNS = [
    "job",
    "marital",
    "education",
    "default",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]

FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS

PAPER_IR: dict[str, tuple[float, float]] = {
    "target_campaign": (7.351, 7.351),
    "target_personal_loan": (5.095, 5.095),
    "target_housing_loan": (1.204, 1.204),
}


@dataclass
class PreprocessConfig:
    raw_path: str
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    target_columns: list[str] = field(default_factory=lambda: list(TARGET_COLUMNS))
    label_encodings: dict[str, dict[str, int]] = field(default_factory=dict)
    notes: list[str] = field(
        default_factory=lambda: [
            "Drop rows where any categorical field is 'unknown' (UCI convention).",
            "Paper Target = y (term deposit subscription).",
        ]
    )


@dataclass
class PreprocessResult:
    features: pd.DataFrame
    targets: pd.DataFrame
    config: PreprocessConfig
    row_counts: dict[str, int]
    imbalance_ratios: dict[str, dict[str, float]]
    feature_column_names: list[str]


def preprocess_bank(raw_path: Path | str) -> PreprocessResult:
    raw_path = Path(raw_path)
    config = PreprocessConfig(raw_path=str(raw_path.resolve()))

    df = pd.read_csv(raw_path)
    counts: dict[str, int] = {"raw": len(df)}
    df = drop_rows_with_unknown_categories(df)
    counts["after_drop_unknown"] = len(df)
    df = df.drop_duplicates()
    counts["after_dedup"] = len(df)
    df = df.dropna()
    counts["final"] = len(df)

    if counts["final"] != PAPER_ROW_COUNT:
        raise ValueError(f"Row count {counts['final']} != paper {PAPER_ROW_COUNT}")

    targets = pd.DataFrame(index=df.index)
    for raw_col, target_col in RAW_TO_TARGET.items():
        targets[target_col] = strip_object_columns(df)[raw_col].astype(str)

    features, encodings = build_mixed_features(
        df, NUMERIC_FEATURE_COLUMNS, CATEGORICAL_FEATURE_COLUMNS
    )
    config.label_encodings = encodings

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
    print("Bank Marketing preprocessing validation vs paper Table 1")
    print("=" * 60)
    print(f"Rows (paper/ours): {PAPER_ROW_COUNT} / {result.row_counts['final']}")
    print(f"Features (paper/ours): {PAPER_FEATURE_COUNT} / {len(result.feature_column_names)}")
    for target in TARGET_COLUMNS:
        ours = result.imbalance_ratios[target]
        paper = PAPER_IR[target]
        print(f"  {target}: IR1 {ours['IR1']} vs {paper[0]}, IR2 {ours['IR2']} vs {paper[1]}")
