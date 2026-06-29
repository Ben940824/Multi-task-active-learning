"""Flight Feedback preprocessing (report_multitaskAL.pdf Table 1)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from multitask_al.preprocess.common import (
    build_mixed_features,
    build_standard_report,
    compute_imbalance_ratios,
    mark_missing_values,
    save_preprocessed_artifacts,
)

PAPER_ROW_COUNT = 129487
PAPER_TOTAL_COLUMN_COUNT = 23
PAPER_FEATURE_COUNT = 20

RAW_TARGET_COLUMNS = ["satisfaction", "Customer Type", "Class"]
TARGET_COLUMNS = [
    "target_satisfaction",
    "target_customer_type",
    "target_class",
]
RAW_TO_TARGET = {
    "satisfaction": "target_satisfaction",
    "Customer Type": "target_customer_type",
    "Class": "target_class",
}

NUMERIC_FEATURE_COLUMNS = [
    "Age",
    "Flight Distance",
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Inflight service",
    "Cleanliness",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]

CATEGORICAL_FEATURE_COLUMNS = ["Gender", "Type of Travel"]

FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS

PAPER_IR: dict[str, tuple[float, float]] = {
    "target_satisfaction": (1.302, 1.302),
    "target_customer_type": (4.46, 4.46),
    "target_class": (3.838, 6.609),
}


@dataclass
class PreprocessConfig:
    raw_path: str
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    target_columns: list[str] = field(default_factory=lambda: list(TARGET_COLUMNS))
    label_encodings: dict[str, dict[str, int]] = field(default_factory=dict)
    notes: list[str] = field(
        default_factory=lambda: [
            "Standard clean: dedup + dropna (393 rows missing Arrival Delay).",
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


def preprocess_flight(raw_path: Path | str) -> PreprocessResult:
    raw_path = Path(raw_path)
    config = PreprocessConfig(raw_path=str(raw_path.resolve()))

    df = pd.read_csv(raw_path, low_memory=False)
    counts: dict[str, int] = {"raw": len(df)}
    df = mark_missing_values(df)
    df = df.drop_duplicates()
    counts["after_dedup"] = len(df)
    df = df.dropna()
    counts["final"] = len(df)

    if counts["final"] != PAPER_ROW_COUNT:
        raise ValueError(f"Row count {counts['final']} != paper {PAPER_ROW_COUNT}")

    targets = pd.DataFrame(index=df.index)
    for raw_col, target_col in RAW_TO_TARGET.items():
        targets[target_col] = df[raw_col].astype(str)

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
    print("Flight Feedback preprocessing validation vs paper Table 1")
    print("=" * 60)
    print(f"Rows (paper/ours): {PAPER_ROW_COUNT} / {result.row_counts['final']}")
    print(f"Features (paper/ours): {PAPER_FEATURE_COUNT} / {len(result.feature_column_names)}")
    for target in TARGET_COLUMNS:
        ours = result.imbalance_ratios[target]
        paper = PAPER_IR[target]
        print(f"  {target}: IR1 {ours['IR1']} vs {paper[0]}, IR2 {ours['IR2']} vs {paper[1]}")
