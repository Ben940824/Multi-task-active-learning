"""Online Shopper Intention preprocessing (report_multitaskAL.pdf Table 1)."""

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

PAPER_ROW_COUNT = 12124
PAPER_TOTAL_COLUMN_COUNT = 18
PAPER_FEATURE_COUNT = 15

RAW_TARGET_COLUMNS = ["Revenue", "VisitorType", "SpecialDay"]
TARGET_COLUMNS = ["target_revenue", "target_visitor_type", "target_special_day"]

RAW_TO_TARGET = {
    "Revenue": "target_revenue",
    "VisitorType": "target_visitor_type",
    "SpecialDay": "target_special_day",
}

NUMERIC_FEATURE_COLUMNS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "OperatingSystems",
    "Region",
    "TrafficType",
]

CATEGORICAL_FEATURE_COLUMNS = [
    "Month",
    "Weekend",
    "Browser",
]

# SpecialDay appears in both features (as numeric page context) and as a target.
# Paper lists SpecialDay as target (M); exclude from features to avoid leakage.
FEATURE_NUMERIC_COLUMNS = [c for c in NUMERIC_FEATURE_COLUMNS if c != "SpecialDay"]
FEATURE_COLUMNS = FEATURE_NUMERIC_COLUMNS + CATEGORICAL_FEATURE_COLUMNS

PAPER_IR: dict[str, tuple[float, float]] = {
    "target_revenue": (5.408, 5.408),
    "target_visitor_type": (6.161, 6.161),
    "target_special_day": (48.22, 70.617),
}


@dataclass
class PreprocessConfig:
    raw_path: str
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    target_columns: list[str] = field(default_factory=lambda: list(TARGET_COLUMNS))
    label_encodings: dict[str, dict[str, int]] = field(default_factory=dict)
    notes: list[str] = field(
        default_factory=lambda: [
            "Paper: remove VisitorType=='Other' (typo: 'Bank Intention' in report).",
            "SpecialDay kept as numeric target; excluded from features.",
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


def preprocess_shopper(raw_path: Path | str) -> PreprocessResult:
    raw_path = Path(raw_path)
    config = PreprocessConfig(raw_path=str(raw_path.resolve()))

    df = pd.read_csv(raw_path)
    counts: dict[str, int] = {"raw": len(df)}
    df = mark_missing_values(df)
    df = df.drop_duplicates()
    counts["after_dedup"] = len(df)
    df = df.dropna()
    counts["after_dropna"] = len(df)
    df = df[df["VisitorType"] != "Other"].copy()
    counts["final"] = len(df)

    if counts["final"] != PAPER_ROW_COUNT:
        raise ValueError(f"Row count {counts['final']} != paper {PAPER_ROW_COUNT}")

    targets = pd.DataFrame(index=df.index)
    targets["target_revenue"] = df["Revenue"].astype(str)
    targets["target_visitor_type"] = df["VisitorType"].astype(str)
    targets["target_special_day"] = df["SpecialDay"].astype(float)

    features, encodings = build_mixed_features(
        df, FEATURE_NUMERIC_COLUMNS, CATEGORICAL_FEATURE_COLUMNS
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
    print("Shopper Intention preprocessing validation vs paper Table 1")
    print("=" * 60)
    print(f"Rows (paper/ours): {PAPER_ROW_COUNT} / {result.row_counts['final']}")
    print(f"Features (paper/ours): {PAPER_FEATURE_COUNT} / {len(result.feature_column_names)}")
    for target in TARGET_COLUMNS:
        ours = result.imbalance_ratios[target]
        paper = PAPER_IR[target]
        print(f"  {target}: IR1 {ours['IR1']} vs {paper[0]}, IR2 {ours['IR2']} vs {paper[1]}")
