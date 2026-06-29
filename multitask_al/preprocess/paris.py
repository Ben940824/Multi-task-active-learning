"""Paris Housing preprocessing (report_multitaskAL.pdf Table 1)."""

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

PAPER_ROW_COUNT = 10000
PAPER_TOTAL_COLUMN_COUNT = 17
PAPER_SCHEMA_FEATURE_COUNT = 14
PAPER_MODEL_FEATURE_COUNT = 13

DROPPED_COLUMNS = ["price"]
LEAKAGE_SUPPRESSED_COLUMNS = ["made"]

RAW_TARGET_COLUMNS = ["category", "isNewBuilt", "hasStorageRoom"]
TARGET_COLUMNS = ["target_category", "target_new_built", "target_storage_room"]
RAW_TO_TARGET = {
    "category": "target_category",
    "isNewBuilt": "target_new_built",
    "hasStorageRoom": "target_storage_room",
}

NUMERIC_FEATURE_COLUMNS = [
    "squareMeters",
    "numberOfRooms",
    "hasYard",
    "hasPool",
    "floors",
    "cityCode",
    "cityPartRange",
    "numPrevOwners",
    "hasStormProtector",
    "basement",
    "attic",
    "garage",
    "hasGuestRoom",
]

CATEGORICAL_FEATURE_COLUMNS: list[str] = []

FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS

PAPER_IR: dict[str, tuple[float, float]] = {
    "target_category": (6.905, 6.905),
    "target_new_built": (1.004, 1.004),
    "target_storage_room": (1.012, 1.012),
}


@dataclass
class PreprocessConfig:
    raw_path: str
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    target_columns: list[str] = field(default_factory=lambda: list(TARGET_COLUMNS))
    dropped_columns: list[str] = field(default_factory=lambda: list(DROPPED_COLUMNS))
    leakage_suppressed_columns: list[str] = field(
        default_factory=lambda: list(LEAKAGE_SUPPRESSED_COLUMNS)
    )
    notes: list[str] = field(
        default_factory=lambda: [
            "Drop price (not a paper attribute).",
            "Paper suppresses 'made' when isNewBuilt is a target (data leakage).",
            "Model features exclude 'made'; paper table still counts 17 total columns.",
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


def preprocess_paris(raw_path: Path | str) -> PreprocessResult:
    raw_path = Path(raw_path)
    config = PreprocessConfig(raw_path=str(raw_path.resolve()))

    df = pd.read_csv(raw_path)
    counts: dict[str, int] = {"raw": len(df)}
    df = mark_missing_values(df)
    df = df.drop(columns=DROPPED_COLUMNS, errors="ignore")
    df = df.drop_duplicates()
    counts["after_dedup"] = len(df)
    df = df.dropna()
    counts["final"] = len(df)

    if counts["final"] != PAPER_ROW_COUNT:
        raise ValueError(f"Row count {counts['final']} != paper {PAPER_ROW_COUNT}")

    targets = pd.DataFrame(index=df.index)
    targets["target_category"] = df["category"].astype(str)
    targets["target_new_built"] = df["isNewBuilt"].astype(int).astype(str)
    targets["target_storage_room"] = df["hasStorageRoom"].astype(int).astype(str)

    features, _ = build_mixed_features(df, NUMERIC_FEATURE_COLUMNS, CATEGORICAL_FEATURE_COLUMNS)

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
        PAPER_MODEL_FEATURE_COUNT,
        PAPER_TOTAL_COLUMN_COUNT,
        TARGET_COLUMNS,
        result.targets,
        result.imbalance_ratios,
        PAPER_IR,
    )
    report["paper_schema_feature_count"] = PAPER_SCHEMA_FEATURE_COUNT
    report["leakage_suppressed_columns"] = LEAKAGE_SUPPRESSED_COLUMNS

    config_payload: dict[str, Any] = {
        **asdict(result.config),
        "feature_column_names": result.feature_column_names,
        "raw_to_target": RAW_TO_TARGET,
        "paper_targets": {
            "row_count": PAPER_ROW_COUNT,
            "total_column_count": PAPER_TOTAL_COLUMN_COUNT,
            "schema_feature_count": PAPER_SCHEMA_FEATURE_COUNT,
            "model_feature_count": PAPER_MODEL_FEATURE_COUNT,
        },
    }
    return save_preprocessed_artifacts(
        result.features, result.targets, output_dir, config_payload, report
    )


def print_validation_summary(result: PreprocessResult) -> None:
    print("=" * 60)
    print("Paris Housing preprocessing validation vs paper Table 1")
    print("=" * 60)
    print(f"Rows (paper/ours): {PAPER_ROW_COUNT} / {result.row_counts['final']}")
    print(
        f"Features (paper schema/model): {PAPER_SCHEMA_FEATURE_COUNT} / "
        f"{len(result.feature_column_names)}"
    )
    for target in TARGET_COLUMNS:
        ours = result.imbalance_ratios[target]
        paper = PAPER_IR[target]
        print(f"  {target}: IR1 {ours['IR1']} vs {paper[0]}, IR2 {ours['IR2']} vs {paper[1]}")
