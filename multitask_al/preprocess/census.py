"""
Census (UCI Adult) preprocessing aligned with report_multitaskAL.pdf (Table 1).

Paper documents:
  - 15 columns in the raw table (3 targets + 12 features)
  - 45194 instances after dropping unknown/null values and duplicates
  - Income (B), Marital Status (M), Work Class (M) as the three targets

The raw CSV has no header row; columns follow the UCI Adult dataset schema.

Preprocessing:
  - Strip whitespace from string fields
  - Treat "?" as missing and drop rows with any missing value
  - Remove duplicate rows
  - Normalize income labels by removing trailing "."
  - Label-encode categorical feature columns; keep numeric features as-is
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Paper-aligned constants
# ---------------------------------------------------------------------------

PAPER_ROW_COUNT = 45194
PAPER_TOTAL_COLUMN_COUNT = 15
PAPER_FEATURE_COUNT = 12

MISSING_TOKEN = "?"

# UCI Adult column order (headerless raw CSV).
RAW_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

RAW_TARGET_COLUMNS = ["income", "marital-status", "workclass"]

NUMERIC_FEATURE_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

CATEGORICAL_FEATURE_COLUMNS = [
    "education",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS

TARGET_COLUMNS = ["target_income", "target_marital_status", "target_workclass"]

RAW_TO_TARGET = {
    "income": "target_income",
    "marital-status": "target_marital_status",
    "workclass": "target_workclass",
}

# Income is binarized as <=50K vs >50K (paper: Income (B)).
INCOME_POSITIVE_LABEL = ">50K"
INCOME_NEGATIVE_LABEL = "<=50K"

PAPER_IR: dict[str, tuple[float, float]] = {
    "target_income": (3.033, 3.033),
    "target_marital_status": (122.011, 657.75),
    "target_workclass": (277.553, 1584.81),
}


@dataclass
class PreprocessConfig:
    """Serializable record of every preprocessing decision."""

    raw_path: str
    missing_token: str = MISSING_TOKEN
    raw_columns: list[str] = field(default_factory=lambda: list(RAW_COLUMNS))
    numeric_feature_columns: list[str] = field(
        default_factory=lambda: list(NUMERIC_FEATURE_COLUMNS)
    )
    categorical_feature_columns: list[str] = field(
        default_factory=lambda: list(CATEGORICAL_FEATURE_COLUMNS)
    )
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    raw_target_columns: list[str] = field(default_factory=lambda: list(RAW_TARGET_COLUMNS))
    target_columns: list[str] = field(default_factory=lambda: list(TARGET_COLUMNS))
    income_positive_label: str = INCOME_POSITIVE_LABEL
    income_negative_label: str = INCOME_NEGATIVE_LABEL
    label_encodings: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass
class PreprocessResult:
    """Outputs of the Census preprocessing pipeline."""

    features: pd.DataFrame
    targets: pd.DataFrame
    config: PreprocessConfig
    row_counts: dict[str, int]
    imbalance_ratios: dict[str, dict[str, float]]
    feature_column_names: list[str]


def compute_imbalance_ratios(series: pd.Series) -> dict[str, float]:
    """
    IR definitions from the paper (Table 1 footnote):
      IR2 = majority_count / minority_count
      IR1 = mean(majority_count / count_j) over non-majority classes j
    """
    counts = series.value_counts()
    values = counts.values.astype(float)
    majority = values.max()
    minority = values.min()
    majority_idx = int(values.argmax())

    ir2 = majority / minority
    ir1 = float(
        np.mean([majority / n for i, n in enumerate(values) if i != majority_idx])
    )
    return {"IR1": round(ir1, 3), "IR2": round(ir2, 3)}


def _load_raw(raw_path: Path) -> pd.DataFrame:
    """Load headerless UCI Adult CSV with canonical column names."""
    df = pd.read_csv(
        raw_path,
        header=None,
        names=RAW_COLUMNS,
        skipinitialspace=True,
    )
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def _clean_rows(df: pd.DataFrame, missing_token: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop unknown tokens, duplicates, and normalize income labels."""
    counts: dict[str, int] = {"raw": len(df)}
    df = df.replace(missing_token, pd.NA)
    df = df.drop_duplicates()
    counts["after_dedup"] = len(df)
    df = df.dropna()
    counts["after_dropna"] = len(df)
    df = df.copy()
    df["income"] = df["income"].str.rstrip(".")
    counts["after_income_normalize"] = len(df)
    return df.reset_index(drop=True), counts


def _build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create the three classification targets described in the paper."""
    targets = pd.DataFrame(index=df.index)
    targets["target_income"] = np.where(
        df["income"] == INCOME_POSITIVE_LABEL,
        INCOME_POSITIVE_LABEL,
        INCOME_NEGATIVE_LABEL,
    )
    targets["target_marital_status"] = df["marital-status"].astype(str)
    targets["target_workclass"] = df["workclass"].astype(str)
    return targets


def _build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Build the 12-column feature matrix."""
    features = pd.DataFrame(index=df.index)
    encodings: dict[str, dict[str, int]] = {}

    for col in NUMERIC_FEATURE_COLUMNS:
        features[col] = df[col].astype(float)

    for col in CATEGORICAL_FEATURE_COLUMNS:
        encoder = LabelEncoder()
        features[col] = encoder.fit_transform(df[col].astype(str))
        encodings[col] = {
            str(label): int(code)
            for label, code in zip(encoder.classes_, encoder.transform(encoder.classes_))
        }

    return features, encodings


def preprocess_census(raw_path: Path | str) -> PreprocessResult:
    """
    Run the full Census preprocessing pipeline.

    Returns feature matrix (12 cols), targets (3 cols), and diagnostic metadata.
    """
    raw_path = Path(raw_path)
    config = PreprocessConfig(raw_path=str(raw_path.resolve()))

    df = _load_raw(raw_path)
    df, row_counts = _clean_rows(df, MISSING_TOKEN)

    if row_counts["after_income_normalize"] != PAPER_ROW_COUNT:
        raise ValueError(
            f"Row count {row_counts['after_income_normalize']} != paper target "
            f"{PAPER_ROW_COUNT}. Check missing-token handling."
        )

    targets = _build_targets(df)
    features, encodings = _build_features(df)
    config.label_encodings = encodings

    if len(features.columns) != PAPER_FEATURE_COUNT:
        raise ValueError(
            f"Feature count {len(features.columns)} != paper target {PAPER_FEATURE_COUNT}."
        )

    imbalance_ratios = {
        col: compute_imbalance_ratios(targets[col]) for col in TARGET_COLUMNS
    }

    return PreprocessResult(
        features=features,
        targets=targets,
        config=config,
        row_counts=row_counts,
        imbalance_ratios=imbalance_ratios,
        feature_column_names=list(features.columns),
    )


def save_preprocessed(result: PreprocessResult, output_dir: Path | str) -> Path:
    """Write processed artifacts and a human-readable report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result.features.to_parquet(output_dir / "features.parquet", index=False)
    result.features.to_csv(output_dir / "features.csv", index=False)
    result.targets.to_parquet(output_dir / "targets.parquet", index=False)
    result.targets.to_csv(output_dir / "targets.csv", index=False)

    combined = pd.concat([result.features, result.targets], axis=1)
    combined.to_parquet(output_dir / "combined.parquet", index=False)
    combined.to_csv(output_dir / "combined.csv", index=False)

    config_payload: dict[str, Any] = {
        **asdict(result.config),
        "feature_column_names": result.feature_column_names,
        "target_columns": TARGET_COLUMNS,
        "raw_to_target": RAW_TO_TARGET,
        "paper_targets": {
            "row_count": PAPER_ROW_COUNT,
            "total_column_count": PAPER_TOTAL_COLUMN_COUNT,
            "feature_count": PAPER_FEATURE_COUNT,
        },
    }
    with open(output_dir / "preprocessing_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config_payload, f, sort_keys=False, allow_unicode=True)

    report = {
        "row_counts": result.row_counts,
        "paper_row_count": PAPER_ROW_COUNT,
        "row_count_delta_vs_paper": result.row_counts["after_income_normalize"]
        - PAPER_ROW_COUNT,
        "feature_count": len(result.feature_column_names),
        "paper_feature_count": PAPER_FEATURE_COUNT,
        "paper_total_column_count": PAPER_TOTAL_COLUMN_COUNT,
        "combined_column_count": len(result.feature_column_names) + len(TARGET_COLUMNS),
        "imbalance_ratios": result.imbalance_ratios,
        "paper_imbalance_ratios": {
            k: {"IR1": v[0], "IR2": v[1]} for k, v in PAPER_IR.items()
        },
        "target_class_counts": {
            col: result.targets[col].value_counts().to_dict() for col in TARGET_COLUMNS
        },
    }
    with open(output_dir / "preprocessing_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return output_dir


def print_validation_summary(result: PreprocessResult) -> None:
    """Print a side-by-side comparison against paper Table 1."""
    print("=" * 60)
    print("Census preprocessing validation vs paper Table 1")
    print("=" * 60)
    rc = result.row_counts
    print(f"Rows (paper):              {PAPER_ROW_COUNT}")
    print(f"Rows (ours):               {rc['after_income_normalize']}")
    print(
        f"  raw -> dedup -> dropna:  {rc['raw']} -> "
        f"{rc['after_dedup']} -> {rc['after_dropna']}"
    )
    print(f"Features (paper):          {PAPER_FEATURE_COUNT}")
    print(f"Features (ours):           {len(result.feature_column_names)}")
    print(f"Total columns (paper):     {PAPER_TOTAL_COLUMN_COUNT} (12 features + 3 targets)")
    print()
    print("Imbalance ratios (ours vs paper):")
    for target in TARGET_COLUMNS:
        ours = result.imbalance_ratios[target]
        paper = PAPER_IR[target]
        print(
            f"  {target}:"
            f" IR1 {ours['IR1']} vs {paper[0]},"
            f" IR2 {ours['IR2']} vs {paper[1]}"
        )
    print()
    print("Target class counts:")
    for target in TARGET_COLUMNS:
        counts = result.targets[target].value_counts().to_dict()
        print(f"  {target}: {counts}")
