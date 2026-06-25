"""
Mushroom preprocessing aligned with report_multitaskAL.pdf (Table 1).

Paper documents:
  - 23 columns in the raw table (3 targets + 20 categorical features)
  - 5644 instances after dropping unknown/null values
  - Class (B), Population (M), Habitat (M) as the three targets

Preprocessing:
  - Treat "?" as missing and drop rows with any missing value (stalk-root)
  - Remove duplicates (none in UCI export, but applied for consistency)
  - Label-encode each of the 20 feature columns to integers
  - Keep target labels as original single-letter strings for sklearn
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

PAPER_ROW_COUNT = 5644
PAPER_TOTAL_COLUMN_COUNT = 23
PAPER_FEATURE_COUNT = 20

MISSING_TOKEN = "?"

# Raw CSV column names (Class + 20 features + 2 auxiliary targets).
RAW_TARGET_COLUMNS = ["Class", "population", "habitat"]

FEATURE_COLUMNS = [
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
]

TARGET_COLUMNS = ["target_class", "target_population", "target_habitat"]

RAW_TO_TARGET = {
    "Class": "target_class",
    "population": "target_population",
    "habitat": "target_habitat",
}

PAPER_IR: dict[str, tuple[float, float]] = {
    "target_class": (1.618, 1.618),
    "target_population": (11.767, 41.538),
    "target_habitat": (11.994, 38.938),
}


@dataclass
class PreprocessConfig:
    """Serializable record of every preprocessing decision."""

    raw_path: str
    missing_token: str = MISSING_TOKEN
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    raw_target_columns: list[str] = field(default_factory=lambda: list(RAW_TARGET_COLUMNS))
    target_columns: list[str] = field(default_factory=lambda: list(TARGET_COLUMNS))
    label_encodings: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass
class PreprocessResult:
    """Outputs of the Mushroom preprocessing pipeline."""

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


def _clean_rows(df: pd.DataFrame, missing_token: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove duplicates and rows with unknown/null tokens."""
    counts: dict[str, int] = {"raw": len(df)}
    df = df.drop_duplicates()
    counts["after_dedup"] = len(df)
    df = df.replace(missing_token, pd.NA).dropna()
    counts["after_dropna"] = len(df)
    return df.reset_index(drop=True), counts


def _build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw target columns to paper-style target names."""
    targets = pd.DataFrame(index=df.index)
    for raw_col, target_col in RAW_TO_TARGET.items():
        targets[target_col] = df[raw_col].astype(str)
    return targets


def _label_encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Fit per-column label encoders on the cleaned feature columns."""
    features = pd.DataFrame(index=df.index)
    encodings: dict[str, dict[str, int]] = {}

    for col in FEATURE_COLUMNS:
        encoder = LabelEncoder()
        features[col] = encoder.fit_transform(df[col].astype(str))
        encodings[col] = {
            str(label): int(code) for label, code in zip(encoder.classes_, encoder.transform(encoder.classes_))
        }

    return features, encodings


def preprocess_mushroom(raw_path: Path | str) -> PreprocessResult:
    """
    Run the full Mushroom preprocessing pipeline.

    Returns feature matrix (20 cols), targets (3 cols), and diagnostic metadata.
    """
    raw_path = Path(raw_path)
    config = PreprocessConfig(raw_path=str(raw_path.resolve()))

    df = pd.read_csv(raw_path)
    df, row_counts = _clean_rows(df, MISSING_TOKEN)

    if row_counts["after_dropna"] != PAPER_ROW_COUNT:
        raise ValueError(
            f"Row count {row_counts['after_dropna']} != paper target {PAPER_ROW_COUNT}. "
            "Check missing-token handling."
        )

    targets = _build_targets(df)
    features, encodings = _label_encode_features(df)
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

    # Combined table for inspection (23 columns = 20 features + 3 targets).
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
        "row_count_delta_vs_paper": result.row_counts["after_dropna"] - PAPER_ROW_COUNT,
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
    print("Mushroom preprocessing validation vs paper Table 1")
    print("=" * 60)
    print(f"Rows (paper):              {PAPER_ROW_COUNT}")
    print(f"Rows (ours):               {result.row_counts['after_dropna']}")
    print(f"  raw -> dedup -> dropna:  {result.row_counts['raw']} -> "
          f"{result.row_counts['after_dedup']} -> {result.row_counts['after_dropna']}")
    print(f"Features (paper):          {PAPER_FEATURE_COUNT}")
    print(f"Features (ours):           {len(result.feature_column_names)}")
    print(f"Total columns (paper):     {PAPER_TOTAL_COLUMN_COUNT} (20 features + 3 targets)")
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
