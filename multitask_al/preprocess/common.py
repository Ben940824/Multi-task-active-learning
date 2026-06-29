"""Shared preprocessing utilities aligned with report_multitaskAL.pdf."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder

MISSING_TOKEN = "?"
UNKNOWN_TOKENS = {"unknown", "Unknown", "?"}


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


def strip_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from string columns."""
    out = df.copy()
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].astype(str).str.strip()
    return out


def mark_missing_values(df: pd.DataFrame, extra_tokens: set[str] | None = None) -> pd.DataFrame:
    """Map ?, empty strings, and optional tokens to NA."""
    tokens = set(UNKNOWN_TOKENS) | (extra_tokens or set())
    out = strip_object_columns(df)
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].replace({token: pd.NA for token in tokens})
        out[col] = out[col].replace({"": pd.NA})
    return out


def drop_rows_with_unknown_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where any object column equals unknown/Unknown/?."""
    out = strip_object_columns(df)
    mask = pd.Series(True, index=out.index)
    for col in out.select_dtypes(include=["object", "string"]).columns:
        mask &= ~out[col].isin(UNKNOWN_TOKENS)
    return out.loc[mask].copy()


def build_mixed_features(
    df: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Label-encode categorical columns; keep numeric columns as float."""
    features = pd.DataFrame(index=df.index)
    encodings: dict[str, dict[str, int]] = {}

    for col in numeric_columns:
        features[col] = df[col].astype(float)

    for col in categorical_columns:
        encoder = LabelEncoder()
        features[col] = encoder.fit_transform(df[col].astype(str))
        encodings[col] = {
            str(label): int(code)
            for label, code in zip(encoder.classes_, encoder.transform(encoder.classes_))
        }

    return features, encodings


def save_preprocessed_artifacts(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    output_dir: Path | str,
    config_payload: dict[str, Any],
    report: dict[str, Any],
) -> Path:
    """Write standard preprocessed dataset artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features.to_parquet(output_dir / "features.parquet", index=False)
    features.to_csv(output_dir / "features.csv", index=False)
    targets.to_parquet(output_dir / "targets.parquet", index=False)
    targets.to_csv(output_dir / "targets.csv", index=False)

    combined = pd.concat([features, targets], axis=1)
    combined.to_parquet(output_dir / "combined.parquet", index=False)
    combined.to_csv(output_dir / "combined.csv", index=False)

    with open(output_dir / "preprocessing_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config_payload, f, sort_keys=False, allow_unicode=True)

    with open(output_dir / "preprocessing_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return output_dir


def build_standard_report(
    row_counts: dict[str, int],
    paper_row_count: int,
    feature_count: int,
    paper_feature_count: int,
    paper_total_column_count: int,
    target_columns: list[str],
    targets: pd.DataFrame,
    imbalance_ratios: dict[str, dict[str, float]],
    paper_ir: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    """Build preprocessing_report.json payload."""
    return {
        "row_counts": row_counts,
        "paper_row_count": paper_row_count,
        "row_count_delta_vs_paper": row_counts.get("final", row_counts.get("after_dropna", 0))
        - paper_row_count,
        "feature_count": feature_count,
        "paper_feature_count": paper_feature_count,
        "paper_total_column_count": paper_total_column_count,
        "combined_column_count": feature_count + len(target_columns),
        "imbalance_ratios": imbalance_ratios,
        "paper_imbalance_ratios": {
            k: {"IR1": v[0], "IR2": v[1]} for k, v in paper_ir.items()
        },
        "target_class_counts": {
            col: targets[col].value_counts().to_dict() for col in target_columns
        },
    }
