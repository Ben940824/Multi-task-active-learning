"""
IMDB preprocessing aligned with report_multitaskAL.pdf (Table 1).

Paper documents:
  - remove duplicates, drop null/unknown, remove unique identifiers
  - IMDb score -> Bad / Average / Good
  - Gross -> binary at 15M USD
  - Content Rating -> partial relabel (mapping documented in config)

Paper does NOT document how to reach 45 feature columns; we reverse-engineer:
  14 numeric + 1 color binary + 4 language one-hot + 4 country one-hot + 22 genre flags = 45
  (cleaned data contains only 22 unique genre tokens, so all are kept)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
# ---------------------------------------------------------------------------
# Paper-aligned constants
# ---------------------------------------------------------------------------

PAPER_ROW_COUNT = 3737
PAPER_FEATURE_COUNT = 45

# Columns treated as unique identifiers / high-cardinality text (dropped from X).
IDENTIFIER_COLUMNS = [
    "director_name",
    "actor_1_name",
    "actor_2_name",
    "actor_3_name",
    "movie_title",
    "movie_imdb_link",
    "plot_keywords",
]

# Numeric predictors kept after cleaning (targets excluded).
NUMERIC_FEATURE_COLUMNS = [
    "num_critic_for_reviews",
    "duration",
    "director_facebook_likes",
    "actor_3_facebook_likes",
    "actor_1_facebook_likes",
    "num_voted_users",
    "cast_total_facebook_likes",
    "facenumber_in_poster",
    "num_user_for_reviews",
    "budget",
    "title_year",
    "actor_2_facebook_likes",
    "aspect_ratio",
    "movie_facebook_likes",
]

# After dedup+dropna only 22 unique genre tokens exist; we use all of them.
# Remaining slots filled with one-hot for top languages / countries:
#   14 numeric + 1 color + 4 language + 4 country + 22 genre = 45
LANGUAGE_TOP_K = 4
COUNTRY_TOP_K = 4

GROSS_THRESHOLD_USD = 15_000_000

# Content-rating relabel: paper says "a few classes relabelled to avoid high similarity".
CONTENT_RATING_REMAP: dict[str, str] = {
    "Approved": "PG",
    "Passed": "PG",
    "GP": "PG",
    "M": "R",
    "X": "NC-17",
    "Not Rated": "Unrated",
    "TV-14": "Unrated",
    "TV-MA": "Unrated",
    "TV-PG": "Unrated",
    "TV-G": "Unrated",
    "TV-Y": "Unrated",
    "TV-Y7": "Unrated",
}

TARGET_COLUMNS = ["target_imdb_score", "target_content_rating", "target_gross"]

PAPER_IR: dict[str, tuple[float, float]] = {
    "target_imdb_score": (13.196, 15.395),
    "target_content_rating": (40.701, 105.938),
    "target_gross": (1.908, 1.908),
}


@dataclass
class PreprocessConfig:
    """Serializable record of every preprocessing decision."""

    raw_path: str
    gross_threshold_usd: int = GROSS_THRESHOLD_USD
    language_top_k: int = LANGUAGE_TOP_K
    country_top_k: int = COUNTRY_TOP_K
    content_rating_remap: dict[str, str] = field(
        default_factory=lambda: dict(CONTENT_RATING_REMAP)
    )
    identifier_columns: list[str] = field(default_factory=lambda: list(IDENTIFIER_COLUMNS))
    numeric_feature_columns: list[str] = field(
        default_factory=lambda: list(NUMERIC_FEATURE_COLUMNS)
    )
    imdb_score_bins: dict[str, str] = field(
        default_factory=lambda: {
            "Bad": "0 <= score < 5",
            "Average": "5 <= score < 8",
            "Good": "8 <= score < 10",
        }
    )


@dataclass
class PreprocessResult:
    """Outputs of the IMDB preprocessing pipeline."""

    features: pd.DataFrame
    targets: pd.DataFrame
    config: PreprocessConfig
    genre_columns: list[str]
    language_onehot_columns: list[str]
    country_onehot_columns: list[str]
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


def bin_imdb_score(score: float) -> str:
    """Map continuous IMDb score to Bad / Average / Good per paper."""
    if score >= 8:
        return "Good"
    if score >= 5:
        return "Average"
    return "Bad"


def relabel_content_rating(value: str) -> str:
    """Apply paper-inspired content-rating merges."""
    if pd.isna(value):
        return value
    return CONTENT_RATING_REMAP.get(str(value), str(value))


def _clean_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove duplicates and rows with any null (paper: drop unknown or null values)."""
    counts: dict[str, int] = {"raw": len(df)}
    df = df.drop_duplicates()
    counts["after_dedup"] = len(df)
    df = df.dropna()
    counts["after_dropna"] = len(df)
    return df.reset_index(drop=True), counts


def _build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create the three classification targets described in the paper."""
    targets = pd.DataFrame(index=df.index)
    targets["target_imdb_score"] = df["imdb_score"].apply(bin_imdb_score)
    targets["target_content_rating"] = df["content_rating"].apply(relabel_content_rating)
    targets["target_gross"] = (df["gross"] >= GROSS_THRESHOLD_USD).astype(int)
    return targets


def _top_tokens(series: pd.Series, k: int) -> list[str]:
    """Return the k most frequent categorical tokens."""
    counts = series.fillna("").astype(str).str.strip()
    counts = counts[counts != ""]
    return counts.value_counts().head(k).index.tolist()


def _all_genres(series: pd.Series) -> list[str]:
    """Return every genre token sorted by frequency (post-cleaning there are 22)."""
    tokens = series.fillna("").str.split("|").explode().str.strip()
    tokens = tokens[tokens != ""]
    return tokens.value_counts().index.tolist()


def _one_hot_top(
    df: pd.DataFrame,
    column: str,
    prefix: str,
    top_values: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Create binary columns for the most frequent values of a categorical field."""
    block = pd.DataFrame(index=df.index)
    col_names: list[str] = []
    for value in top_values:
        safe = value.replace(" ", "_").replace("/", "_")
        name = f"{prefix}__{safe}"
        block[name] = (df[column].astype(str) == value).astype(int)
        col_names.append(name)
    return block, col_names


def _build_features(
    df: pd.DataFrame,
    genre_names: list[str],
    language_top: list[str],
    country_top: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build the 45-column feature matrix."""
    features = pd.DataFrame(index=df.index)

    # 14 numeric columns.
    for col in NUMERIC_FEATURE_COLUMNS:
        features[col] = df[col].astype(float)

    # color -> binary (Color=1, Black and White=0).
    features["color_is_color"] = (df["color"] == "Color").astype(int)

    lang_block, lang_cols = _one_hot_top(df, "language", "language", language_top)
    country_block, country_cols = _one_hot_top(df, "country", "country", country_top)
    features = pd.concat([features, lang_block, country_block], axis=1)

    # genres -> binary flags for every genre token in the cleaned dataset.
    genres_filled = df["genres"].fillna("")
    for genre in genre_names:
        col_name = f"genre__{genre.replace(' ', '_')}"
        features[col_name] = genres_filled.str.contains(
            genre, regex=False, na=False
        ).astype(int)

    return features, lang_cols, country_cols


def preprocess_imdb(raw_path: Path | str) -> PreprocessResult:
    """
    Run the full IMDB preprocessing pipeline.

    Returns feature matrix (45 cols), targets (3 cols), and diagnostic metadata.
    """
    raw_path = Path(raw_path)
    config = PreprocessConfig(raw_path=str(raw_path.resolve()))

    df = pd.read_csv(raw_path)
    df, row_counts = _clean_rows(df)

    genre_names = _all_genres(df["genres"])
    language_top = _top_tokens(df["language"], LANGUAGE_TOP_K)
    country_top = _top_tokens(df["country"], COUNTRY_TOP_K)
    targets = _build_targets(df)
    features, lang_cols, country_cols = _build_features(
        df, genre_names, language_top, country_top
    )

    expected_features = (
        len(NUMERIC_FEATURE_COLUMNS)
        + 1
        + len(language_top)
        + len(country_top)
        + len(genre_names)
    )
    if expected_features != PAPER_FEATURE_COUNT:
        raise ValueError(
            f"Feature count {expected_features} != paper target {PAPER_FEATURE_COUNT}. "
            "Adjust LANGUAGE_TOP_K / COUNTRY_TOP_K."
        )

    genre_columns = [c for c in features.columns if c.startswith("genre__")]
    imbalance_ratios = {
        col: compute_imbalance_ratios(targets[col]) for col in TARGET_COLUMNS
    }

    return PreprocessResult(
        features=features,
        targets=targets,
        config=config,
        genre_columns=genre_columns,
        language_onehot_columns=lang_cols,
        country_onehot_columns=country_cols,
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

    # Combined table for inspection (48 columns = 45 features + 3 targets).
    combined = pd.concat([result.features, result.targets], axis=1)
    combined.to_parquet(output_dir / "combined.parquet", index=False)
    combined.to_csv(output_dir / "combined.csv", index=False)

    config_payload: dict[str, Any] = {
        **asdict(result.config),
        "genre_columns": result.genre_columns,
        "feature_column_names": result.feature_column_names,
        "language_onehot_columns": result.language_onehot_columns,
        "country_onehot_columns": result.country_onehot_columns,
        "language_top_values": [c.replace("language__", "") for c in result.language_onehot_columns],
        "country_top_values": [c.replace("country__", "") for c in result.country_onehot_columns],
        "target_columns": TARGET_COLUMNS,
        "paper_targets": {
            "row_count": PAPER_ROW_COUNT,
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
    print("IMDB preprocessing validation vs paper Table 1")
    print("=" * 60)
    print(f"Rows (paper):     {PAPER_ROW_COUNT}")
    print(f"Rows (ours):      {result.row_counts['after_dropna']}")
    print(f"  raw -> dedup:   {result.row_counts['raw']} -> {result.row_counts['after_dedup']}")
    print(f"Features (paper): {PAPER_FEATURE_COUNT}")
    print(f"Features (ours):  {len(result.feature_column_names)}")
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
