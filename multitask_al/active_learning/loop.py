"""Active learning orchestration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from multitask_al.active_learning.query import select_query_indices
from multitask_al.data.split import DatasetSplit
from multitask_al.eval.metrics import classification_metrics
from multitask_al.eval.plots import plot_all_metrics
from multitask_al.models.rf_single_output import SingleOutputRFModels
from multitask_al.preprocess.imdb import TARGET_COLUMNS

DEFAULT_N_STEPS = 20
DEFAULT_M_PER_TARGET = 15
DEFAULT_QUERY_BUDGET = 45


@dataclass
class StepMetrics:
    """Metrics recorded after one train/eval cycle."""

    step: int
    n_train: int
    n_pool: int
    n_queried: int
    target_imdb_score_accuracy: float
    target_imdb_score_f1_macro: float
    target_content_rating_accuracy: float
    target_content_rating_f1_macro: float
    target_gross_accuracy: float
    target_gross_f1_macro: float


@dataclass
class ActiveLearningRun:
    """Full baseline AL run state and history."""

    steps: list[StepMetrics] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)


def _evaluate(
    models: SingleOutputRFModels,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Return per-target accuracy and macro-F1 on the fixed test set."""
    y_pred = models.predict(X_test)
    return {
        col: classification_metrics(y_test[col], y_pred[col])
        for col in models.target_columns
    }


def _metrics_row(
    step: int,
    n_train: int,
    n_pool: int,
    n_queried: int,
    evals: dict[str, dict[str, float]],
) -> StepMetrics:
    return StepMetrics(
        step=step,
        n_train=n_train,
        n_pool=n_pool,
        n_queried=n_queried,
        target_imdb_score_accuracy=evals["target_imdb_score"]["accuracy"],
        target_imdb_score_f1_macro=evals["target_imdb_score"]["f1_macro"],
        target_content_rating_accuracy=evals["target_content_rating"]["accuracy"],
        target_content_rating_f1_macro=evals["target_content_rating"]["f1_macro"],
        target_gross_accuracy=evals["target_gross"]["accuracy"],
        target_gross_f1_macro=evals["target_gross"]["f1_macro"],
    )


def run_baseline_active_learning(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    split: DatasetSplit,
    target_columns: list[str] | None = None,
    n_steps: int = DEFAULT_N_STEPS,
    m_per_target: int = DEFAULT_M_PER_TARGET,
    query_budget: int = DEFAULT_QUERY_BUDGET,
    rf_random_state: int = 42,
) -> ActiveLearningRun:
    """
    Paper Algorithm 1 baseline (no resampling).

    Loop (n_steps times):
      train 3 independent RF on D_train -> eval D_test -> query m_per_target
      least-confident pool rows per target -> move queried rows train -> pool
    """
    target_columns = target_columns or list(TARGET_COLUMNS)

    train_indices = list(split.train_indices)
    pool_indices = list(split.pool_indices)
    test_indices = split.test_indices

    X_test = features.iloc[test_indices]
    y_test = targets.iloc[test_indices]

    rf_kwargs = {
        "n_estimators": 100,
        "random_state": rf_random_state,
        "n_jobs": -1,
    }
    models = SingleOutputRFModels(target_columns=target_columns, rf_kwargs=rf_kwargs)

    run = ActiveLearningRun(
        config={
            "n_steps": n_steps,
            "m_per_target": m_per_target,
            "query_budget": query_budget,
            "rf_random_state": rf_random_state,
            "sampling": None,
            "split_random_state": split.random_state,
        }
    )

    for step in range(n_steps + 1):
        X_train = features.iloc[train_indices]
        y_train = targets.iloc[train_indices]
        n_train_before_query = len(train_indices)
        n_pool_before_query = len(pool_indices)

        models.fit(X_train, y_train)
        evals = _evaluate(models, X_test, y_test)

        n_queried = 0
        if step < n_steps and pool_indices:
            X_pool = features.iloc[pool_indices]
            uncertainties = models.uncertainty(X_pool)
            queried = select_query_indices(
                pool_indices=pool_indices,
                uncertainties=uncertainties,
                target_columns=target_columns,
                m_per_target=m_per_target,
                budget=query_budget,
            )
            n_queried = len(queried)
            queried_set = set(queried)
            pool_indices = [idx for idx in pool_indices if idx not in queried_set]
            train_indices.extend(queried)

        run.steps.append(
            _metrics_row(
                step,
                n_train_before_query,
                n_pool_before_query,
                n_queried,
                evals,
            )
        )

    return run


def steps_to_dataframe(run: ActiveLearningRun) -> pd.DataFrame:
    """Convert step metrics to a flat table for CSV export."""
    return pd.DataFrame([asdict(s) for s in run.steps])


def save_run(run: ActiveLearningRun, output_dir: Path | str) -> Path:
    """Write metrics CSV, run config JSON, and curve plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = steps_to_dataframe(run)
    df.to_csv(output_dir / "metrics.csv", index=False)
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run.config, f, indent=2)

    plot_all_metrics(df, output_dir)

    return output_dir
