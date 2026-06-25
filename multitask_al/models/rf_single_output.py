"""
Three independent RandomForestClassifier models (single-output baseline).

Each target gets its own classifier trained only on that target's labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

DEFAULT_RF_KWARGS = {
    "n_estimators": 100,
    "random_state": 42,
    "n_jobs": -1,
}


@dataclass
class SingleOutputRFModels:
    """One Random Forest per target column."""

    target_columns: list[str]
    rf_kwargs: dict = field(default_factory=lambda: dict(DEFAULT_RF_KWARGS))
    models: dict[str, RandomForestClassifier] = field(default_factory=dict)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """Train one classifier per target on the same feature matrix."""
        self.models = {}
        for col in self.target_columns:
            clf = RandomForestClassifier(**self.rf_kwargs)
            clf.fit(X, y[col])
            self.models[col] = clf

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict all targets."""
        preds = {col: self.models[col].predict(X) for col in self.target_columns}
        return pd.DataFrame(preds, index=X.index)

    def uncertainty(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """
        Least-confidence score per row: 1 - max(class probability).

        Lower values mean the model is more confident; we query highest values
        (lowest confidence) first.
        """
        scores: dict[str, np.ndarray] = {}
        for col in self.target_columns:
            proba = self.models[col].predict_proba(X)
            confidence = proba.max(axis=1)
            scores[col] = 1.0 - confidence
        return scores
