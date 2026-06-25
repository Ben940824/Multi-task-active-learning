"""Evaluation metrics for per-target classification."""

from __future__ import annotations

from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    """
    Compute accuracy and macro-F1 for one target.

    macro-F1 averages F1 across classes with equal weight (paper: per-target F1).
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
