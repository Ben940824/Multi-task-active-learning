"""Active learning query strategies."""

from __future__ import annotations

import numpy as np


def select_query_indices(
    pool_indices: list[int],
    uncertainties: dict[str, np.ndarray],
    target_columns: list[str],
    m_per_target: int = 15,
    budget: int = 45,
) -> list[int]:
    """
    Select pool rows to label next (spec: union + fill to budget).

    1. Each target contributes ``m_per_target`` pool rows with highest uncertainty
       (lowest confidence = largest 1 - max_proba).
    2. Take the union of those picks.
    3. If union size < budget, fill from remaining pool using mean uncertainty
       across all targets until ``budget`` unique rows are chosen.
    """
    if not pool_indices:
        return []

    n_pool = len(pool_indices)
    selected: set[int] = set()

    # Step 1: top-m uncertain rows per target (highest uncertainty score).
    for col in target_columns:
        scores = uncertainties[col]
        order = np.argsort(-scores)  # descending uncertainty
        for pos in order[:m_per_target]:
            selected.add(pool_indices[int(pos)])

    # Step 2: fill to budget if union is smaller than budget.
    if len(selected) < budget:
        mean_uncertainty = np.mean(
            [uncertainties[col] for col in target_columns],
            axis=0,
        )
        remaining_positions = [
            pos for pos, idx in enumerate(pool_indices) if idx not in selected
        ]
        fill_order = sorted(
            remaining_positions,
            key=lambda pos: mean_uncertainty[pos],
            reverse=True,
        )
        for pos in fill_order:
            selected.add(pool_indices[pos])
            if len(selected) >= budget:
                break

    # Cap at budget when union exceeds it (should be rare with m_per_target=15 x 3).
    result = list(selected)
    if len(result) > budget:
        mean_uncertainty = np.mean(
            [uncertainties[col] for col in target_columns],
            axis=0,
        )
        idx_to_score = {
            pool_indices[pos]: mean_uncertainty[pos] for pos in range(n_pool)
        }
        result = sorted(result, key=lambda idx: idx_to_score[idx], reverse=True)[:budget]

    return result
