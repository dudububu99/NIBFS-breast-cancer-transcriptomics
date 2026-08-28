"""Chance-corrected feature-selection stability estimators."""
from __future__ import annotations

from collections.abc import Iterable
import numpy as np


def nogueira_stability(selected_sets: Iterable[Iterable[str]], p: int, k: int | None = None) -> float:
    """Compute the Nogueira et al. chance-corrected stability estimator.

    Parameters
    ----------
    selected_sets:
        Repeated selected feature sets. All sets must have the same size.
    p:
        Size of the eligible feature universe.
    k:
        Expected selected-set size. If omitted, inferred from the first set.

    Notes
    -----
    This implementation uses the finite-resample correction M/(M-1), matching
    the estimator reported in the manuscript. It assumes constant-cardinality
    feature selection (k features in every resample).
    """
    sets = [set(map(str, s)) for s in selected_sets]
    m = len(sets)
    if m < 2:
        raise ValueError("At least two selected sets are required.")
    sizes = {len(s) for s in sets}
    if len(sizes) != 1:
        raise ValueError(f"Selected sets must have equal cardinality; found {sorted(sizes)}")
    inferred_k = next(iter(sizes))
    if k is None:
        k = inferred_k
    if inferred_k != int(k):
        raise ValueError(f"Expected k={k}, observed set size {inferred_k}.")
    if not (0 < int(k) < int(p)):
        raise ValueError("Require 0 < k < p.")

    counts: dict[str, int] = {}
    for s in sets:
        for feature in s:
            counts[feature] = counts.get(feature, 0) + 1
    q = np.fromiter((c / m for c in counts.values()), dtype=float)
    mean_feature_variance = float(np.sum(q * (1.0 - q)) / int(p))
    chance_variance = (int(k) / int(p)) * (1.0 - int(k) / int(p))
    return float(1.0 - (m / (m - 1.0)) * mean_feature_variance / chance_variance)
