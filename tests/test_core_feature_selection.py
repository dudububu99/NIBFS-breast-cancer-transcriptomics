from __future__ import annotations

import math

import pandas as pd

from src.feature_selection import nibfs_rank, pairwise_jaccard_summary
from src.stability_estimators import nogueira_stability


def test_nibfs_borda_ranking_is_deterministic():
    limma = pd.DataFrame(
        {
            "Gene": ["B", "A", "C"],
            "Rank_stat": [1.0, 1.0, 3.0],
        }
    )
    ppi = pd.DataFrame(
        {
            "Gene": ["A", "B", "C"],
            "Degree": [1.0, 1.0, 3.0],
            "Normalized_degree": [1.0, 1.0, 3.0],
            "Rank_topo": [2.5, 2.5, 1.0],
        }
    )
    ranked = nibfs_rank(limma, ppi, ["A", "B", "C"])
    assert ranked["Gene"].tolist() == ["A", "B", "C"]
    assert ranked["Rank_NIBFS"].tolist() == [1, 2, 3]


def test_pairwise_jaccard_summary_known_value():
    panels = {1: ["A", "B"], 2: ["A", "C"], 3: ["A", "B"]}
    pairwise, freq, summary = pairwise_jaccard_summary(panels, "demo", 2)
    assert len(pairwise) == 3
    assert math.isclose(summary["Mean_Jaccard"], 5.0 / 9.0, rel_tol=0, abs_tol=1e-12)
    counts = dict(zip(freq["Gene"], freq["Fold_Frequency"]))
    assert counts == {"A": 3, "B": 2, "C": 1}


def test_nogueira_identical_panels_equal_one():
    value = nogueira_stability([{"A", "B"}, {"A", "B"}, {"A", "B"}], p=10, k=2)
    assert math.isclose(value, 1.0, rel_tol=0, abs_tol=1e-12)
