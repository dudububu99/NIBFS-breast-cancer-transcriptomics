from __future__ import annotations

import math

from src.modeling import classification_metrics


def test_classification_metrics_perfect_separation():
    metrics = classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], threshold=0.5)
    for key in ("ROC_AUC", "Accuracy", "Balanced_accuracy", "Sensitivity", "Specificity", "Precision", "F1", "MCC"):
        assert math.isclose(float(metrics[key]), 1.0, rel_tol=0, abs_tol=1e-12)
    assert metrics["TN"] == 2 and metrics["TP"] == 2
    assert metrics["FP"] == 0 and metrics["FN"] == 0
