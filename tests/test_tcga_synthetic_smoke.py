
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pandas as pd

MODULE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "tcga_brca_rnaseq_external_validation.py"
)
spec = importlib.util.spec_from_file_location("rna_validation", MODULE)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)

def test_rank_and_signed_score():
    genes = module.EXPECTED_TOP20 + ["A", "B", "C"]
    dev = pd.DataFrame(
        np.arange(46).reshape(2, 23),
        index=["D1", "D2"],
        columns=genes,
    )
    ranked = module._within_sample_percentile_rank(dev)
    panel = pd.DataFrame(
        {
            "Rank_NIBFS": range(1, 21),
            "Gene": module.EXPECTED_TOP20,
            "Discovery_logFC": [
                module.EXPECTED_LOGFC[g]
                for g in module.EXPECTED_TOP20
            ],
            "Discovery_direction": [
                "Up_in_cancer" if module.EXPECTED_LOGFC[g] > 0
                else "Down_in_cancer"
                for g in module.EXPECTED_TOP20
            ],
        }
    )
    score = module._signed_panel_score(
        ranked[module.EXPECTED_TOP20],
        panel,
    )
    assert score.shape == (2,)
    assert np.isfinite(score).all()

def test_pair_bootstrap():
    rows = []
    for pair in range(12):
        for model in ("M1", "M2"):
            for label in (0, 1):
                rows.append(
                    {
                        "Pair_ID": f"P{pair}",
                        "Label": label,
                        "Model": model,
                        "Score": 0.1 if label == 0 else 0.9,
                        "Threshold": 0.5,
                    }
                )
    predictions = pd.DataFrame(rows)
    summary, bootstrap = module._pair_bootstrap_metrics(
        predictions,
        bootstrap_replicates=200,
        random_state=42,
    )
    aucs = summary.loc[
        summary["Metric"].eq("ROC_AUC"),
        "Estimate",
    ]
    assert len(aucs) == 2
    assert (aucs == 1.0).all()
    assert not bootstrap.empty
