from __future__ import annotations

from pathlib import Path
import pandas as pd


def run_loco_eligibility_preflight(
    data,
    output_dir: str | Path,
    *,
    minimum_test_samples: int = 15,
) -> Path:
    """
    Create cohort eligibility only; this is not full LOCO performance.

    Input may be sample-level metadata with GEO_ID and Label, or a CSV
    containing those columns. The jointly ComBat-harmonized matrix is
    never used as a left-out-cohort test matrix here.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    frame = (
        data.copy()
        if isinstance(data, pd.DataFrame)
        else pd.read_csv(Path(data))
    )

    if not {"GEO_ID", "Label"}.issubset(frame.columns):
        raise ValueError(
            "LOCO preflight requires sample-level GEO_ID and Label."
        )

    labels = frame["Label"].astype(str).str.lower()
    sample_level = pd.DataFrame({
        "GEO_ID": frame["GEO_ID"].astype(str),
        "Cancer": labels.str.contains(
            "cancer|tumor",
            regex=True,
        ).astype(int),
        "Normal": labels.str.contains(
            "normal",
            regex=True,
        ).astype(int),
    })

    result = (
        sample_level
        .groupby("GEO_ID", as_index=False)[
            ["Cancer", "Normal"]
        ]
        .sum()
    )
    result["Total"] = result["Cancer"] + result["Normal"]
    result["Both_classes"] = (
        (result["Cancer"] > 0)
        & (result["Normal"] > 0)
    )
    result["Minimum_size_pass"] = (
        result["Total"] >= int(minimum_test_samples)
    )
    result["ROC_AUC_eligible"] = (
        result["Both_classes"]
        & result["Minimum_size_pass"]
    )
    result["LOCO_status"] = result["ROC_AUC_eligible"].map({
        True: "Eligible after full method approval",
        False: "Descriptive only / skip ROC-AUC",
    })
    result["Safety_note"] = (
        "Eligibility only; full leakage-safe LOCO is not implemented."
    )

    result.to_csv(
        output / "loco_cohort_eligibility.csv",
        index=False,
    )
    (output / "README_LOCO.txt").write_text(
        (
            "Eligibility only. Full LOCO performance remains disabled "
            "until a training-only harmonization and unseen-cohort "
            "transfer method is approved.\n"
        ),
        encoding="utf-8",
    )
    return output
