from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .common import (
    AdditionalAnalysisPaths,
    acquire_reference_geo_series_matrix,
    parse_series_matrix_sample_headers,
    write_json,
)


def _patient_code_from_title(title: str) -> str:
    m = re.search(r"\b(BC\d+)\s*[NT]\b", str(title).upper())
    if not m:
        m = re.search(r"\b(BC\d+)[NT]\b", str(title).upper())
    return m.group(1) if m else ""


def _label_from_title(title: str) -> int | None:
    t = str(title).strip().upper()
    if t.startswith("NORMAL ") or re.search(r"BC\d+N\b", t):
        return 0
    if t.startswith("CANCER ") or re.search(r"BC\d+T\b", t):
        return 1
    return None


def pair_cluster_auc_ci(
    frame: pd.DataFrame,
    pair_col: str = "Pair_ID",
    y_col: str = "True_Label",
    p_col: str = "Probability",
    iterations: int = 2000,
    seed: int = 20260829,
    alpha: float = 0.05,
) -> tuple[float, float, np.ndarray]:
    pairs = list(pd.unique(frame[pair_col]))
    blocks = {pid: frame.loc[frame[pair_col].eq(pid), [y_col, p_col]].copy() for pid in pairs}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(iterations):
        sampled = rng.choice(pairs, size=len(pairs), replace=True)
        y_parts: list[np.ndarray] = []
        p_parts: list[np.ndarray] = []
        for pid in sampled:
            block = blocks[pid]
            y_parts.append(block[y_col].to_numpy(dtype=int))
            p_parts.append(block[p_col].to_numpy(dtype=float))
        y = np.concatenate(y_parts)
        prob = np.concatenate(p_parts)
        if len(np.unique(y)) < 2:
            continue
        values.append(float(roc_auc_score(y, prob)))
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise RuntimeError("No valid pair-cluster bootstrap replicates were produced.")
    lo, hi = np.quantile(arr, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi), arr


def run(paths: AdditionalAnalysisPaths, iterations: int = 2000, seed: int = 20260829) -> dict:
    out = paths.results_dir / "01_gse15852_paired"
    out.mkdir(parents=True, exist_ok=True)

    pred_path = paths.tables_dir / "external_GSE15852_predictions.csv"
    if not pred_path.is_file():
        raise FileNotFoundError(f"Required fixed prediction-probability file not found: {pred_path}")

    series_path, raw_hash_audit = acquire_reference_geo_series_matrix(
        paths.repo_dir, paths.tables_dir, "GSE15852", "GPL96",
        paths.raw_geo_dir / "GSE15852_series_matrix.txt.gz",
    )
    metadata = parse_series_matrix_sample_headers(series_path)
    metadata["Pair_ID"] = metadata["Sample_title"].map(_patient_code_from_title)
    metadata["Metadata_Label"] = metadata["Sample_title"].map(_label_from_title)

    if metadata["Pair_ID"].eq("").any():
        bad = metadata.loc[metadata["Pair_ID"].eq(""), ["GSM_ID", "Sample_title"]]
        raise RuntimeError("Could not derive pair IDs from some GSE15852 titles:\n" + bad.to_string(index=False))

    pair_audit = (
        metadata.groupby("Pair_ID", as_index=False)
        .agg(
            N_samples=("GSM_ID", "nunique"),
            N_normal=("Metadata_Label", lambda s: int((pd.Series(s) == 0).sum())),
            N_cancer=("Metadata_Label", lambda s: int((pd.Series(s) == 1).sum())),
        )
        .sort_values("Pair_ID")
    )
    if len(pair_audit) != 43 or not (
        pair_audit["N_samples"].eq(2)
        & pair_audit["N_normal"].eq(1)
        & pair_audit["N_cancer"].eq(1)
    ).all():
        raise RuntimeError(
            "GSE15852 pair audit failed; expected exactly 43 pairs with one normal and one cancer sample."
        )

    pred = pd.read_csv(pred_path)
    required = {"Sample_ID", "True_Label", "Probability", "Classifier"}
    missing = required - set(pred.columns)
    if missing:
        raise KeyError(f"GSE15852 predictions missing columns: {sorted(missing)}")
    pred["Sample_ID"] = pred["Sample_ID"].astype(str)
    merged = pred.merge(
        metadata[["GSM_ID", "Sample_title", "Pair_ID", "Metadata_Label"]],
        left_on="Sample_ID",
        right_on="GSM_ID",
        how="left",
        validate="many_to_one",
    )
    if merged["Pair_ID"].isna().any():
        raise RuntimeError("Some saved GSE15852 predictions could not be mapped to official GEO pair metadata.")
    if not (
        pd.to_numeric(merged["True_Label"]).astype(int)
        == pd.to_numeric(merged["Metadata_Label"]).astype(int)
    ).all():
        raise RuntimeError("Prediction labels disagree with GSE15852 sample titles.")

    rows = []
    bootstrap_long = []
    for classifier, block in merged.groupby("Classifier", sort=True):
        # Exactly one prediction per sample for each classifier after restricting to the frozen NIBFS panel.
        if "Feature_selection_method" in block.columns:
            nib = block["Feature_selection_method"].astype(str).str.casefold().eq("nibfs")
            block = block.loc[nib].copy()
        if "k" in block.columns:
            block = block.loc[pd.to_numeric(block["k"], errors="coerce").eq(20)].copy()
        block = block.drop_duplicates("Sample_ID")
        if len(block) != 86:
            raise RuntimeError(f"{classifier}: expected 86 unique samples, found {len(block)}")
        auc = float(roc_auc_score(block["True_Label"], block["Probability"]))
        lo, hi, dist = pair_cluster_auc_ci(
            block,
            iterations=iterations,
            seed=seed + sum(ord(c) for c in str(classifier)),
        )
        rows.append(
            {
                "Dataset": "GSE15852",
                "Pairs": 43,
                "Samples": 86,
                "Classifier": classifier,
                "ROC_AUC": auc,
                "Pair_cluster_bootstrap_iterations": iterations,
                "AUC_CI_low": lo,
                "AUC_CI_high": hi,
            }
        )
        bootstrap_long.extend(
            {"Classifier": classifier, "Bootstrap_iteration": i + 1, "ROC_AUC": float(v)}
            for i, v in enumerate(dist)
        )

    results = pd.DataFrame(rows).sort_values("Classifier")
    merged.to_csv(out / "GSE15852_predictions_with_official_pair_id.csv", index=False)
    metadata.to_csv(out / "GSE15852_official_pair_mapping.csv", index=False)
    pair_audit.to_csv(out / "GSE15852_pair_structure_audit.csv", index=False)
    results.to_csv(out / "GSE15852_pair_cluster_bootstrap_auc.csv", index=False)
    pd.DataFrame(bootstrap_long).to_csv(out / "GSE15852_pair_cluster_bootstrap_distribution.csv.gz", index=False, compression="gzip")

    summary = {
        "status": "PASS",
        "dataset": "GSE15852",
        "official_pairs": 43,
        "samples": 86,
        "point_predictions_reused": True,
        "models_refit": False,
        "feature_selection_rerun": False,
        "bootstrap_unit": "patient pair",
        "bootstrap_iterations": iterations,
        "raw_series_matrix_hash_audit": raw_hash_audit,
        "results": results.to_dict("records"),
    }
    write_json(summary, out / "GSE15852_paired_analysis_summary.json")
    return summary
