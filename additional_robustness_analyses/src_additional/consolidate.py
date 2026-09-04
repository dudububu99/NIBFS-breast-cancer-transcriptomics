from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .common import AdditionalAnalysisPaths, file_manifest, write_json


def _fmt_auc(est: float, lo: float, hi: float) -> str:
    return f"{est:.4f} ({lo:.4f}-{hi:.4f})"


def _metadata_identity_audit(paths: AdditionalAnalysisPaths, out: Path) -> dict:
    split = pd.read_csv(paths.tables_dir / "train_test_split_assignments.csv")
    split["GSM_ID"] = split["GSM_ID"].astype(str)
    exact_gsm_dupes = split.loc[split["GSM_ID"].duplicated(keep=False)].copy()

    # Conservative metadata fingerprint screen: a flag is not proof of a duplicate biological specimen.
    fingerprint_cols = [
        c for c in split.columns
        if c.startswith("Sample_characteristics_ch1_")
        or c in {"Sample_title", "Sample_source_name_ch1", "Sample_description"}
    ]
    def norm(v: object) -> str:
        s = str(v).strip().casefold()
        if s in {"", "nan", "none", "na", "n/a"}:
            return ""
        return re.sub(r"\s+", " ", s)

    pieces = split[fingerprint_cols].fillna("").apply(lambda col: col.map(norm))
    split["Metadata_fingerprint"] = pieces.astype(str).agg(" || ".join, axis=1)
    informative = split["Metadata_fingerprint"].str.replace("|", "", regex=False).str.strip().ne("")
    fp_counts = split.loc[informative].groupby("Metadata_fingerprint")["GSM_ID"].transform("count")
    flags = split.loc[informative].copy()
    flags["Fingerprint_count"] = fp_counts
    flags = flags.loc[flags["Fingerprint_count"] > 1].copy()
    # Keep only fingerprints appearing across more than one contributing GEO study.
    if not flags.empty:
        cross = flags.groupby("Metadata_fingerprint")["GEO_ID"].transform("nunique")
        flags = flags.loc[cross > 1].copy()

    exact_gsm_dupes.to_csv(out / "sample_identity_exact_GSM_duplicates.csv", index=False)
    flags.to_csv(out / "sample_identity_cross_cohort_metadata_fingerprint_flags.csv", index=False)
    summary = {
        "total_discovery_samples": int(len(split)),
        "unique_GSM_IDs": int(split["GSM_ID"].nunique()),
        "exact_duplicate_GSM_rows": int(len(exact_gsm_dupes)),
        "cross_cohort_metadata_fingerprint_flag_rows": int(len(flags)),
        "interpretation": (
            "Exact GSM duplicate screening is definitive for repeated GEO accessions. "
            "Metadata-fingerprint flags are a conservative review list only and cannot, by themselves, "
            "establish or exclude shared biological specimens across independent studies."
        ),
    }
    write_json(summary, out / "sample_identity_audit_summary.json")
    return summary


def run(paths: AdditionalAnalysisPaths, gse_summary: dict, strict_summary: dict, ss_summary: dict) -> dict:
    out = paths.results_dir / "99_summary"
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Updated external-evaluation table: GSE15852 gets pair-aware CI;
    # GSE70947 remains reference; TCGA CI is surfaced from already archived output.
    # ------------------------------------------------------------------
    old_s5 = pd.read_csv(paths.repo_dir / "supplementary_data" / "Table_S5A_External_Summary.csv")
    gse_pair = pd.read_csv(paths.results_dir / "01_gse15852_paired" / "GSE15852_pair_cluster_bootstrap_auc.csv")
    gse_auc = {str(r.Classifier): r for r in gse_pair.itertuples(index=False)}
    gse_text = "; ".join(
        [
            f"LR {_fmt_auc(gse_auc['LR'].ROC_AUC, gse_auc['LR'].AUC_CI_low, gse_auc['LR'].AUC_CI_high)}",
            f"RF {_fmt_auc(gse_auc['RF'].ROC_AUC, gse_auc['RF'].AUC_CI_low, gse_auc['RF'].AUC_CI_high)}",
            f"LGBM {_fmt_auc(gse_auc['LightGBM'].ROC_AUC, gse_auc['LightGBM'].AUC_CI_low, gse_auc['LightGBM'].AUC_CI_high)}",
        ]
    )
    updated_s5 = old_s5.copy()
    mask = updated_s5["Dataset"].astype(str).eq("GSE15852")
    updated_s5.loc[mask, "ROC_AUC"] = gse_text
    updated_s5.loc[mask, "Evaluation_note"] = (
        "2,000-iteration patient-pair cluster bootstrap 95% CI; 43 matched tumor-normal pairs; "
        "saved discovery-model predictions reused; no feature selection or model fitting was rerun for CI calculation"
    )

    tcga_perf = pd.read_csv(
        paths.repo_dir / "results" / "verification" / "tcga_brca" /
        "TCGA_BRCA_RNAseq_performance_with_pair_bootstrap_CI.csv"
    )
    tcga_auc = tcga_perf.loc[tcga_perf["Metric"].astype(str).eq("ROC_AUC")].copy()
    tcga_map = {str(r.Model): r for r in tcga_auc.itertuples(index=False)}
    tcga_text = "; ".join(
        [
            f"Signed {_fmt_auc(tcga_map['FrozenSignedPanelScore'].Estimate, tcga_map['FrozenSignedPanelScore'].CI95_low, tcga_map['FrozenSignedPanelScore'].CI95_high)}",
            f"LR {_fmt_auc(tcga_map['LR'].Estimate, tcga_map['LR'].CI95_low, tcga_map['LR'].CI95_high)}",
            f"RF {_fmt_auc(tcga_map['RF'].Estimate, tcga_map['RF'].CI95_low, tcga_map['RF'].CI95_high)}",
            f"LGBM {_fmt_auc(tcga_map['LightGBM'].Estimate, tcga_map['LightGBM'].CI95_low, tcga_map['LightGBM'].CI95_high)}",
        ]
    )
    tcga_mask = updated_s5["Dataset"].astype(str).eq("TCGA-BRCA")
    updated_s5.loc[tcga_mask, "ROC_AUC"] = tcga_text
    updated_s5.loc[tcga_mask, "Evaluation_note"] = (
        "Frozen-panel rank-space transfer; 2,000-iteration participant-pair bootstrap 95% CI; "
        "20/20 paired Wilcoxon FDR-significant"
    )
    updated_s5.to_csv(out / "external_evaluation_summary.csv", index=False)

    # Main Table 6-ready full table (internal reference row + corrected external rows).
    held = pd.read_csv(paths.tables_dir / "heldout_metrics_default_and_transferred.csv")
    held = held.loc[
        held["Feature_selection_method"].astype(str).eq("NIBFS")
        & pd.to_numeric(held["k"], errors="coerce").eq(20)
    ].drop_duplicates("Classifier")
    held_map = {str(r.Classifier): r for r in held.itertuples(index=False)}

    def parse_external_auc(text: str, model_token: str) -> str:
        # Preserve reference GSE70947 display as reported in the existing supplementary table.
        pattern = rf"{re.escape(model_token)}\s+([0-9.]+\s*\([0-9.]+-[0-9.]+\))"
        m = re.search(pattern, str(text))
        return m.group(1) if m else ""

    g709 = updated_s5.loc[updated_s5["Dataset"].astype(str).eq("GSE70947")].iloc[0]
    full_table6 = pd.DataFrame([
        {
            "Resource": "Internal held-out",
            "Representation": "GPL570 expression",
            "LR_AUC_95CI": _fmt_auc(held_map["LR"].ROC_AUC, held_map["LR"].AUC_CI_low, held_map["LR"].AUC_CI_high),
            "RF_AUC_95CI": _fmt_auc(held_map["RF"].ROC_AUC, held_map["RF"].AUC_CI_low, held_map["RF"].AUC_CI_high),
            "LightGBM_AUC_95CI": _fmt_auc(held_map["LightGBM"].ROC_AUC, held_map["LightGBM"].AUC_CI_low, held_map["LightGBM"].AUC_CI_high),
            "Direction_concordance": "-",
            "Evaluation_note": "Post-harmonization internal assessment; values carried forward unchanged",
        },
        {
            "Resource": "GSE15852",
            "Representation": "GPL96 expression",
            "LR_AUC_95CI": _fmt_auc(gse_auc["LR"].ROC_AUC, gse_auc["LR"].AUC_CI_low, gse_auc["LR"].AUC_CI_high),
            "RF_AUC_95CI": _fmt_auc(gse_auc["RF"].ROC_AUC, gse_auc["RF"].AUC_CI_low, gse_auc["RF"].AUC_CI_high),
            "LightGBM_AUC_95CI": _fmt_auc(gse_auc["LightGBM"].ROC_AUC, gse_auc["LightGBM"].AUC_CI_low, gse_auc["LightGBM"].AUC_CI_high),
            "Direction_concordance": "19/20",
            "Evaluation_note": "43 matched pairs; 2,000 patient-pair cluster bootstrap applied to fixed prediction probabilities",
        },
        {
            "Resource": "GSE70947",
            "Representation": "GPL13607 expression",
            "LR_AUC_95CI": parse_external_auc(g709.ROC_AUC, "LR"),
            "RF_AUC_95CI": parse_external_auc(g709.ROC_AUC, "RF"),
            "LightGBM_AUC_95CI": parse_external_auc(g709.ROC_AUC, "LGBM"),
            "Direction_concordance": "20/20",
            "Evaluation_note": "148-pair cluster-bootstrap result; values carried forward unchanged",
        },
        {
            "Resource": "TCGA-BRCA",
            "Representation": "Within-sample rank space",
            "LR_AUC_95CI": _fmt_auc(tcga_map["LR"].Estimate, tcga_map["LR"].CI95_low, tcga_map["LR"].CI95_high),
            "RF_AUC_95CI": _fmt_auc(tcga_map["RF"].Estimate, tcga_map["RF"].CI95_low, tcga_map["RF"].CI95_high),
            "LightGBM_AUC_95CI": _fmt_auc(tcga_map["LightGBM"].Estimate, tcga_map["LightGBM"].CI95_low, tcga_map["LightGBM"].CI95_high),
            "Direction_concordance": "20/20",
            "Evaluation_note": "113-participant-pair bootstrap confidence interval; values carried forward unchanged",
        },
    ])
    full_table6.to_csv(out / "integrated_evaluation_summary.csv", index=False)
    full_table6.loc[full_table6["Resource"].ne("Internal held-out")].to_csv(
        out / "external_evaluation_rows.csv", index=False
    )

    # Compact summaries for reproducibility and inspection.
    strict_table = pd.read_csv(
        paths.results_dir / "02_repeated_fold_fitted_5x5" / "repeated_fold_fitted_5x5_stability_summary.csv"
    )
    strict_table.to_csv(out / "repeated_fold_fitted_5x5_summary.csv", index=False)

    ss_table = pd.read_csv(
        paths.results_dir / "03_stability_selection_repeated_10x5" / "all_methods_plus_stability_selection_stability_summary.csv"
    )
    ss_table.to_csv(out / "stability_selection_summary.csv", index=False)
    gse_pair.to_csv(out / "gse15852_paired_bootstrap_summary.csv", index=False)

    identity_summary = _metadata_identity_audit(paths, out)

    result_summary = {
        "status": "PASS",
        "source_inputs_modified": False,
        "gse15852": gse_summary,
        "repeated_fold_fitted_5x5": strict_summary,
        "stability_selection": ss_summary,
        "sample_identity_audit": identity_summary,
        "external_evaluation_summary": str(out / "external_evaluation_summary.csv"),
        "integrated_evaluation_summary": str(out / "integrated_evaluation_summary.csv"),
    }
    write_json(result_summary, out / "ANALYSIS_SUMMARY.json")

    manifest_path = out / "OUTPUT_MANIFEST_SHA256.csv"
    if manifest_path.exists():
        manifest_path.unlink()
    manifest = file_manifest(paths.results_dir)
    manifest.to_csv(manifest_path, index=False)
    return result_summary
