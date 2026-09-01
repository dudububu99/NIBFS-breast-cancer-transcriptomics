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
    out = paths.results_dir / "99_submission_ready_summary"
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
    updated_s5.to_csv(out / "UPDATED_Table_S5A_External_Summary.csv", index=False)

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
    full_table6.to_csv(out / "UPDATED_Main_Table6_COMPLETE.csv", index=False)
    full_table6.loc[full_table6["Resource"].ne("Internal held-out")].to_csv(
        out / "UPDATED_Main_Table6_external_rows.csv", index=False
    )

    # Candidate supplementary tables for later manuscript integration (do not overwrite S1-S9 yet).
    strict_table = pd.read_csv(
        paths.results_dir / "02_repeated_fold_fitted_5x5" / "repeated_fold_fitted_5x5_stability_summary.csv"
    )
    strict_table.to_csv(out / "CANDIDATE_Supplementary_Table_Repeated_FoldFitted_5x5.csv", index=False)
    ss_table = pd.read_csv(
        paths.results_dir / "03_stability_selection_repeated_10x5" / "all_methods_plus_stability_selection_stability_summary.csv"
    )
    ss_table.to_csv(out / "CANDIDATE_Supplementary_Table_StabilitySelection.csv", index=False)
    gse_pair.to_csv(out / "CANDIDATE_Supplementary_Table_GSE15852_PairedBootstrap.csv", index=False)

    identity_summary = _metadata_identity_audit(paths, out)

    # ------------------------------------------------------------------
    # Result-aware manuscript patch guide.
    # ------------------------------------------------------------------
    strict = pd.DataFrame(strict_summary.get("stability_summary", []))
    ss = pd.DataFrame(ss_summary.get("stability_summary", []))
    def row_for(frame: pd.DataFrame, method: str) -> dict:
        if frame.empty or "Method" not in frame.columns:
            return {}
        match = frame.loc[frame["Method"].astype(str).eq(method)]
        return {} if match.empty else match.iloc[0].to_dict()

    strict_n = row_for(strict, "NIBFS")
    strict_deg = row_for(strict, "DEG-only")
    strict_m = row_for(strict, "mRMR")
    strict_l = row_for(strict, "LASSO")
    ss_n = row_for(ss, "NIBFS")
    ss_s = row_for(ss, "Stability-selection")

    text = f"""# CBC additional analysis: manuscript update guide

This file is generated from the additional analysis outputs. It is intentionally separate from the reference v1.1.0 archive. Do not overwrite the old result folders.

## 1. GSE15852 paired correction
GSE15852 contains 43 matched tumor-normal pairs (86 samples). The previously saved NIBFS model probabilities were reused without rerunning feature selection or fitting models during the bootstrap calculation. Pair-cluster bootstrap (2,000 replicates) should replace the earlier class-stratified confidence intervals.

Updated ROC-AUC results:
{gse_text}

Suggested Methods sentence:
"GSE15852 comprises 43 matched tumor-normal pairs. Confidence intervals for ROC-AUC were therefore recomputed using a 2,000-iteration patient-pair cluster bootstrap while retaining the previously frozen panel and saved discovery-model predictions."

## 2. Repeated strict training-fold-fitted sensitivity analysis
Design: 5 repeats x 5 folds. The first partition matches the separately reported strict 1x5 sensitivity analysis; four additional partitions use random states 43-46 under the same preprocessing protocol.

NIBFS mean repeat-level Jaccard: {strict_n.get('Mean_Jaccard', float('nan')):.4f} ± {strict_n.get('SD_between_repeats', float('nan')):.4f}
DEG-only: {strict_deg.get('Mean_Jaccard', float('nan')):.4f} ± {strict_deg.get('SD_between_repeats', float('nan')):.4f}
mRMR: {strict_m.get('Mean_Jaccard', float('nan')):.4f} ± {strict_m.get('SD_between_repeats', float('nan')):.4f}
LASSO: {strict_l.get('Mean_Jaccard', float('nan')):.4f} ± {strict_l.get('SD_between_repeats', float('nan')):.4f}
NIBFS Nogueira across 25 panels: {strict_n.get('Nogueira_stability', float('nan')):.4f}

Suggested Results framing:
"The stricter training-fold-fitted analysis was extended to five repeated five-fold allocations. The ordering of panel stability remained [describe from table], indicating that the earlier result was not specific to a single fold allocation."

Use the exact values in `repeated_fold_fitted_5x5_stability_summary.csv`; do not claim statistical significance unless supported by `repeated_fold_fitted_5x5_repeat_level_inference.csv`.

## 3. Empirical stability-selection comparator
The new comparator is a screened L1-logistic subsampling stability-selection procedure evaluated under the existing the same 10 repeated five-fold evaluation partitions. The comparison uses the corresponding NIBFS/DEG/mRMR/LASSO repeated results from the same evaluation partitions.

NIBFS repeated mean Jaccard: {ss_n.get('Mean_Jaccard', float('nan')):.4f}
Stability-selection repeated mean Jaccard: {ss_s.get('Mean_Jaccard', float('nan')):.4f}
NIBFS Nogueira: {ss_n.get('Nogueira_stability', float('nan')):.4f}
Stability-selection Nogueira: {ss_s.get('Nogueira_stability', float('nan')):.4f}
Stability-selection LR OOF ROC-AUC: {ss_summary.get('new_comparator_lr_oof_auc_mean', float('nan')):.4f} ± {ss_summary.get('new_comparator_lr_oof_auc_sd', float('nan')):.4f}
Archived NIBFS LR OOF ROC-AUC: {ss_summary.get('archived_nibfs_lr_oof_auc_mean', float('nan')):.4f} ± {ss_summary.get('archived_nibfs_lr_oof_auc_sd', float('nan')):.4f}

Recommended terminology:
"screened L1-logistic stability-selection comparator with fixed-k top-20 reporting"

Do not describe the thresholded pi>=0.90 set as formal error-controlled stability selection. The top-20 comparison is based on selection probability, then mean absolute coefficient, under a prespecified 50 half-sample resampling scheme and fold-local 1,000-gene screen.

## 4. TCGA-BRCA confidence intervals
No new TCGA training or validation run was required. The reference archive already contains participant-pair bootstrap confidence intervals. Updated ROC-AUC display:
{tcga_text}

## 5. Sample identity audit
Total discovery rows: {identity_summary['total_discovery_samples']}
Unique GEO sample accessions: {identity_summary['unique_GSM_IDs']}
Exact duplicate GSM rows: {identity_summary['exact_duplicate_GSM_rows']}
Cross-cohort metadata-fingerprint flag rows: {identity_summary['cross_cohort_metadata_fingerprint_flag_rows']}

Use cautious wording: the audit establishes whether GEO sample accession IDs are duplicated. Public metadata alone cannot definitively prove or exclude shared biological specimens across studies.

## 6. What remains unchanged
Do NOT rerun or replace the reference primary 5-fold analysis, repeated 10x5 NIBFS/DEG/mRMR/LASSO analysis, random-anchor 1,000-control experiment, degree-preserving rewiring, LOCO, frozen top-20 panel, GSE70947 evaluation, or original TCGA model predictions. Those remain reference v1.1.0 results.
"""
    (out / "MANUSCRIPT_UPDATE_GUIDE.md").write_text(text, encoding="utf-8")

    # Final manifest and compact machine-readable summary.
    result_summary = {
        "status": "PASS",
        "old_results_modified": False,
        "gse15852": gse_summary,
        "repeated_fold_fitted_5x5": strict_summary,
        "stability_selection": ss_summary,
        "sample_identity_audit": identity_summary,
        "updated_external_table": str(out / "UPDATED_Table_S5A_External_Summary.csv"),
        "manuscript_update_guide": str(out / "MANUSCRIPT_UPDATE_GUIDE.md"),
    }
    write_json(result_summary, out / "CBC_ADDITIONAL ANALYSIS_MASTER_SUMMARY.json")
    manifest = file_manifest(paths.results_dir)
    manifest.to_csv(out / "CBC_ADDITIONAL ANALYSIS_OUTPUT_MANIFEST_SHA256.csv", index=False)
    return result_summary
