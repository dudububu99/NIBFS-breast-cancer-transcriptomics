from __future__ import annotations
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
import ast
import gzip
import hashlib
import io
import json
import os
import platform
import re
import shutil
import urllib.error
import urllib.request
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import networkx as nx

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

from .download_utils import (
    HGNC_COMPLETE_SET_URL, download_first_available, download_if_missing,
    geo_annotation_url, geo_series_urls, string_reference_urls,
)
from .hgnc import HGNCResolver
from .geo_io import (
    create_sample_metadata, label_discovery_sample, label_external_samples,
    parse_series_matrix, read_geo_annotation,
)
from .preprocessing import (
    binary_labels, combat_harmonize, conditional_log2_probe_table,
    filter_bottom_variance, highest_variance_probe_per_gene,
    quantile_normalize_samples,
)
from .quality_control import (
    expression_distribution_summary, missing_value_summary, pca_table,
    sample_correlation_audit,
)
from .ppi import build_gene_edges_and_degree, read_string_protein_to_gene
from .feature_selection import (
    lasso_gene_ranking, nibfs_rank, prepare_ppi_rank_table, run_limma_rpy2,
    select_mrmr_features, select_top_k,
)
from .reporting import copy_file, export_expression_matrix
from .figures import (
    plot_correlation_heatmap, plot_expression_distributions, plot_missing_values,
    plot_outlier_audit, plot_pca_before_after, plot_preprocessing_qc_overview,
)

def dirs(project: Path) -> dict[str, Path]:
    d = {
        "project": project,
        "geo": project / "data/raw/geo",
        "ref": project / "data/reference",
        "tables": project / "results/tables",
        "figures": project / "results/figures",
        "supp": project / "supplementary",
        "manual": project / "manual_inputs",
        "models": project / "results/models",
        "logs": project / "results/logs",
        "assets": project / "manuscript_assets",
    }
    for path in d.values():
        path.mkdir(parents=True, exist_ok=True)
    return d

def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def references(project: Path, cfg: dict, d: dict[str, Path]):
    hgnc = download_if_missing(HGNC_COMPLETE_SET_URL, d["ref"] / "hgnc_complete_set.txt")
    urls = string_reference_urls(int(cfg["ppi"]["species"]), str(cfg["ppi"]["string_version"]))
    info = download_if_missing(urls["protein_info"], d["ref"] / Path(urls["protein_info"]).name)
    links = download_if_missing(urls["protein_links"], d["ref"] / Path(urls["protein_links"]).name)
    return hgnc, info, links

def _numeric_scale_row(gse: str, numeric: pd.DataFrame, threshold: float) -> dict:
    values = numeric.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    maximum = float(np.max(finite)) if finite.size else np.nan
    return {
        "GEO_ID": gse,
        "N_values": int(finite.size),
        "Minimum": float(np.min(finite)) if finite.size else np.nan,
        "Q1": float(np.quantile(finite, 0.25)) if finite.size else np.nan,
        "Median": float(np.median(finite)) if finite.size else np.nan,
        "Q3": float(np.quantile(finite, 0.75)) if finite.size else np.nan,
        "Maximum": maximum,
        "Log2_threshold": float(threshold),
        "Log2_required": bool(maximum > threshold) if np.isfinite(maximum) else False,
        "Decision": "Apply log2(x+1)" if np.isfinite(maximum) and maximum > threshold else "Already approximately log2 scale",
    }

def prepare_discovery(project: Path, cfg: dict, d: dict[str, Path], resolver: HGNCResolver):
    accession = pd.read_csv(project / "data_accession_list.csv")
    discovery = accession[accession.Role.str.startswith("Discovery")].copy()
    discovery.to_csv(d["supp"] / "Table_S1_discovery_cohorts.csv", index=False)
    discovery.to_csv(d["tables"] / "dataset_composition_table.csv", index=False)

    annotation = download_if_missing(geo_annotation_url("GPL570"), d["geo"] / "GPL570.annot.gz")
    probe_map = read_geo_annotation(annotation, resolver)
    probe_map.to_csv(d["tables"] / "GPL570_probe_to_HGNC_mapping.csv", index=False)

    matrices: list[pd.DataFrame] = []
    metas: list[pd.DataFrame] = []
    cohort_qc: list[dict] = []
    scale_rows: list[dict] = []
    metadata_rows: list[pd.DataFrame] = []

    for row in discovery.itertuples(index=False):
        gse = row.GEO_ID
        series = download_first_available(geo_series_urls(gse, row.Platform), d["geo"] / f"{gse}_series_matrix.txt.gz")
        expr, md = parse_series_matrix(series)
        meta = create_sample_metadata(md)
        meta["GEO_ID"] = gse
        meta["Label"] = meta.apply(lambda r: label_discovery_sample(gse, r), axis=1)
        meta = meta[meta.Label.isin(["Cancer", "Normal"])].copy()
        ids = [sample for sample in meta.GSM_ID if sample in expr.columns]
        expr = expr[["ID_REF"] + ids]
        raw_numeric = expr[ids].apply(pd.to_numeric, errors="coerce")
        scale_rows.append(_numeric_scale_row(gse, raw_numeric, float(cfg["preprocessing"]["log2_threshold"])))
        raw_missing = int(raw_numeric.isna().sum().sum())
        raw_probes = len(expr)
        expr, was_log = conditional_log2_probe_table(expr, float(cfg["preprocessing"]["log2_threshold"]))
        X = highest_variance_probe_per_gene(expr, probe_map).loc[ids]
        matrices.append(X)
        cohort_meta = meta.set_index("GSM_ID").loc[ids].reset_index()
        metas.append(cohort_meta)
        metadata_rows.append(cohort_meta)
        cohort_qc.append(
            {
                "GEO_ID": gse,
                "Series_matrix_path": str(series),
                "Raw_probe_sets": raw_probes,
                "Samples": len(ids),
                "Cancer": int((cohort_meta.Label == "Cancer").sum()),
                "Normal": int((cohort_meta.Label == "Normal").sum()),
                "Genes_after_probe_mapping": X.shape[1],
                "Log2_applied": was_log,
                "Raw_missing_values": raw_missing,
            }
        )

    common = sorted(set.intersection(*(set(matrix.columns) for matrix in matrices)))
    X_joint = pd.concat([matrix[common] for matrix in matrices])
    metadata = pd.concat(metas, ignore_index=True).set_index("GSM_ID").loc[X_joint.index].reset_index()
    y = binary_labels(metadata.Label)

    before, ev_before = pca_table(X_joint, metadata, "Before", int(cfg["project"]["random_state"]))
    dist_before = expression_distribution_summary(X_joint, metadata, "Before")

    X_qn = quantile_normalize_samples(X_joint)
    X_combat, estimates = combat_harmonize(
        X_qn,
        metadata.GEO_ID,
        y,
        bool(cfg["preprocessing"]["combat_preserve_class"]),
    )
    after, ev_after = pca_table(X_combat, metadata, "After", int(cfg["project"]["random_state"]))
    dist_after = expression_distribution_summary(X_combat, metadata, "After")

    X_final, cutoff = filter_bottom_variance(
        X_combat, float(cfg["preprocessing"]["variance_bottom_fraction"])
    )
    final_pca, ev_final = pca_table(X_final, metadata, "Final", int(cfg["project"]["random_state"]))
    dist_final = expression_distribution_summary(X_final, metadata, "Final")
    qc, corr = sample_correlation_audit(
        X_final,
        metadata,
        final_pca,
        float(cfg["quality_control"]["robust_z_threshold"]),
    )

    missing = missing_value_summary(
        {
            "Joint_common_gene_matrix": X_joint,
            "Quantile_normalized": X_qn,
            "ComBat_harmonized": X_combat,
            "Variance_filtered": X_final,
        }
    )
    stages = pd.DataFrame(
        [
            {
                "Stage": "Per-cohort probe mapping and common-gene intersection",
                "Samples": X_joint.shape[0],
                "Genes": X_joint.shape[1],
            },
            {"Stage": "Joint quantile normalization", "Samples": X_qn.shape[0], "Genes": X_qn.shape[1]},
            {"Stage": "ComBat harmonization", "Samples": X_combat.shape[0], "Genes": X_combat.shape[1]},
            {"Stage": "Bottom-10% variance filter", "Samples": X_final.shape[0], "Genes": X_final.shape[1]},
        ]
    )

    cohort_qc_df = pd.DataFrame(cohort_qc)
    scale_df = pd.DataFrame(scale_rows)
    all_dist = pd.concat([dist_before, dist_after, dist_final], ignore_index=True)
    variance = pd.concat([ev_before, ev_after, ev_final], ignore_index=True)

    cohort_qc_df.to_csv(d["tables"] / "discovery_cohort_QC.csv", index=False)
    cohort_qc_df.to_csv(d["tables"] / "geo_parse_summary.csv", index=False)
    scale_df.to_csv(d["tables"] / "expression_scale_check_before_mapping.csv", index=False)
    scale_df[["GEO_ID", "Maximum", "Log2_threshold", "Log2_required", "Decision"]].to_csv(
        d["tables"] / "expression_scale_decision_summary.csv", index=False
    )
    stages.to_csv(d["tables"] / "preprocessing_stage_dimensions.csv", index=False)
    stages.to_csv(d["supp"] / "Table_S2_preprocessing_stage_dimensions.csv", index=False)
    stages.to_csv(d["tables"] / "gene_count_summary.csv", index=False)
    all_dist.to_csv(d["tables"] / "sample_expression_distribution_summary.csv", index=False)
    all_dist.to_csv(d["tables"] / "sample_distribution_summary_all_stages.csv", index=False)
    dist_before.to_csv(d["tables"] / "sample_distribution_before_harmonization.csv", index=False)
    dist_after.to_csv(d["tables"] / "sample_distribution_after_harmonization.csv", index=False)
    dist_final.to_csv(d["tables"] / "sample_distribution_final_harmonized.csv", index=False)
    missing.to_csv(d["tables"] / "missing_value_summary.csv", index=False)
    missing.to_csv(d["tables"] / "missing_value_and_integrity_summary.csv", index=False)
    qc.to_csv(d["tables"] / "sample_QC_audit_metrics.csv", index=False)
    qc.to_csv(d["tables"] / "sample_outlier_audit_final_harmonized.csv", index=False)
    qc.to_csv(d["tables"] / "pca_outlier_audit_final_harmonized.csv", index=False)
    qc[qc.Audit_flag].to_csv(d["tables"] / "flagged_samples_for_audit.csv", index=False)
    qc[qc.Distribution_flag].to_csv(d["tables"] / "flagged_samples_distribution_outliers.csv", index=False)
    qc_decision = qc.copy()
    qc_decision["Decision"] = np.where(qc_decision.Audit_flag, "Flagged for audit; retained unless technical inconsistency is confirmed", "Retained")
    qc_decision["Removed"] = False
    qc_decision.to_csv(d["tables"] / "qc_flagged_samples_decision_table.csv", index=False)
    pd.DataFrame(
        [
            {
                "Total_samples": len(qc),
                "Distribution_flags": int(qc.Distribution_flag.sum()),
                "PCA_flags": int(qc.PCA_flag.sum()),
                "Low_correlation_flags": int(qc.Low_correlation_flag.sum()),
                "Combined_audit_flags": int(qc.Audit_flag.sum()),
                "Samples_removed": 0,
                "Policy": "No automatic removal; retain unless technical inconsistency is established",
            }
        ]
    ).to_csv(d["tables"] / "qc_flagged_samples_decision_summary.csv", index=False)
    qc.groupby(["GEO_ID", "Label"], as_index=False).agg(
        Samples=("GSM_ID", "count"), Audit_flags=("Audit_flag", "sum")
    ).to_csv(d["tables"] / "sample_outlier_summary_final_harmonized.csv", index=False)
    qc.groupby(["GEO_ID", "Label"], as_index=False).agg(
        Samples=("GSM_ID", "count"), PCA_flags=("PCA_flag", "sum")
    ).to_csv(d["tables"] / "pca_outlier_summary_final_harmonized.csv", index=False)
    corr.to_csv(d["tables"] / "sample_correlation_matrix.csv.gz", compression="gzip")
    variance.to_csv(d["tables"] / "PCA_variance_explained.csv", index=False)
    variance.to_csv(d["tables"] / "pca_variance_summary_before_after_harmonization.csv", index=False)
    before.to_csv(d["tables"] / "PCA_before_coordinates.csv", index=False)
    after.to_csv(d["tables"] / "PCA_after_coordinates.csv", index=False)
    final_pca.to_csv(d["tables"] / "PCA_final_coordinates.csv", index=False)
    pd.crosstab(metadata.GEO_ID, metadata.Label, margins=True).to_csv(d["tables"] / "batch_class_crosstab.csv")
    pd.DataFrame({"Gene": common}).to_csv(d["tables"] / "common_gene_universe_before_variance_filter.csv", index=False)
    pd.DataFrame({"Gene": X_final.columns}).to_csv(d["tables"] / "eligible_gene_universe.csv", index=False)
    metadata.to_csv(d["tables"] / "discovery_sample_metadata.csv", index=False)
    metadata.to_csv(d["tables"] / "harmonized_metadata.csv", index=False)
    metadata.to_csv(d["tables"] / "integrated_sample_metadata_with_labels.csv", index=False)
    metadata.to_csv(d["tables"] / "sample_metadata_ready_by_dataset.csv", index=False)
    export_expression_matrix(X_final, d["tables"] / "harmonized_expression_matrix.csv")
    X_final.iloc[:10, :10].to_csv(d["tables"] / "final_expression_matrix_preview_10x10.csv", index=True, index_label="GSM_ID")

    plot_pca_before_after(before, after, pd.concat([ev_before, ev_after]), d["figures"] / "Figure_2_PCA_before_after_harmonization.png")
    plot_expression_distributions(all_dist, d["figures"] / "Figure_S1_expression_distributions.png")
    plot_missing_values(missing, d["figures"] / "Figure_S2_missing_value_audit.png")
    plot_correlation_heatmap(corr, metadata, d["figures"] / "Figure_S3_sample_correlation_heatmap.png")
    plot_outlier_audit(qc, d["figures"] / "Figure_S4_sample_outlier_audit.png")
    plot_preprocessing_qc_overview(all_dist, missing, pd.concat([ev_before, ev_after]), qc, d["figures"] / "preprocessing_qc_overview.png")

    prep = {
        "common_genes": len(common),
        "eligible_genes": X_final.shape[1],
        "variance_cutoff": cutoff,
        "flagged_samples": int(qc.Audit_flag.sum()),
        "combat_estimates_keys": sorted(estimates.keys()),
    }
    artifacts = {
        "X_joint": X_joint,
        "X_qn": X_qn,
        "X_combat": X_combat,
        "dist": all_dist,
        "missing": missing,
        "variance": variance,
        "qc": qc,
    }
    return X_final, y, metadata, prep, artifacts

def prepare_ppi(cfg: dict, d: dict[str, Path], resolver: HGNCResolver, info: Path, links: Path, genes):
    edge_path = d["tables"] / "STRING_gene_edges_score700.csv.gz"
    degree_path = d["tables"] / "ppi_degree_table.csv"
    if edge_path.exists() and degree_path.exists():
        return pd.read_csv(edge_path), pd.read_csv(degree_path)
    mapping = read_string_protein_to_gene(info, resolver)
    mapping.to_csv(d["tables"] / "STRING_preferred_name_HGNC_mapping.csv", index=False)
    edges, degree = build_gene_edges_and_degree(
        links,
        mapping,
        genes,
        int(cfg["ppi"]["required_score"]),
        int(cfg["ppi"]["chunk_size"]),
    )
    edges.to_csv(edge_path, index=False, compression="gzip")
    degree.to_csv(degree_path, index=False)
    return edges, degree

def rankings(X: pd.DataFrame, y, degree: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    limma = run_limma_rpy2(X, y)
    ppi = prepare_ppi_rank_table(degree, X.columns)
    nib = nibfs_rank(limma, ppi, X.columns)
    max_k = max(map(int, cfg["project"]["sensitivity_k"]))
    fs = cfg["feature_selection"]
    mrmr = select_mrmr_features(
        X,
        y,
        max_k,
        int(fs["mrmr_candidate_size"]),
        int(cfg["project"]["random_state"]),
    )
    lasso = lasso_gene_ranking(
        X,
        y,
        float(fs["lasso_C"]),
        int(cfg["project"]["random_state"]),
        str(fs["lasso_solver"]),
        int(fs["lasso_max_iter"]),
    )
    return {"NIBFS": nib, "DEG-only": limma, "mRMR": mrmr, "LASSO": lasso}

def panels(ranking_tables: dict[str, pd.DataFrame], k: int) -> dict[str, list[str]]:
    rank_columns = {
        "NIBFS": "Rank_NIBFS",
        "DEG-only": "Rank_stat",
        "mRMR": "Selection_Order",
        "LASSO": "Rank_LASSO",
    }
    return {method: select_top_k(table, k, rank_columns[method]) for method, table in ranking_tables.items()}

def load_external(cfg: dict, d: dict[str, Path], resolver: HGNCResolver):
    gse = cfg["external_validation"]["gse"]
    gpl = cfg["external_validation"]["platform"]
    series = download_first_available(geo_series_urls(gse, gpl), d["geo"] / f"{gse}_series_matrix.txt.gz")
    annotation = download_if_missing(geo_annotation_url(gpl), d["geo"] / f"{gpl}.annot.gz")
    expr, md = parse_series_matrix(series)
    meta = label_external_samples(create_sample_metadata(md), gse)
    ids = [sample for sample in meta.GSM_ID if sample in expr.columns]
    expr = expr[["ID_REF"] + ids]
    expr, _ = conditional_log2_probe_table(expr, float(cfg["preprocessing"]["log2_threshold"]))
    probe_map = read_geo_annotation(annotation, resolver)
    X = highest_variance_probe_per_gene(expr, probe_map).loc[ids]
    X = quantile_normalize_samples(X)
    meta = meta.set_index("GSM_ID").loc[ids].reset_index()
    return X, meta.Label_binary.to_numpy(int), meta

def copy_table(source: str | Path, destination: str | Path) -> None:
    copy_file(source, destination)

def environment_table() -> pd.DataFrame:
    import importlib.metadata as im

    packages = [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "lightgbm",
        "matplotlib",
        "networkx",
        "rpy2",
        "neuroCombat",
        "gprofiler-official",
    ]
    rows = []
    for package in packages:
        try:
            version = im.version(package)
        except Exception:
            version = "not installed"
        rows.append({"Package": package, "Version": version})
    return pd.DataFrame(rows)

def _long_panels(fold: int, k: int, selected: dict[str, list[str]]) -> list[dict]:
    rows = []
    for method, genes in selected.items():
        rows.extend(
            {"Fold": fold, "Method": method, "k": k, "Selection_rank": rank, "Gene": gene}
            for rank, gene in enumerate(genes, 1)
        )
    return rows

def _full_panels(final_panels: dict[int, dict[str, list[str]]]) -> pd.DataFrame:
    rows = []
    for k, method_panels in final_panels.items():
        for method, genes in method_panels.items():
            rows.extend(
                {"Method": method, "k": k, "Selection_rank": rank, "Gene": gene}
                for rank, gene in enumerate(genes, 1)
            )
    return pd.DataFrame(rows)

def _source_subset(enrichment: pd.DataFrame, token: str) -> pd.DataFrame:
    if enrichment.empty:
        return enrichment.copy()
    return enrichment[enrichment.Database.astype(str).str.upper().str.contains(token.upper(), regex=False)].copy()
