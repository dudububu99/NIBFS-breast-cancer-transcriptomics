# ============================================================
# FULL TRANSFER-SAFE LOCO — GENE-LEVEL FIXED VERSION
#
# Run from the existing executed notebook with:
#   %run -i src/full_transfer_safe_loco_GENELEVEL_V2.py
#
# Required active notebook objects:
#   all_expression, all_metadata, probe_map, string_edges
#
# This script DOES NOT rerun the main notebook analysis. It performs only
# a standalone leave-one-cohort-out sensitivity analysis.
#
# Key safeguards in every LOCO iteration:
#   1. Held-out cohort is excluded from representative-probe selection.
#   2. Probe IDs are mapped to HGNC gene symbols before NIBFS.
#   3. Quantile reference, imputation, variance filtering, limma,
#      PPI degree, NIBFS selection, scaling, and model fitting use
#      held-out-excluded training cohorts only.
#   4. Standard ComBat is not used because a previously unseen batch
#      cannot be corrected with training-batch parameters in the usual
#      ComBat formulation.
#   5. Decision threshold is fixed at 0.5.
# ============================================================

from pathlib import Path
from time import perf_counter
import gc
import json
import math
import re
import zlib
import warnings

import numpy as np
import pandas as pd
import yaml

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    matthews_corrcoef,
    brier_score_loss,
    confusion_matrix,
)

try:
    from lightgbm import LGBMClassifier
except Exception as exc:
    raise ImportError(
        "LightGBM tidak tersedia. Jalankan sel instalasi notebook, "
        "kemudian jalankan kembali LOCO gene-level."
    ) from exc


# ------------------------------------------------------------
# 0. Configuration and active-object checks
# ------------------------------------------------------------
REQUIRED_OBJECTS = [
    "all_expression",
    "all_metadata",
    "probe_map",
    "string_edges",
]

missing = [name for name in REQUIRED_OBJECTS if name not in globals()]
if missing:
    raise RuntimeError(
        "Objek aktif berikut belum ditemukan: " + ", ".join(missing) + ". "
        "Jangan ulang seluruh notebook; jalankan hanya sel data/reference "
        "loading sampai objek tersebut aktif."
    )

PROJECT_DIR = Path.cwd()
CONFIG_PATH = PROJECT_DIR / "config.yaml"
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"config.yaml tidak ditemukan di {PROJECT_DIR}")

with CONFIG_PATH.open("r", encoding="utf-8") as handle:
    CFG = yaml.safe_load(handle)

RANDOM_STATE = int(CFG.get("project", {}).get("random_state", 42))
FINAL_K = int(CFG.get("project", {}).get("final_k", 20))
BOTTOM_VARIANCE_FRACTION = float(
    CFG.get("preprocessing", {}).get("variance_bottom_fraction", 0.10)
)
LOG2_THRESHOLD = float(
    CFG.get("preprocessing", {}).get("log2_threshold", 100.0)
)
STRING_REQUIRED_SCORE = int(CFG.get("ppi", {}).get("required_score", 700))
BOOTSTRAP_ITERATIONS = 1000
PRIMARY_MIN_TOTAL = 15  # reproduces the earlier eligibility rule

OUTPUT_DIR = PROJECT_DIR / "results" / "LOCO_FULL_TRANSFER_SAFE_GENELEVEL_V2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Project        :", PROJECT_DIR)
print("Output         :", OUTPUT_DIR)
print("Final k        :", FINAL_K)
print("LOCO mode      : gene-level V2, training-only representative probes, primary-matched variance cutoff")


# ------------------------------------------------------------
# 1. Generic helpers
# ------------------------------------------------------------
def _find_column(frame: pd.DataFrame, candidates: list[str]):
    normalized = {str(c).strip().lower(): c for c in frame.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def _labels_to_binary(values) -> np.ndarray:
    series = pd.Series(values).copy()
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        unique = set(numeric.astype(int).unique())
        if unique.issubset({0, 1}):
            return numeric.astype(int).to_numpy()

    text = series.astype(str).str.strip().str.lower()
    positive = text.str.contains(
        r"cancer|tumou?r|malignan|carcinoma", regex=True, na=False
    )
    negative = text.str.contains(
        r"normal|control|non[- ]?tumou?r|healthy|benign", regex=True, na=False
    )

    output = np.full(len(text), np.nan)
    output[positive.to_numpy()] = 1
    output[negative.to_numpy()] = 0

    if np.isnan(output).any():
        unknown = sorted(series.loc[np.isnan(output)].astype(str).unique())
        raise ValueError("Label tidak dapat dipetakan ke 0/1: " + str(unknown))
    return output.astype(int)


def _clean_gene_symbol(value: object) -> str | None:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "na", "---"}:
        return None
    return text


def _stable_seed(text: str) -> int:
    return int((RANDOM_STATE + zlib.crc32(text.encode("utf-8"))) % (2**32 - 1))


# ------------------------------------------------------------
# 2. Standardize annotation and STRING network
# ------------------------------------------------------------
probe_map_df = pd.DataFrame(probe_map).copy()
probe_col = _find_column(probe_map_df, ["ID_REF", "Probe", "Probe_ID", "probe_id"])
gene_col = _find_column(probe_map_df, ["Gene", "Symbol", "Gene_symbol", "HGNC_symbol"])
if probe_col is None or gene_col is None:
    raise KeyError(
        "probe_map harus memiliki kolom probe dan Gene. "
        f"Kolom tersedia: {list(probe_map_df.columns)}"
    )

probe_map_df = probe_map_df[[probe_col, gene_col]].copy()
probe_map_df.columns = ["ID_REF", "Gene"]
probe_map_df["ID_REF"] = probe_map_df["ID_REF"].astype(str).str.strip()
probe_map_df["Gene"] = probe_map_df["Gene"].map(_clean_gene_symbol)
probe_map_df = probe_map_df.dropna(subset=["ID_REF", "Gene"])
# A probe must map deterministically to one gene for this analysis.
probe_map_df = (
    probe_map_df.sort_values(["ID_REF", "Gene"])
    .drop_duplicates("ID_REF", keep="first")
    .reset_index(drop=True)
)

string_df = pd.DataFrame(string_edges).copy()
gene1_col = _find_column(string_df, ["Gene1", "gene1", "Protein1_symbol"])
gene2_col = _find_column(string_df, ["Gene2", "gene2", "Protein2_symbol"])
score_col = _find_column(string_df, ["combined_score", "Combined_score", "score"])
if gene1_col is None or gene2_col is None:
    raise KeyError(
        "string_edges harus memiliki kolom Gene1 dan Gene2. "
        f"Kolom tersedia: {list(string_df.columns)}"
    )

keep_cols = [gene1_col, gene2_col] + ([score_col] if score_col is not None else [])
string_df = string_df[keep_cols].copy()
rename = {gene1_col: "Gene1", gene2_col: "Gene2"}
if score_col is not None:
    rename[score_col] = "combined_score"
string_df = string_df.rename(columns=rename)
string_df["Gene1"] = string_df["Gene1"].astype(str).str.strip()
string_df["Gene2"] = string_df["Gene2"].astype(str).str.strip()
if "combined_score" in string_df.columns:
    string_df["combined_score"] = pd.to_numeric(
        string_df["combined_score"], errors="coerce"
    ).fillna(0)
    string_df = string_df.loc[
        string_df["combined_score"] >= STRING_REQUIRED_SCORE
    ].copy()
else:
    string_df["combined_score"] = STRING_REQUIRED_SCORE

string_df = string_df.loc[string_df["Gene1"] != string_df["Gene2"]].copy()
string_df[["A", "B"]] = np.sort(string_df[["Gene1", "Gene2"]].to_numpy(), axis=1)
string_df = (
    string_df.sort_values("combined_score", ascending=False)
    .drop_duplicates(["A", "B"])
    .rename(columns={"A": "Gene1_clean", "B": "Gene2_clean"})
)


# ------------------------------------------------------------
# 3. Raw cohort standardization
# ------------------------------------------------------------
def _standardize_raw_cohort(gse: str, expression_object, metadata_object):
    expr = pd.DataFrame(expression_object).copy()
    meta = pd.DataFrame(metadata_object).copy()

    if expr.empty or meta.empty:
        raise ValueError(f"{gse}: ekspresi atau metadata kosong.")

    # Recover ID_REF if it was stored as the index.
    id_ref_col = _find_column(expr, ["ID_REF", "Probe", "Probe_ID", "probe_id"])
    if id_ref_col is None:
        expr = expr.reset_index().rename(columns={expr.index.name or "index": "ID_REF"})
        id_ref_col = "ID_REF"
    elif id_ref_col != "ID_REF":
        expr = expr.rename(columns={id_ref_col: "ID_REF"})

    expr["ID_REF"] = expr["ID_REF"].astype(str).str.strip()
    expr = expr.drop_duplicates("ID_REF", keep="first")

    sample_col = _find_column(
        meta, ["GSM_ID", "Sample_ID", "sample", "geo_accession", "ID"]
    )
    label_col = _find_column(
        meta, ["Label", "Class", "Status", "Phenotype", "Diagnosis"]
    )
    if label_col is None:
        raise KeyError(f"{gse}: kolom label tidak ditemukan: {list(meta.columns)}")

    if sample_col is None:
        meta_ids = meta.index.astype(str)
    else:
        meta_ids = meta[sample_col].astype(str)

    meta = meta.copy()
    meta.index = meta_ids
    meta = meta.loc[~meta.index.duplicated(keep="first")]

    expr.columns = expr.columns.astype(str)
    sample_ids = [sid for sid in meta.index if sid in expr.columns]
    if not sample_ids:
        raise ValueError(f"{gse}: sample metadata tidak cocok dengan kolom ekspresi.")

    meta = meta.loc[sample_ids].copy()
    y = _labels_to_binary(meta[label_col])

    numeric = expr[["ID_REF"] + sample_ids].copy()
    numeric[sample_ids] = numeric[sample_ids].apply(pd.to_numeric, errors="coerce")

    finite = numeric[sample_ids].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    maximum = float(np.max(finite)) if finite.size else np.nan
    minimum = float(np.min(finite)) if finite.size else np.nan
    log2_applied = bool(np.isfinite(maximum) and maximum > LOG2_THRESHOLD)

    if log2_applied:
        shift = 1.0 - minimum if np.isfinite(minimum) and minimum < 0 else 1.0
        numeric[sample_ids] = np.log2(numeric[sample_ids] + shift)

    numeric = numeric.set_index("ID_REF")

    return {
        "probe_by_sample": numeric,
        "sample_ids": sample_ids,
        "metadata": meta,
        "y": y,
        "log2_applied": log2_applied,
        "raw_probe_count": int(len(numeric)),
    }


expression_key_map = {str(k): k for k in all_expression.keys()}
metadata_key_map = {str(k): k for k in all_metadata.keys()}
cohort_names = sorted(set(expression_key_map).intersection(metadata_key_map))
if len(cohort_names) < 2:
    raise RuntimeError("all_expression/all_metadata tidak berisi cukup cohort.")

cohort_raw = {}
for cohort in cohort_names:
    item = _standardize_raw_cohort(
        cohort,
        all_expression[expression_key_map[cohort]],
        all_metadata[metadata_key_map[cohort]],
    )
    cohort_raw[cohort] = item
    print(
        f"{cohort}: {len(item['sample_ids'])} samples, "
        f"{item['raw_probe_count']} probes, "
        f"cancer={int((item['y'] == 1).sum())}, "
        f"normal={int((item['y'] == 0).sum())}, "
        f"log2_applied={item['log2_applied']}"
    )

# Annotation-defined technical probe universe; no cohort expression is used here.
ANNOTATED_PROBES = pd.Index(probe_map_df["ID_REF"].unique())
ANNOTATED_GENES = pd.Index(probe_map_df["Gene"].unique())
print("\nAnnotated probes:", len(ANNOTATED_PROBES))
print("Annotated genes :", len(ANNOTATED_GENES))


# ------------------------------------------------------------
# 4. Training-only representative-probe selection
# ------------------------------------------------------------
def select_training_representative_probes(training_cohorts: list[str]):
    matrices = []
    for cohort in training_cohorts:
        mat = cohort_raw[cohort]["probe_by_sample"].reindex(ANNOTATED_PROBES)
        matrices.append(mat)

    pooled = pd.concat(matrices, axis=1)
    probe_variance = pooled.var(axis=1, ddof=1).fillna(0.0)

    table = probe_map_df.merge(
        probe_variance.rename("Training_probe_variance"),
        left_on="ID_REF",
        right_index=True,
        how="inner",
    )
    table = (
        table.sort_values(
            ["Gene", "Training_probe_variance", "ID_REF"],
            ascending=[True, False, True],
        )
        .drop_duplicates("Gene", keep="first")
        .reset_index(drop=True)
    )

    if len(table) < FINAL_K:
        raise RuntimeError("Training-derived mapped gene count is smaller than final k.")
    return table


def extract_gene_matrix(cohort: str, selected_probe_table: pd.DataFrame) -> pd.DataFrame:
    raw = cohort_raw[cohort]["probe_by_sample"]
    probes = selected_probe_table["ID_REF"].astype(str).tolist()
    genes = selected_probe_table["Gene"].astype(str).tolist()

    subset = raw.reindex(probes)
    subset.index = genes
    # rows=samples, columns=genes
    out = subset.T
    out.index = cohort_raw[cohort]["sample_ids"]
    out.index.name = "GSM_ID"
    return out


# ------------------------------------------------------------
# 5. Training-only quantile mapping and variance filtering
# ------------------------------------------------------------
def _fit_quantile_reference(X_train: pd.DataFrame) -> np.ndarray:
    array = X_train.to_numpy(dtype=np.float64, copy=True)
    return np.nanmean(np.sort(array, axis=1), axis=0)


def _apply_quantile_reference(X: pd.DataFrame, reference: np.ndarray) -> pd.DataFrame:
    array = X.to_numpy(dtype=np.float64, copy=True)
    if array.shape[1] != len(reference):
        raise ValueError("Jumlah gen tidak sama dengan panjang quantile reference.")

    normalized = np.empty_like(array)
    for i in range(array.shape[0]):
        values = array[i]
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        mapped = reference.copy()

        # Average the reference over tied ranks.
        start = 0
        while start < len(sorted_values):
            end = start + 1
            while end < len(sorted_values) and sorted_values[end] == sorted_values[start]:
                end += 1
            if end - start > 1:
                mapped[start:end] = np.mean(reference[start:end])
            start = end

        normalized[i, order] = mapped

    return pd.DataFrame(normalized, index=X.index, columns=X.columns)


def fit_transform_training_only(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    variance_bottom_fraction: float,
):
    training_medians = X_train_raw.median(axis=0).fillna(0.0)
    X_train_filled = X_train_raw.fillna(training_medians)
    X_test_filled = X_test_raw.fillna(training_medians)

    reference = _fit_quantile_reference(X_train_filled)
    X_train_qn = _apply_quantile_reference(X_train_filled, reference)
    X_test_qn = _apply_quantile_reference(X_test_filled, reference)

    variances = X_train_qn.var(axis=0, ddof=1).fillna(0.0)

    # Match the primary notebook exactly: remove genes at or below the
    # training-only 10th-percentile variance cutoff.  With 19,134 mapped
    # genes and no cutoff ties this retains 17,220 eligible genes.
    variance_cutoff = float(variances.quantile(variance_bottom_fraction))
    eligible_index = variances[variances > variance_cutoff].index.astype(str)

    # Deterministic alphabetical ordering after applying the cutoff.
    eligible_genes = sorted(eligible_index.tolist())
    if len(eligible_genes) < FINAL_K:
        raise RuntimeError(
            f"Variance filtering retained only {len(eligible_genes)} genes, "
            f"fewer than FINAL_K={FINAL_K}."
        )

    variance_table = pd.DataFrame(
        {"Gene": variances.index.astype(str), "Training_variance": variances.values}
    ).sort_values(["Training_variance", "Gene"], ascending=[False, True])
    variance_table["Variance_cutoff"] = variance_cutoff
    variance_table["Eligible"] = variance_table["Training_variance"] > variance_cutoff

    return (
        X_train_qn[eligible_genes].copy(),
        X_test_qn[eligible_genes].copy(),
        variance_table,
        eligible_genes,
    )


# ------------------------------------------------------------
# 6. R limma, training-only PPI degree, and NIBFS
# ------------------------------------------------------------
def run_limma_loco(X: pd.DataFrame, y: np.ndarray, training_cohort: np.ndarray):
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.conversion import localconverter
    except Exception as exc:
        raise ImportError("LOCO manuscript run memerlukan rpy2 dan R limma.") from exc

    ro.r("suppressPackageStartupMessages(library(limma))")

    genes = X.columns.astype(str).tolist()
    expr = X.to_numpy(dtype=float).T
    y_arr = np.asarray(y, dtype=int)
    cohort_arr = np.asarray(training_cohort, dtype=str)

    with localconverter(ro.default_converter + numpy2ri.converter):
        ro.globalenv["loco_expr_matrix"] = expr
        ro.globalenv["loco_group_vector"] = y_arr
    ro.globalenv["loco_training_cohort"] = ro.StrVector(cohort_arr.tolist())
    ro.globalenv["loco_gene_names"] = ro.StrVector(genes)

    ro.r(
        """
        rownames(loco_expr_matrix) <- loco_gene_names
        loco_group_factor <- factor(
            loco_group_vector, levels=c(0,1), labels=c("Normal","Cancer")
        )
        loco_batch_factor <- factor(loco_training_cohort)
        loco_design <- model.matrix(~ loco_group_factor + loco_batch_factor)
        loco_fit <- lmFit(loco_expr_matrix, loco_design)
        loco_fit <- eBayes(loco_fit)
        loco_result <- topTable(
            loco_fit, coef=2, number=Inf, adjust.method="BH", sort.by="none"
        )
        loco_result$Gene <- rownames(loco_result)
        """
    )

    columns = list(ro.r("colnames(loco_result)"))
    result = pd.DataFrame(
        {col: list(ro.r(f"loco_result${col}")) for col in columns}
    ).rename(columns={"adj.P.Val": "FDR", "P.Value": "P_value"})

    result["Gene"] = result["Gene"].astype(str)
    result["FDR"] = pd.to_numeric(result["FDR"], errors="coerce").fillna(1.0)
    result["logFC"] = pd.to_numeric(result["logFC"], errors="coerce").fillna(0.0)
    result["Stat_score"] = result["logFC"].abs() * (
        -np.log10(result["FDR"].clip(lower=1e-300))
    )
    result["Rank_stat"] = result["Stat_score"].rank(
        method="average", ascending=False
    )
    return result.sort_values(["Rank_stat", "Gene"]).reset_index(drop=True)


def training_only_degree_table(eligible_genes: list[str]) -> pd.DataFrame:
    genes = pd.Index(pd.Series(eligible_genes).astype(str).drop_duplicates())
    gene_set = set(genes)
    edges = string_df.loc[
        string_df["Gene1_clean"].isin(gene_set)
        & string_df["Gene2_clean"].isin(gene_set),
        ["Gene1_clean", "Gene2_clean", "combined_score"],
    ].copy()

    degree = pd.concat(
        [edges["Gene1_clean"], edges["Gene2_clean"]], ignore_index=True
    ).value_counts()

    out = pd.DataFrame({"Gene": genes})
    out["Network_degree"] = out["Gene"].map(degree).fillna(0).astype(int)
    out["Rank_topo"] = out["Network_degree"].rank(
        method="average", ascending=False
    )
    out.attrs["edge_count"] = int(len(edges))
    return out


def build_nibfs_ranking(limma_table: pd.DataFrame, degree_table: pd.DataFrame):
    p = len(degree_table)
    ranking = limma_table.merge(degree_table, on="Gene", how="left")
    ranking["Network_degree"] = ranking["Network_degree"].fillna(0)
    ranking["Rank_topo"] = ranking["Rank_topo"].fillna(float(p))
    ranking["Borda_stat"] = p - ranking["Rank_stat"] + 1
    ranking["Borda_topo"] = p - ranking["Rank_topo"] + 1
    ranking["Borda_score"] = ranking["Borda_stat"] + ranking["Borda_topo"]
    ranking = ranking.sort_values(
        ["Borda_score", "Rank_stat", "Gene"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    ranking["Rank_NIBFS"] = np.arange(1, len(ranking) + 1)
    return ranking


# ------------------------------------------------------------
# 7. Models and metrics
# ------------------------------------------------------------
def create_models():
    model_cfg = CFG.get("models", {})
    return {
        "LR": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        penalty="l2",
                        solver="lbfgs",
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "RF": RandomForestClassifier(
            n_estimators=int(model_cfg.get("rf_n_estimators", 500)),
            max_features=model_cfg.get("rf_max_features", "sqrt"),
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=int(model_cfg.get("lgbm_n_estimators", 500)),
            learning_rate=float(model_cfg.get("lgbm_learning_rate", 0.03)),
            num_leaves=int(model_cfg.get("lgbm_num_leaves", 31)),
            subsample=float(model_cfg.get("lgbm_subsample", 0.9)),
            colsample_bytree=float(model_cfg.get("lgbm_colsample_bytree", 0.9)),
            objective="binary",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        ),
    }


def bootstrap_auc_ci(y_true, probabilities, iterations, seed):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(iterations):
        idx = np.concatenate(
            [
                rng.choice(pos, size=len(pos), replace=True),
                rng.choice(neg, size=len(neg), replace=True),
            ]
        )
        values.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.quantile(values, [0.025, 0.975])
    return float(lo), float(hi)


def metric_row(y_true, probabilities, threshold, seed):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    has_two = bool(np.any(y == 0) and np.any(y == 1))
    if has_two:
        auc = float(roc_auc_score(y, p))
        auc_lo, auc_hi = bootstrap_auc_ci(
            y, p, BOOTSTRAP_ITERATIONS, seed
        )
        bal = float(balanced_accuracy_score(y, pred))
        f1 = float(f1_score(y, pred, zero_division=0))
        mcc = float(matthews_corrcoef(y, pred))
    else:
        auc = auc_lo = auc_hi = bal = f1 = mcc = np.nan

    sensitivity = float(tp / (tp + fn)) if (tp + fn) else np.nan
    specificity = float(tn / (tn + fp)) if (tn + fp) else np.nan
    precision = float(precision_score(y, pred, zero_division=0)) if np.any(y == 1) else np.nan

    return {
        "ROC_AUC": auc,
        "ROC_AUC_CI_low": auc_lo,
        "ROC_AUC_CI_high": auc_hi,
        "Accuracy": float(accuracy_score(y, pred)),
        "Balanced_accuracy": bal,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "F1": f1,
        "MCC": mcc,
        "Brier_score": float(brier_score_loss(y, p)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


# ------------------------------------------------------------
# 8. LOCO execution with per-cohort checkpoints
# ------------------------------------------------------------
all_metric_rows = []
all_prediction_rows = []
all_panel_rows = []
all_probe_rows = []
all_audit_rows = []

for fold_index, heldout in enumerate(cohort_names, start=1):
    prefix = f"{fold_index:02d}_{heldout}"
    metric_file = OUTPUT_DIR / f"{prefix}_metrics.csv"
    prediction_file = OUTPUT_DIR / f"{prefix}_predictions.csv"
    panel_file = OUTPUT_DIR / f"{prefix}_selected_panel.csv"
    probe_file = OUTPUT_DIR / f"{prefix}_selected_probe_map.csv"
    audit_file = OUTPUT_DIR / f"{prefix}_audit.csv"

    checkpoint_files = [metric_file, prediction_file, panel_file, probe_file, audit_file]
    if all(path.exists() for path in checkpoint_files):
        print(f"\n[{fold_index}/{len(cohort_names)}] {heldout}: checkpoint loaded")
        all_metric_rows.append(pd.read_csv(metric_file))
        all_prediction_rows.append(pd.read_csv(prediction_file))
        all_panel_rows.append(pd.read_csv(panel_file))
        all_probe_rows.append(pd.read_csv(probe_file))
        all_audit_rows.append(pd.read_csv(audit_file))
        continue

    print(f"\n[{fold_index}/{len(cohort_names)}] Held out: {heldout}")
    started = perf_counter()
    training_cohorts = [c for c in cohort_names if c != heldout]

    selected_probes = select_training_representative_probes(training_cohorts)

    train_frames = []
    train_y_parts = []
    train_cohort_parts = []
    for cohort in training_cohorts:
        frame = extract_gene_matrix(cohort, selected_probes)
        train_frames.append(frame)
        train_y_parts.append(cohort_raw[cohort]["y"])
        train_cohort_parts.append(np.repeat(cohort, len(frame)))

    X_train_raw = pd.concat(train_frames, axis=0)
    y_train = np.concatenate(train_y_parts)
    train_cohort_vector = np.concatenate(train_cohort_parts)

    X_test_raw = extract_gene_matrix(heldout, selected_probes)
    y_test = cohort_raw[heldout]["y"]

    (
        X_train,
        X_test,
        variance_table,
        eligible_genes,
    ) = fit_transform_training_only(
        X_train_raw,
        X_test_raw,
        BOTTOM_VARIANCE_FRACTION,
    )

    print(
        f"  preprocessing complete | mapped_genes={X_train_raw.shape[1]} | "
        f"eligible_genes={X_train.shape[1]} | train={X_train.shape} | test={X_test.shape}"
    )

    limma_table = run_limma_loco(X_train, y_train, train_cohort_vector)
    degree_table = training_only_degree_table(eligible_genes)
    ranking = build_nibfs_ranking(limma_table, degree_table)
    panel = ranking.head(FINAL_K).copy()

    selected_genes = panel["Gene"].astype(str).tolist()
    if len(selected_genes) != FINAL_K or len(set(selected_genes)) != FINAL_K:
        raise RuntimeError(f"{heldout}: panel tidak berisi {FINAL_K} gene symbols unik.")
    numeric_like = sum(bool(re.fullmatch(r"\d+", gene)) for gene in selected_genes)
    if numeric_like > 0:
        raise RuntimeError(
            f"{heldout}: ditemukan {numeric_like} panel entries numerik. "
            "Probe-to-gene mapping gagal; hasil dihentikan."
        )

    ppi_match_count = int((degree_table["Network_degree"] > 0).sum())
    ppi_match_prop = float(ppi_match_count / len(degree_table))
    selected_with_degree = int((panel["Network_degree"] > 0).sum())

    X_train_panel = X_train[selected_genes]
    X_test_panel = X_test[selected_genes]

    both_classes = bool(np.unique(y_test).size == 2)
    total = int(len(y_test))
    normal_n = int((y_test == 0).sum())
    cancer_n = int((y_test == 1).sum())
    primary_eligible = bool(both_classes and total >= PRIMARY_MIN_TOTAL)

    if not both_classes:
        status = "Descriptive: normal-only cohort" if cancer_n == 0 else "Descriptive: cancer-only cohort"
    elif total < PRIMARY_MIN_TOTAL:
        status = "Descriptive: very small cohort"
    else:
        status = "Primary LOCO ROC-AUC eligible"

    metric_rows = []
    prediction_rows = []
    for model_name, model in create_models().items():
        model.fit(X_train_panel, y_train)
        probability = model.predict_proba(X_test_panel)[:, 1]
        metrics = metric_row(
            y_test,
            probability,
            threshold=0.5,
            seed=_stable_seed(f"{heldout}|{model_name}"),
        )
        metric_rows.append(
            {
                "Held_out_cohort": heldout,
                "Model": model_name,
                "Normal": normal_n,
                "Cancer": cancer_n,
                "Total": total,
                "Both_classes": both_classes,
                "Primary_ROC_AUC_eligible": primary_eligible,
                "LOCO_status": status,
                "Panel_size": FINAL_K,
                "Threshold": 0.5,
                **metrics,
            }
        )

        sample_ids = X_test_panel.index.astype(str).tolist()
        prediction_rows.extend(
            {
                "Held_out_cohort": heldout,
                "GSM_ID": sid,
                "Model": model_name,
                "True_label": int(label),
                "Probability_cancer": float(prob),
                "Predicted_label": int(prob >= 0.5),
            }
            for sid, label, prob in zip(sample_ids, y_test, probability)
        )

    panel_out = panel[
        [
            "Gene",
            "Rank_NIBFS",
            "Borda_score",
            "Rank_stat",
            "Rank_topo",
            "Network_degree",
            "logFC",
            "FDR",
            "Stat_score",
        ]
    ].copy()
    panel_out.insert(0, "Held_out_cohort", heldout)

    probe_out = selected_probes.copy()
    probe_out.insert(0, "Held_out_cohort", heldout)
    probe_out["Selection_source"] = "Held-out-excluded pooled training cohorts"

    elapsed = perf_counter() - started
    audit = pd.DataFrame(
        [
            {
                "Held_out_cohort": heldout,
                "Training_cohorts": "|".join(training_cohorts),
                "Training_samples": int(len(X_train)),
                "Held_out_samples": int(len(X_test)),
                "Held_out_normal": normal_n,
                "Held_out_cancer": cancer_n,
                "Representative_probe_source": "Held-out-excluded pooled training cohorts",
                "Heldout_used_for_probe_selection": False,
                "Mapped_genes_before_variance_filter": int(X_train_raw.shape[1]),
                "Eligible_genes": int(X_train.shape[1]),
                "Selected_panel_size": FINAL_K,
                "Panel_is_gene_symbols": True,
                "Numeric_panel_entries": numeric_like,
                "STRING_edges_in_training_eligible_network": int(degree_table.attrs.get("edge_count", 0)),
                "Genes_with_nonzero_STRING_degree": ppi_match_count,
                "Genes_with_nonzero_STRING_degree_proportion": ppi_match_prop,
                "Selected_genes_with_nonzero_STRING_degree": selected_with_degree,
                "Variance_bottom_fraction": BOTTOM_VARIANCE_FRACTION,
                "Quantile_reference_source": "Held-out-excluded training samples",
                "Imputation_source": "Held-out-excluded training medians",
                "LR_scaling_source": "Held-out-excluded training samples",
                "Limma_source": "Held-out-excluded training samples with training-cohort covariate",
                "PPI_degree_source": "STRING induced network on training-only eligible genes",
                "ComBat_used": False,
                "Reason_ComBat_not_used": "Standard ComBat cannot directly transfer fitted batch parameters to an unseen held-out batch.",
                "Decision_threshold": 0.5,
                "Elapsed_seconds": elapsed,
                "Elapsed_minutes": elapsed / 60.0,
            }
        ]
    )

    metric_df = pd.DataFrame(metric_rows)
    prediction_df = pd.DataFrame(prediction_rows)

    metric_df.to_csv(metric_file, index=False)
    prediction_df.to_csv(prediction_file, index=False)
    panel_out.to_csv(panel_file, index=False)
    probe_out.to_csv(probe_file, index=False)
    audit.to_csv(audit_file, index=False)

    all_metric_rows.append(metric_df)
    all_prediction_rows.append(prediction_df)
    all_panel_rows.append(panel_out)
    all_probe_rows.append(probe_out)
    all_audit_rows.append(audit)

    print(
        f"  completed | {elapsed/60.0:.2f} minutes | "
        f"panel={', '.join(selected_genes[:5])}, ... | "
        f"selected with PPI degree={selected_with_degree}/{FINAL_K}"
    )

    del X_train_raw, X_test_raw, X_train, X_test, X_train_panel, X_test_panel
    gc.collect()


# ------------------------------------------------------------
# 9. Aggregate outputs
# ------------------------------------------------------------
loco_metrics_genelevel = pd.concat(all_metric_rows, ignore_index=True)
loco_predictions_genelevel = pd.concat(all_prediction_rows, ignore_index=True)
loco_panels_genelevel = pd.concat(all_panel_rows, ignore_index=True)
loco_probe_maps_genelevel = pd.concat(all_probe_rows, ignore_index=True)
loco_audit_genelevel = pd.concat(all_audit_rows, ignore_index=True)

loco_metrics_genelevel.to_csv(OUTPUT_DIR / "loco_metrics_genelevel.csv", index=False)
loco_predictions_genelevel.to_csv(
    OUTPUT_DIR / "loco_predictions_all_samples_genelevel.csv", index=False
)
loco_panels_genelevel.to_csv(
    OUTPUT_DIR / "loco_selected_panels_genelevel.csv", index=False
)
loco_probe_maps_genelevel.to_csv(
    OUTPUT_DIR / "loco_selected_probe_maps_training_only.csv", index=False
)
loco_audit_genelevel.to_csv(
    OUTPUT_DIR / "loco_leakage_audit_genelevel.csv", index=False
)

frequency = (
    loco_panels_genelevel.groupby("Gene")
    .agg(
        Selection_frequency=("Held_out_cohort", "nunique"),
        Mean_panel_position=("Rank_NIBFS", "mean"),
        Mean_network_degree=("Network_degree", "mean"),
    )
    .reset_index()
)
frequency["Selection_proportion"] = frequency["Selection_frequency"] / len(cohort_names)
frequency = frequency.sort_values(
    ["Selection_frequency", "Mean_panel_position", "Gene"],
    ascending=[False, True, True],
)
frequency.to_csv(OUTPUT_DIR / "loco_gene_selection_frequency_genelevel.csv", index=False)

eligible = loco_metrics_genelevel.loc[
    loco_metrics_genelevel["Primary_ROC_AUC_eligible"]
    & loco_metrics_genelevel["ROC_AUC"].notna()
].copy()

eligible_summary = (
    eligible.groupby("Model")["ROC_AUC"]
    .agg(
        Eligible_cohorts="count",
        Mean_ROC_AUC="mean",
        SD_ROC_AUC="std",
        Median_ROC_AUC="median",
        Minimum_ROC_AUC="min",
        Maximum_ROC_AUC="max",
    )
    .reset_index()
)
eligible_summary.to_csv(
    OUTPUT_DIR / "loco_primary_eligible_summary_genelevel.csv", index=False
)

manifest = {
    "analysis": "Full transfer-safe LOCO sensitivity analysis, gene-level corrected",
    "project_dir": str(PROJECT_DIR),
    "output_dir": str(OUTPUT_DIR),
    "held_out_cohorts": cohort_names,
    "final_k": FINAL_K,
    "probe_mapping": "HGNC gene symbols",
    "representative_probe_rule": "highest pooled training variance per gene; held-out excluded",
    "quantile_reference": "training-only",
    "variance_filter": "training-only",
    "limma": "training-only with training-cohort covariate",
    "PPI_degree": "STRING induced network on training-only eligible genes",
    "ComBat": False,
    "threshold": 0.5,
    "primary_eligibility_rule": f"both classes and total >= {PRIMARY_MIN_TOTAL}",
}
with (OUTPUT_DIR / "loco_run_manifest_genelevel.json").open("w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2, ensure_ascii=False)

print("\n" + "=" * 76)
print("FULL TRANSFER-SAFE LOCO — GENE-LEVEL FIX COMPLETED")
print("=" * 76)
print("\nPrimary eligible summary:")
display(eligible_summary)
print("\nPer-cohort metrics:")
display(loco_metrics_genelevel)
print("\nMost recurrent LOCO genes:")
display(frequency.head(30))
print("\nLeakage and mapping audit:")
display(loco_audit_genelevel)
print("\nGenerated files:")
for path in sorted(OUTPUT_DIR.glob("*")):
    print(" -", path.name)
