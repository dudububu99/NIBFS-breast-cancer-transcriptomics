# ============================================================
# REPEATED 10 x 5-FOLD CV V2 — K=20, LR, FOUR SELECTORS
# ============================================================
#
# Run from the already executed NIBFS notebook with:
#
#   %run -i src/repeated_10x5_k20_lr_V2.py
#
# This script DOES NOT rerun or overwrite the main notebook analysis.
# It performs only the additional repeated-CV robustness analysis on the
# 608-sample model-development set.
#
# Primary design:
#   - 10 repeats x stratified 5-fold CV = 50 validation folds
#   - development set only (the designated 152-sample internal assessment set is excluded)
#   - k = 20
#   - selectors: NIBFS, DEG-only, mRMR, LASSO
#   - classifier: logistic regression only
#   - limma and every supervised selector are refitted inside each fold
#   - fixed decision threshold = 0.5
#   - fold-level checkpoints allow safe resume after interruption
#
# Optional notebook overrides before running:
#
#   REPEATED_REPEATS = 10
#   REPEATED_FOLDS = 5
#   REPEATED_MAX_NEW_FOLDS = None   # e.g. 1 for a benchmark, None for full
#   REPEATED_FORCE_RERUN = False
#
# Scientific role:
#   This is a robustness/sensitivity analysis. It does not replace the
#   original five-fold CV, does not redefine the frozen top-20 panel, and
#   does not alter the reported internal or external evaluation results.
# ============================================================

from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path
from time import perf_counter
import gc
import json
import math
import os
import re
import warnings

import numpy as np
import pandas as pd
import yaml

from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import friedmanchisquare, wilcoxon


# ------------------------------------------------------------
# 0. Configuration
# ------------------------------------------------------------

def _resolve_project_dir() -> Path:
    """Resolve project without requiring config.yaml.

    Priority:
    1. REPEATED_PROJECT_DIR supplied from the notebook;
    2. parent of this script (src/..);
    3. current working directory or one of its parents.
    """
    override = globals().get("REPEATED_PROJECT_DIR")
    if override:
        candidate = Path(str(override)).expanduser().resolve()
        if (candidate / "src").exists():
            return candidate
        raise FileNotFoundError(
            f"REPEATED_PROJECT_DIR tidak valid: {candidate}"
        )

    script_file = globals().get("__file__")
    if script_file:
        candidate = Path(str(script_file)).expanduser().resolve().parent.parent
        if (candidate / "src").exists():
            return candidate

    start = Path.cwd().resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "src").exists() and (candidate / "results").exists():
            return candidate

    raise FileNotFoundError(
        "Folder proyek tidak dapat ditentukan. Tetapkan REPEATED_PROJECT_DIR "
        "ke folder utama proyek sebelum menjalankan script."
    )


PROJECT_DIR = _resolve_project_dir()
CONFIG_PATH = PROJECT_DIR / "config.yaml"

# config.yaml bersifat opsional pada V2. Jika tidak tersedia, gunakan
# parameter yang sudah dikunci untuk analisis repeated paper.
CFG = {}
if CONFIG_PATH.exists():
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        CFG = yaml.safe_load(handle) or {}
else:
    warnings.warn(
        f"config.yaml tidak ditemukan di {PROJECT_DIR}. "
        "Repeated V2 memakai parameter paper yang dikunci.",
        RuntimeWarning,
    )

RANDOM_STATE = int(
    globals().get(
        "REPEATED_RANDOM_STATE",
        CFG.get("project", {}).get("random_state", 42),
    )
)
FINAL_K = int(
    globals().get(
        "REPEATED_FINAL_K",
        CFG.get("project", {}).get("final_k", 20),
    )
)
REPEATS = int(globals().get("REPEATED_REPEATS", 10))
FOLDS = int(globals().get("REPEATED_FOLDS", 5))
MAX_NEW_FOLDS = globals().get("REPEATED_MAX_NEW_FOLDS", None)
FORCE_RERUN = bool(globals().get("REPEATED_FORCE_RERUN", False))

if MAX_NEW_FOLDS is not None:
    MAX_NEW_FOLDS = int(MAX_NEW_FOLDS)
    if MAX_NEW_FOLDS < 1:
        raise ValueError("REPEATED_MAX_NEW_FOLDS harus None atau bilangan >= 1.")

if FINAL_K != 20:
    raise ValueError(
        f"Script ini dikunci untuk k=20, tetapi FINAL_K={FINAL_K}."
    )
if REPEATS != 10 or FOLDS != 5:
    warnings.warn(
        f"Desain aktif adalah {REPEATS} repeats x {FOLDS} folds. "
        "Untuk analisis paper gunakan 10 x 5.",
        RuntimeWarning,
    )

METHODS = ["NIBFS", "DEG-only", "mRMR", "LASSO"]
CLASSIFIER = "LR"
DEFAULT_THRESHOLD = float(
    globals().get(
        "REPEATED_THRESHOLD",
        CFG.get("models", {}).get("default_decision_threshold", 0.5),
    )
)
MRMR_CANDIDATE_SIZE = int(
    globals().get(
        "REPEATED_MRMR_CANDIDATE_SIZE",
        CFG.get("feature_selection", {}).get("mrmr_candidate_size", 1000),
    )
)
LASSO_C = float(
    globals().get(
        "REPEATED_LASSO_C",
        CFG.get("feature_selection", {}).get("lasso_C", 1.0),
    )
)
LASSO_SOLVER = str(
    globals().get(
        "REPEATED_LASSO_SOLVER",
        CFG.get("feature_selection", {}).get("lasso_solver", "saga"),
    )
)
LASSO_MAX_ITER = int(
    globals().get(
        "REPEATED_LASSO_MAX_ITER",
        CFG.get("feature_selection", {}).get("lasso_max_iter", 10000),
    )
)

OUTPUT_DIR = PROJECT_DIR / "results" / "REPEATED_10X5_K20_LR"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FIGURE_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 1. Data recovery: use active notebook objects first
# ------------------------------------------------------------

def _latest_path(patterns: list[str]) -> Path | None:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(PROJECT_DIR.rglob(pattern))
    matches = [p for p in matches if p.is_file()]
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _find_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {str(c).strip().lower(): str(c) for c in frame.columns}
    for candidate in candidates:
        if candidate.strip().lower() in lookup:
            return lookup[candidate.strip().lower()]
    return None


def _labels_to_binary(values) -> np.ndarray:
    s = pd.Series(values)
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().all() and set(numeric.astype(int).unique()).issubset({0, 1}):
        return numeric.astype(int).to_numpy()
    text = s.astype(str).str.strip().str.lower()
    out = np.full(len(text), np.nan)
    out[text.str.contains(r"cancer|tumou?r|malignan|carcinoma", regex=True)] = 1
    out[text.str.contains(r"normal|control|healthy", regex=True)] = 0
    if np.isnan(out).any():
        unknown = sorted(s[np.isnan(out)].astype(str).unique())
        raise ValueError(f"Label tidak dapat dipetakan menjadi 0/1: {unknown}")
    return out.astype(int)


def _load_development_from_disk() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    expression_path = _latest_path(
        ["harmonized_expression_matrix.csv.gz", "harmonized_expression_matrix.csv"]
    )
    split_path = _latest_path(
        [
            "train_test_split_assignments.csv",
            "discovery_train_test_assignments.csv",
        ]
    )
    if expression_path is None or split_path is None:
        raise RuntimeError(
            "X_train/y_train tidak aktif dan file ekspresi/split tidak ditemukan. "
            "Jalankan hanya sel load hasil utama sampai X_train, y_train, dan "
            "metadata_train aktif; tidak perlu mengulang analisis utama."
        )

    print("Loading development matrix from disk:", expression_path, flush=True)
    expression = pd.read_csv(expression_path)
    sample_col = _find_column(expression, ["GSM_ID", "Sample_ID", "sample"])
    if sample_col is None:
        sample_col = str(expression.columns[0])
    expression[sample_col] = expression[sample_col].astype(str)
    expression = expression.set_index(sample_col)

    split = pd.read_csv(split_path)
    gsm_col = _find_column(split, ["GSM_ID", "Sample_ID", "sample"])
    set_col = _find_column(split, ["Set", "Subset", "Role"])
    label_col = _find_column(split, ["Label_binary", "y", "Label"])
    if gsm_col is None or set_col is None or label_col is None:
        raise KeyError(
            f"Kolom split tidak lengkap pada {split_path}: {list(split.columns)}"
        )

    is_development = split[set_col].astype(str).str.lower().str.contains(
        r"model-development|development|train", regex=True
    )
    dev = split.loc[is_development].copy()
    dev[gsm_col] = dev[gsm_col].astype(str)
    dev = dev.drop_duplicates(gsm_col).set_index(gsm_col)

    missing_ids = dev.index.difference(expression.index)
    if len(missing_ids):
        raise RuntimeError(
            f"Sebanyak {len(missing_ids)} development sample tidak ditemukan "
            "pada harmonized expression matrix."
        )

    X = expression.loc[dev.index].copy()
    y = _labels_to_binary(dev[label_col])
    metadata = dev.reset_index().rename(columns={gsm_col: "GSM_ID"})
    metadata.index = X.index
    return X, y, metadata


def _load_ppi_degree_from_disk() -> pd.DataFrame:
    path = _latest_path(["ppi_degree_table.csv"])
    if path is None:
        raise RuntimeError(
            "ppi_degree tidak aktif dan ppi_degree_table.csv tidak ditemukan."
        )
    print("Loading PPI degree table from disk:", path, flush=True)
    return pd.read_csv(path)


if "X_train" in globals() and "y_train" in globals():
    X_DEV = pd.DataFrame(globals()["X_train"]).copy()
    Y_DEV = np.asarray(globals()["y_train"], dtype=int)
    if "metadata_train" in globals():
        META_DEV = pd.DataFrame(globals()["metadata_train"]).copy()
    else:
        META_DEV = pd.DataFrame(index=X_DEV.index)
else:
    X_DEV, Y_DEV, META_DEV = _load_development_from_disk()

if "ppi_degree" in globals():
    PPI_DEGREE = pd.DataFrame(globals()["ppi_degree"]).copy()
else:
    PPI_DEGREE = _load_ppi_degree_from_disk()

X_DEV.index = X_DEV.index.astype(str)
X_DEV.columns = X_DEV.columns.astype(str)
Y_DEV = np.asarray(Y_DEV, dtype=int)

if len(X_DEV) != len(Y_DEV):
    raise ValueError("Jumlah baris X_train tidak sama dengan panjang y_train.")
if X_DEV.index.duplicated().any():
    raise ValueError("GSM_ID pada development set tidak unik.")
if set(np.unique(Y_DEV)) != {0, 1}:
    raise ValueError("Development labels harus mengandung kelas 0 dan 1.")

# Strict manuscript-run guard: prevents accidental use of the full 760 samples
# or the wrong feature universe.
if X_DEV.shape != (608, 17220):
    raise RuntimeError(
        "Repeated analysis harus memakai development matrix (608, 17220), "
        f"tetapi ditemukan {X_DEV.shape}. Jangan gunakan seluruh 760 sampel."
    )

# Align metadata without assuming its current index convention.
if len(META_DEV) == len(X_DEV):
    meta = META_DEV.copy()
    gsm_col = _find_column(meta, ["GSM_ID", "Sample_ID", "sample"])
    if gsm_col is not None:
        meta[gsm_col] = meta[gsm_col].astype(str)
        meta = meta.drop_duplicates(gsm_col).set_index(gsm_col)
        if set(X_DEV.index).issubset(set(meta.index)):
            META_DEV = meta.loc[X_DEV.index].copy()
        else:
            META_DEV = pd.DataFrame(index=X_DEV.index)
    else:
        meta.index = X_DEV.index
        META_DEV = meta
else:
    META_DEV = pd.DataFrame(index=X_DEV.index)

if "GSM_ID" not in META_DEV.columns:
    META_DEV["GSM_ID"] = X_DEV.index
if "GEO_ID" not in META_DEV.columns:
    META_DEV["GEO_ID"] = "unknown"
if "Label" not in META_DEV.columns:
    META_DEV["Label"] = np.where(Y_DEV == 1, "Cancer", "Normal")

print("Project          :", PROJECT_DIR)
print("Output           :", OUTPUT_DIR)
print("Development data :", X_DEV.shape)
print("Class counts     :", dict(pd.Series(Y_DEV).value_counts().sort_index()))
print("Design           :", f"{REPEATS} repeats x {FOLDS} folds = {REPEATS * FOLDS} folds")
print("Selectors        :", ", ".join(METHODS))
print("Classifier       :", CLASSIFIER)
print("Final k          :", FINAL_K)
print("Checkpoint mode  : enabled")


# ------------------------------------------------------------
# 2. Fold-local feature-selection methods
# ------------------------------------------------------------

def run_limma_rpy2(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.conversion import localconverter
    except Exception as exc:
        raise ImportError(
            "Repeated analysis requires rpy2 and R limma. "
            "Jalankan sel instalasi R/limma notebook terlebih dahulu."
        ) from exc

    ro.r("suppressPackageStartupMessages(library(limma))")
    genes = X.columns.astype(str).tolist()
    expr = X.to_numpy(dtype=float).T
    y_arr = np.asarray(y, dtype=int)

    with localconverter(ro.default_converter + numpy2ri.converter):
        ro.globalenv["expr_matrix_repeated"] = expr
        ro.globalenv["group_vector_repeated"] = y_arr
    ro.globalenv["gene_names_repeated"] = ro.StrVector(genes)

    ro.r(
        """
        rownames(expr_matrix_repeated) <- gene_names_repeated
        group_factor_repeated <- factor(
            group_vector_repeated,
            levels=c(0,1),
            labels=c("Normal","Cancer")
        )
        design_repeated <- model.matrix(~ group_factor_repeated)
        fit_repeated <- lmFit(expr_matrix_repeated, design_repeated)
        fit_repeated <- eBayes(fit_repeated)
        limma_result_repeated <- topTable(
            fit_repeated,
            coef=2,
            number=Inf,
            adjust.method="BH",
            sort.by="none"
        )
        limma_result_repeated$Gene <- rownames(limma_result_repeated)
        """
    )

    cols = list(ro.r("colnames(limma_result_repeated)"))
    result = pd.DataFrame(
        {c: list(ro.r(f"limma_result_repeated${c}")) for c in cols}
    )
    result = result.rename(columns={"adj.P.Val": "FDR", "P.Value": "P_value"})
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


def prepare_ppi_rank_table(
    ppi_degree: pd.DataFrame, gene_universe: list[str]
) -> pd.DataFrame:
    genes = pd.Index(pd.Series(gene_universe).astype(str).drop_duplicates())
    gene_col = _find_column(ppi_degree, ["Gene", "Symbol", "Gene_symbol"])
    degree_col = _find_column(ppi_degree, ["Degree", "degree"])
    normalized_col = _find_column(
        ppi_degree, ["Normalized_degree", "normalized_degree"]
    )
    if gene_col is None or degree_col is None:
        raise KeyError(
            f"PPI degree table lacks Gene/Degree columns: {list(ppi_degree.columns)}"
        )

    ppi = ppi_degree[[gene_col, degree_col]].copy()
    ppi.columns = ["Gene", "Degree"]
    ppi["Gene"] = ppi["Gene"].astype(str)
    ppi["Degree"] = pd.to_numeric(ppi["Degree"], errors="coerce").fillna(0.0)

    if normalized_col is not None:
        ppi["Normalized_degree"] = pd.to_numeric(
            ppi_degree[normalized_col], errors="coerce"
        ).fillna(0.0)
    else:
        ppi["Normalized_degree"] = ppi["Degree"]

    ppi = ppi.groupby("Gene", as_index=False).agg(
        Degree=("Degree", "max"),
        Normalized_degree=("Normalized_degree", "max"),
    )
    out = pd.DataFrame({"Gene": genes}).merge(ppi, on="Gene", how="left")
    out[["Degree", "Normalized_degree"]] = out[
        ["Degree", "Normalized_degree"]
    ].fillna(0.0)
    out["Rank_topo"] = out["Normalized_degree"].rank(
        method="average", ascending=False
    )
    return out.sort_values(["Rank_topo", "Gene"]).reset_index(drop=True)


def nibfs_rank(
    limma: pd.DataFrame,
    ppi_rank: pd.DataFrame,
    gene_universe: list[str],
) -> pd.DataFrame:
    p = len(gene_universe)
    out = limma.merge(
        ppi_rank[["Gene", "Degree", "Normalized_degree", "Rank_topo"]],
        on="Gene",
        how="left",
    )
    out["Degree"] = out["Degree"].fillna(0.0)
    out["Normalized_degree"] = out["Normalized_degree"].fillna(0.0)
    out["Rank_topo"] = out["Rank_topo"].fillna(float(p))
    out["Borda_stat"] = p - out["Rank_stat"] + 1
    out["Borda_topo"] = p - out["Rank_topo"] + 1
    out["Borda_score"] = out["Borda_stat"] + out["Borda_topo"]
    out = out.sort_values(
        ["Borda_score", "Rank_stat", "Gene"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    out["Rank_NIBFS"] = np.arange(1, len(out) + 1)
    return out


def select_mrmr_features(
    X: pd.DataFrame,
    y: np.ndarray,
    k: int,
    candidate_size: int,
    random_state: int,
) -> pd.DataFrame:
    y_arr = np.asarray(y, dtype=int)
    relevance = pd.Series(
        mutual_info_classif(X, y_arr, random_state=random_state),
        index=X.columns,
    )
    candidates = relevance.sort_values(ascending=False).head(
        min(candidate_size, X.shape[1])
    ).index.tolist()
    corr = X[candidates].corr().abs().fillna(0.0)

    selected: list[str] = []
    rows: list[dict] = []
    while len(selected) < min(k, len(candidates)):
        best_gene = None
        best_tuple = None
        for gene in candidates:
            if gene in selected:
                continue
            redundancy = float(corr.loc[gene, selected].mean()) if selected else 0.0
            score = float(relevance.loc[gene] - redundancy)
            candidate_tuple = (score, float(relevance.loc[gene]), -redundancy)
            if best_tuple is None or candidate_tuple > best_tuple:
                best_tuple = candidate_tuple
                best_gene = gene
        if best_gene is None or best_tuple is None:
            break
        selected.append(str(best_gene))
        rows.append(
            {
                "Gene": str(best_gene),
                "Selection_Order": len(selected),
                "mRMR_score": best_tuple[0],
                "Relevance": best_tuple[1],
                "Redundancy": -best_tuple[2],
            }
        )
    return pd.DataFrame(rows)


def lasso_gene_ranking(
    X: pd.DataFrame,
    y: np.ndarray,
    C: float,
    random_state: int,
    solver: str,
    max_iter: int,
) -> tuple[pd.DataFrame, bool]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(
        C=C,
        penalty="l1",
        solver=solver,
        class_weight="balanced",
        max_iter=max_iter,
        n_jobs=-1 if solver == "saga" else None,
        random_state=random_state,
    )
    converged = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X_scaled, np.asarray(y, dtype=int))
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)

    out = pd.DataFrame(
        {
            "Gene": X.columns.astype(str),
            "Coefficient": model.coef_.ravel(),
        }
    )
    out["Abs_Coefficient"] = out["Coefficient"].abs()
    out = out.sort_values(
        ["Abs_Coefficient", "Gene"], ascending=[False, True]
    ).reset_index(drop=True)
    out["Rank_LASSO"] = np.arange(1, len(out) + 1)
    out["C"] = float(C)
    return out, converged


def select_top_k(table: pd.DataFrame, k: int, rank_col: str) -> list[str]:
    genes = (
        table.sort_values([rank_col, "Gene"])
        .head(k)["Gene"]
        .astype(str)
        .tolist()
    )
    if len(genes) != k or len(set(genes)) != k:
        raise RuntimeError(
            f"Panel {rank_col} tidak menghasilkan {k} gen unik: {genes}"
        )
    return genes


# Fixed external PPI rank source. PPI does not use sample labels and is not
# re-estimated from held-out validation samples.
PPI_RANK = prepare_ppi_rank_table(PPI_DEGREE, X_DEV.columns.tolist())


# ------------------------------------------------------------
# 3. Prediction and metric helpers
# ------------------------------------------------------------

def create_lr(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def classification_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "Threshold": float(threshold),
        "ROC_AUC": float(roc_auc_score(y, p)),
        "Accuracy": float(accuracy_score(y, pred)),
        "Balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "Sensitivity": float(recall_score(y, pred, zero_division=0)),
        "Specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y, pred)),
        "Brier_score": float(brier_score_loss(y, p)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def bh_adjust(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate(
        (ranked * n / np.arange(1, n + 1))[::-1]
    )[::-1]
    out = np.empty(n)
    out[order] = np.clip(adjusted, 0, 1)
    return out


def _safe_wilcoxon(a, b, alternative="greater") -> tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) == 0:
        return np.nan, np.nan
    try:
        statistic, p_value = wilcoxon(
            a,
            b,
            alternative=alternative,
            zero_method="wilcox",
        )
        return float(statistic), float(p_value)
    except ValueError:
        return np.nan, 1.0


# ------------------------------------------------------------
# 4. Checkpoint helpers
# ------------------------------------------------------------

def _checkpoint_prefix(repeat: int, fold: int) -> Path:
    return CHECKPOINT_DIR / f"repeat_{repeat:02d}_fold_{fold:02d}"


def _checkpoint_files(repeat: int, fold: int) -> dict[str, Path]:
    prefix = _checkpoint_prefix(repeat, fold)
    return {
        "metrics": prefix.with_name(prefix.name + "_metrics.csv"),
        "predictions": prefix.with_name(prefix.name + "_predictions.csv"),
        "panels": prefix.with_name(prefix.name + "_panels.csv"),
        "assignments": prefix.with_name(prefix.name + "_assignments.csv"),
        "runtime": prefix.with_name(prefix.name + "_runtime.json"),
        "done": prefix.with_name(prefix.name + "_DONE.json"),
    }


def _checkpoint_complete(repeat: int, fold: int) -> bool:
    files = _checkpoint_files(repeat, fold)
    return all(files[key].exists() for key in ["metrics", "predictions", "panels", "assignments", "runtime", "done"])


def _remove_checkpoint(repeat: int, fold: int) -> None:
    for path in _checkpoint_files(repeat, fold).values():
        if path.exists():
            path.unlink()


def _save_json_atomic(data: dict, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def _save_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


# ------------------------------------------------------------
# 5. Execute or resume 50 fold-validation runs
# ------------------------------------------------------------
TOTAL_FOLDS = REPEATS * FOLDS
completed_before = sum(
    _checkpoint_complete(repeat, fold)
    for repeat in range(1, REPEATS + 1)
    for fold in range(1, FOLDS + 1)
)

if FORCE_RERUN:
    print("FORCE_RERUN=True: removing existing repeated-CV checkpoints.", flush=True)
    for repeat in range(1, REPEATS + 1):
        for fold in range(1, FOLDS + 1):
            _remove_checkpoint(repeat, fold)
    completed_before = 0

print(f"Existing complete checkpoints: {completed_before}/{TOTAL_FOLDS}", flush=True)

new_fold_count = 0
session_fold_times: list[float] = []
stop_requested = False

for repeat in range(1, REPEATS + 1):
    cv = StratifiedKFold(
        n_splits=FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE + repeat - 1,
    )

    for fold, (fit_idx, val_idx) in enumerate(cv.split(X_DEV, Y_DEV), 1):
        files = _checkpoint_files(repeat, fold)
        global_position = (repeat - 1) * FOLDS + fold

        if _checkpoint_complete(repeat, fold):
            print(
                f"[{global_position:02d}/{TOTAL_FOLDS}] "
                f"repeat {repeat}/{REPEATS}, fold {fold}/{FOLDS}: checkpoint found — skipped",
                flush=True,
            )
            continue

        if MAX_NEW_FOLDS is not None and new_fold_count >= MAX_NEW_FOLDS:
            stop_requested = True
            break

        fold_start = perf_counter()
        fold_seed = RANDOM_STATE + (repeat - 1) * FOLDS + fold
        X_fit = X_DEV.iloc[fit_idx]
        y_fit = Y_DEV[fit_idx]
        X_val = X_DEV.iloc[val_idx]
        y_val = Y_DEV[val_idx]

        print("\n" + "=" * 76, flush=True)
        print(
            f"[{global_position:02d}/{TOTAL_FOLDS}] "
            f"REPEAT {repeat}/{REPEATS} — FOLD {fold}/{FOLDS}",
            flush=True,
        )
        print(
            f"training={X_fit.shape}, validation={X_val.shape}, "
            f"train classes={dict(pd.Series(y_fit).value_counts().sort_index())}",
            flush=True,
        )

        print("  [1/5] R limma on fold-training samples ...", flush=True)
        limma_table = run_limma_rpy2(X_fit, y_fit)

        print("  [2/5] Equal-weight NIBFS ranking (limma + fixed STRING degree) ...", flush=True)
        nibfs_table = nibfs_rank(limma_table, PPI_RANK, X_fit.columns.tolist())

        print(
            f"  [3/5] mRMR selection (candidate size={MRMR_CANDIDATE_SIZE}) ...",
            flush=True,
        )
        mrmr_table = select_mrmr_features(
            X_fit,
            y_fit,
            FINAL_K,
            MRMR_CANDIDATE_SIZE,
            fold_seed,
        )

        print(
            f"  [4/5] LASSO ranking (C={LASSO_C}, solver={LASSO_SOLVER}) ...",
            flush=True,
        )
        lasso_table, lasso_converged = lasso_gene_ranking(
            X_fit,
            y_fit,
            LASSO_C,
            fold_seed,
            LASSO_SOLVER,
            LASSO_MAX_ITER,
        )

        panels = {
            "NIBFS": select_top_k(nibfs_table, FINAL_K, "Rank_NIBFS"),
            "DEG-only": select_top_k(limma_table, FINAL_K, "Rank_stat"),
            "mRMR": select_top_k(mrmr_table, FINAL_K, "Selection_Order"),
            "LASSO": select_top_k(lasso_table, FINAL_K, "Rank_LASSO"),
        }

        print("  [5/5] Logistic-regression validation for four panels ...", flush=True)
        metrics_rows: list[dict] = []
        prediction_rows: list[dict] = []
        panel_rows: list[dict] = []

        for method in METHODS:
            genes = panels[method]
            model = create_lr(fold_seed)
            model.fit(X_fit[genes], y_fit)
            probability = model.predict_proba(X_val[genes])[:, 1]

            metric = classification_metrics(y_val, probability, DEFAULT_THRESHOLD)
            metrics_rows.append(
                {
                    "Repeat": repeat,
                    "Fold": fold,
                    "Method": method,
                    "Classifier": CLASSIFIER,
                    "k": FINAL_K,
                    "Training_samples": len(fit_idx),
                    "Validation_samples": len(val_idx),
                    "LASSO_converged": lasso_converged if method == "LASSO" else np.nan,
                    **metric,
                }
            )

            for sample_id, true_label, prob in zip(
                X_val.index.astype(str), y_val, probability
            ):
                prediction_rows.append(
                    {
                        "Repeat": repeat,
                        "Fold": fold,
                        "Method": method,
                        "Classifier": CLASSIFIER,
                        "k": FINAL_K,
                        "Sample_ID": sample_id,
                        "True_Label": int(true_label),
                        "Probability": float(prob),
                    }
                )

            detail_lookup = None
            if method == "NIBFS":
                detail_lookup = nibfs_table.set_index("Gene")
            elif method == "DEG-only":
                detail_lookup = limma_table.set_index("Gene")
            elif method == "mRMR":
                detail_lookup = mrmr_table.set_index("Gene")
            elif method == "LASSO":
                detail_lookup = lasso_table.set_index("Gene")

            for rank, gene in enumerate(genes, 1):
                row = {
                    "Repeat": repeat,
                    "Fold": fold,
                    "Method": method,
                    "k": FINAL_K,
                    "Selection_rank": rank,
                    "Gene": gene,
                }
                if detail_lookup is not None and gene in detail_lookup.index:
                    details = detail_lookup.loc[gene]
                    if isinstance(details, pd.DataFrame):
                        details = details.iloc[0]
                    for column in [
                        "logFC",
                        "FDR",
                        "Stat_score",
                        "Rank_stat",
                        "Degree",
                        "Normalized_degree",
                        "Rank_topo",
                        "Borda_score",
                        "Rank_NIBFS",
                        "mRMR_score",
                        "Relevance",
                        "Redundancy",
                        "Coefficient",
                        "Abs_Coefficient",
                        "Rank_LASSO",
                    ]:
                        if column in details.index:
                            row[column] = details[column]
                panel_rows.append(row)

        assignment_rows = []
        for local_idx in val_idx:
            sample_id = str(X_DEV.index[local_idx])
            meta_row = META_DEV.loc[sample_id]
            assignment_rows.append(
                {
                    "Repeat": repeat,
                    "Fold": fold,
                    "Sample_ID": sample_id,
                    "GEO_ID": str(meta_row.get("GEO_ID", "unknown")),
                    "Label": str(meta_row.get("Label", "Cancer" if Y_DEV[local_idx] == 1 else "Normal")),
                    "Label_binary": int(Y_DEV[local_idx]),
                    "Subset": "Validation",
                }
            )

        elapsed = perf_counter() - fold_start
        runtime_data = {
            "Repeat": repeat,
            "Fold": fold,
            "Elapsed_seconds": elapsed,
            "Elapsed_minutes": elapsed / 60.0,
            "Random_state": fold_seed,
            "Training_samples": len(fit_idx),
            "Validation_samples": len(val_idx),
            "LASSO_converged": bool(lasso_converged),
            "Completed": True,
        }

        _save_csv_atomic(pd.DataFrame(metrics_rows), files["metrics"])
        _save_csv_atomic(pd.DataFrame(prediction_rows), files["predictions"])
        _save_csv_atomic(pd.DataFrame(panel_rows), files["panels"])
        _save_csv_atomic(pd.DataFrame(assignment_rows), files["assignments"])
        _save_json_atomic(runtime_data, files["runtime"])
        _save_json_atomic(
            {
                "status": "complete",
                "repeat": repeat,
                "fold": fold,
                "elapsed_seconds": elapsed,
            },
            files["done"],
        )

        new_fold_count += 1
        session_fold_times.append(elapsed)
        complete_now = sum(
            _checkpoint_complete(r, f)
            for r in range(1, REPEATS + 1)
            for f in range(1, FOLDS + 1)
        )
        remaining = TOTAL_FOLDS - complete_now
        mean_seconds = float(np.mean(session_fold_times))
        eta_minutes = remaining * mean_seconds / 60.0

        print(
            f"  completed in {elapsed / 60.0:.2f} minutes | "
            f"checkpoints={complete_now}/{TOTAL_FOLDS} | "
            f"estimated remaining={eta_minutes:.1f} minutes",
            flush=True,
        )
        print(
            "  NIBFS panel: " + ", ".join(panels["NIBFS"][:8]) + ", ...",
            flush=True,
        )
        if not lasso_converged:
            print(
                "  WARNING: LASSO emitted a convergence warning; status was recorded.",
                flush=True,
            )

        del X_fit, X_val, y_fit, y_val
        del limma_table, nibfs_table, mrmr_table, lasso_table
        gc.collect()

    if stop_requested:
        break


# ------------------------------------------------------------
# 6. Aggregate all complete checkpoints
# ------------------------------------------------------------
metrics_parts: list[pd.DataFrame] = []
prediction_parts: list[pd.DataFrame] = []
panel_parts: list[pd.DataFrame] = []
assignment_parts: list[pd.DataFrame] = []
runtime_rows: list[dict] = []

for repeat in range(1, REPEATS + 1):
    for fold in range(1, FOLDS + 1):
        if not _checkpoint_complete(repeat, fold):
            continue
        files = _checkpoint_files(repeat, fold)
        metrics_parts.append(pd.read_csv(files["metrics"]))
        prediction_parts.append(pd.read_csv(files["predictions"]))
        panel_parts.append(pd.read_csv(files["panels"]))
        assignment_parts.append(pd.read_csv(files["assignments"]))
        runtime_rows.append(json.loads(files["runtime"].read_text(encoding="utf-8")))

if not metrics_parts:
    raise RuntimeError("Belum ada repeated-CV fold yang selesai.")

fold_metrics = pd.concat(metrics_parts, ignore_index=True).sort_values(
    ["Repeat", "Fold", "Method"]
)
predictions = pd.concat(prediction_parts, ignore_index=True).sort_values(
    ["Repeat", "Fold", "Method", "Sample_ID"]
)
selected_panels = pd.concat(panel_parts, ignore_index=True).sort_values(
    ["Repeat", "Fold", "Method", "Selection_rank"]
)
fold_assignments = pd.concat(assignment_parts, ignore_index=True).sort_values(
    ["Repeat", "Fold", "Sample_ID"]
)
runtime_table = pd.DataFrame(runtime_rows).sort_values(["Repeat", "Fold"])

fold_metrics.to_csv(OUTPUT_DIR / "repeated_fold_metrics.csv", index=False)
predictions.to_csv(OUTPUT_DIR / "repeated_predictions.csv", index=False)
selected_panels.to_csv(OUTPUT_DIR / "repeated_selected_panels.csv", index=False)
fold_assignments.to_csv(OUTPUT_DIR / "repeated_fold_assignments.csv", index=False)
runtime_table.to_csv(OUTPUT_DIR / "repeated_runtime.csv", index=False)


# ------------------------------------------------------------
# 7. Predictive summaries: fold level and OOF repeat level
# ------------------------------------------------------------
metric_columns = [
    "ROC_AUC",
    "Accuracy",
    "Balanced_accuracy",
    "Sensitivity",
    "Specificity",
    "Precision",
    "F1",
    "MCC",
    "Brier_score",
]

fold_summary = (
    fold_metrics.groupby(["Method", "Classifier", "k"])[metric_columns]
    .agg(["count", "mean", "std", "median", "min", "max"])
    .reset_index()
)
fold_summary.columns = [
    "_".join(str(x) for x in col if str(x)) if isinstance(col, tuple) else str(col)
    for col in fold_summary.columns
]
fold_summary.to_csv(
    OUTPUT_DIR / "repeated_fold_performance_summary.csv", index=False
)

oof_repeat_rows: list[dict] = []
for (repeat, method), group in predictions.groupby(["Repeat", "Method"], sort=True):
    duplicated = group["Sample_ID"].duplicated().any()
    if duplicated:
        raise RuntimeError(
            f"Repeat {repeat}, method {method}: validation prediction duplicated."
        )
    if len(group) != len(X_DEV):
        raise RuntimeError(
            f"Repeat {repeat}, method {method}: expected {len(X_DEV)} OOF predictions, "
            f"found {len(group)}."
        )
    metric = classification_metrics(
        group["True_Label"].to_numpy(int),
        group["Probability"].to_numpy(float),
        DEFAULT_THRESHOLD,
    )
    oof_repeat_rows.append(
        {
            "Repeat": int(repeat),
            "Method": method,
            "Classifier": CLASSIFIER,
            "k": FINAL_K,
            "Samples": len(group),
            **metric,
        }
    )

oof_metrics = pd.DataFrame(oof_repeat_rows).sort_values(["Repeat", "Method"])
oof_metrics.to_csv(
    OUTPUT_DIR / "repeated_oof_metrics_by_repeat.csv", index=False
)

oof_summary = (
    oof_metrics.groupby(["Method", "Classifier", "k"])[metric_columns]
    .agg(["count", "mean", "std", "median", "min", "max"])
    .reset_index()
)
oof_summary.columns = [
    "_".join(str(x) for x in col if str(x)) if isinstance(col, tuple) else str(col)
    for col in oof_summary.columns
]
oof_summary.to_csv(
    OUTPUT_DIR / "repeated_oof_performance_summary.csv", index=False
)


# ------------------------------------------------------------
# 8. Stability summaries and recurrence
# ------------------------------------------------------------
pairwise_rows: list[dict] = []
repeat_stability_rows: list[dict] = []

for (repeat, method), group in selected_panels.groupby(
    ["Repeat", "Method"], sort=True
):
    panel_by_fold = {
        int(fold): set(fold_group.sort_values("Selection_rank")["Gene"].astype(str))
        for fold, fold_group in group.groupby("Fold")
    }
    values: list[float] = []
    for fold_1, fold_2 in combinations(sorted(panel_by_fold), 2):
        a = panel_by_fold[fold_1]
        b = panel_by_fold[fold_2]
        intersection = len(a & b)
        union = len(a | b)
        jaccard = intersection / union if union else np.nan
        values.append(jaccard)
        pairwise_rows.append(
            {
                "Repeat": int(repeat),
                "Method": method,
                "k": FINAL_K,
                "Fold_1": fold_1,
                "Fold_2": fold_2,
                "Intersection": intersection,
                "Union": union,
                "Jaccard": jaccard,
            }
        )
    repeat_stability_rows.append(
        {
            "Repeat": int(repeat),
            "Method": method,
            "k": FINAL_K,
            "Fold_panels": len(panel_by_fold),
            "Pairwise_comparisons": len(values),
            "Mean_Jaccard": float(np.mean(values)) if values else np.nan,
            "SD_Jaccard": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
            "Median_Jaccard": float(np.median(values)) if values else np.nan,
            "Minimum_Jaccard": float(np.min(values)) if values else np.nan,
            "Maximum_Jaccard": float(np.max(values)) if values else np.nan,
        }
    )

pairwise_jaccard = pd.DataFrame(pairwise_rows).sort_values(
    ["Repeat", "Method", "Fold_1", "Fold_2"]
)
repeat_stability = pd.DataFrame(repeat_stability_rows).sort_values(
    ["Repeat", "Method"]
)
pairwise_jaccard.to_csv(
    OUTPUT_DIR / "repeated_within_repeat_pairwise_jaccard.csv", index=False
)
repeat_stability.to_csv(
    OUTPUT_DIR / "repeated_stability_by_repeat.csv", index=False
)

stability_summary = (
    repeat_stability.groupby(["Method", "k"])["Mean_Jaccard"]
    .agg(["count", "mean", "std", "median", "min", "max"])
    .reset_index()
    .rename(
        columns={
            "count": "Repeats",
            "mean": "Mean_of_repeat_mean_Jaccard",
            "std": "SD_of_repeat_mean_Jaccard",
            "median": "Median_of_repeat_mean_Jaccard",
            "min": "Minimum_repeat_mean_Jaccard",
            "max": "Maximum_repeat_mean_Jaccard",
        }
    )
)
stability_summary.to_csv(
    OUTPUT_DIR / "repeated_stability_summary.csv", index=False
)

recurrence_rows: list[dict] = []
for method, group in selected_panels.groupby("Method", sort=True):
    counts = Counter(group["Gene"].astype(str))
    total_panels = group[["Repeat", "Fold"]].drop_duplicates().shape[0]
    positions = group.groupby("Gene")["Selection_rank"].mean()
    for gene, frequency in counts.items():
        recurrence_rows.append(
            {
                "Method": method,
                "Gene": gene,
                "Selection_frequency": int(frequency),
                "Total_fold_panels": int(total_panels),
                "Selection_proportion": float(frequency / total_panels),
                "Mean_selection_rank": float(positions.loc[gene]),
            }
        )

gene_recurrence = pd.DataFrame(recurrence_rows).sort_values(
    ["Method", "Selection_frequency", "Mean_selection_rank", "Gene"],
    ascending=[True, False, True, True],
)
gene_recurrence.to_csv(
    OUTPUT_DIR / "repeated_gene_selection_frequency.csv", index=False
)


# ------------------------------------------------------------
# 9. Paired statistical tests across repeats
# ------------------------------------------------------------
def paired_comparison_table(
    frame: pd.DataFrame,
    value_column: str,
    analysis: str,
) -> pd.DataFrame:
    wide = frame.pivot(index="Repeat", columns="Method", values=value_column).dropna()
    rows: list[dict] = []

    if all(method in wide.columns for method in METHODS) and len(wide) >= 2:
        try:
            statistic, p_value = friedmanchisquare(
                *[wide[method].to_numpy(float) for method in METHODS]
            )
        except ValueError:
            statistic, p_value = np.nan, np.nan
        rows.append(
            {
                "Analysis": analysis,
                "Test": "Friedman",
                "Proposed": "NIBFS",
                "Comparator": "All methods",
                "Alternative": "two-sided",
                "N_repeats": len(wide),
                "Statistic": statistic,
                "P_value": p_value,
                "Mean_difference": np.nan,
                "Median_difference": np.nan,
            }
        )

    posthoc_rows = []
    if "NIBFS" in wide.columns:
        for comparator in [m for m in METHODS if m != "NIBFS" and m in wide.columns]:
            statistic, p_value = _safe_wilcoxon(
                wide["NIBFS"], wide[comparator], alternative="greater"
            )
            difference = wide["NIBFS"] - wide[comparator]
            posthoc_rows.append(
                {
                    "Analysis": analysis,
                    "Test": "Paired Wilcoxon",
                    "Proposed": "NIBFS",
                    "Comparator": comparator,
                    "Alternative": "greater",
                    "N_repeats": len(wide),
                    "Statistic": statistic,
                    "P_value": p_value,
                    "Mean_difference": float(difference.mean()),
                    "Median_difference": float(difference.median()),
                }
            )

    if posthoc_rows:
        adjusted = bh_adjust([row["P_value"] for row in posthoc_rows])
        for row, adjusted_p in zip(posthoc_rows, adjusted):
            row["BH_adjusted_p"] = adjusted_p
        rows.extend(posthoc_rows)

    result = pd.DataFrame(rows)
    if "BH_adjusted_p" not in result.columns:
        result["BH_adjusted_p"] = np.nan
    return result


performance_tests = paired_comparison_table(
    oof_metrics,
    "ROC_AUC",
    "Repeated OOF ROC-AUC",
)
stability_tests = paired_comparison_table(
    repeat_stability,
    "Mean_Jaccard",
    "Repeated panel stability",
)
statistical_tests = pd.concat(
    [performance_tests, stability_tests], ignore_index=True
)
statistical_tests.to_csv(
    OUTPUT_DIR / "repeated_paired_statistical_tests.csv", index=False
)


# ------------------------------------------------------------
# 10. Audit, manifest, and figures
# ------------------------------------------------------------
complete_fold_count = runtime_table[["Repeat", "Fold"]].drop_duplicates().shape[0]
analysis_complete = complete_fold_count == TOTAL_FOLDS

# Every sample should be in validation exactly once per repeat.
assignment_counts = (
    fold_assignments.groupby(["Repeat", "Sample_ID"]).size().reset_index(name="Count")
)
assignment_ok = bool((assignment_counts["Count"] == 1).all())

repeated_audit = pd.DataFrame(
    [
        {
            "Analysis": "Repeated stratified cross-validation robustness analysis",
            "Development_samples": len(X_DEV),
            "Eligible_genes": X_DEV.shape[1],
            "Locked_internal_test_used": False,
            "Repeated_repeats": REPEATS,
            "Folds_per_repeat": FOLDS,
            "Expected_validation_folds": TOTAL_FOLDS,
            "Completed_validation_folds": complete_fold_count,
            "Complete": analysis_complete,
            "Each_sample_validated_once_per_repeat": assignment_ok,
            "k": FINAL_K,
            "NIBFS_weighting": "Equal 1:1 Borda aggregation",
            "Feature_selection_methods": "|".join(METHODS),
            "Classifier": CLASSIFIER,
            "Decision_threshold": DEFAULT_THRESHOLD,
            "Limma_scope": "Refitted on each fold-training partition only",
            "mRMR_scope": "Refitted on each fold-training partition only",
            "LASSO_scope": "Refitted on each fold-training partition only",
            "PPI_scope": "Fixed external STRING topology; restricted to eligible genes",
            "Frozen_primary_panel_changed": False,
            "Main_heldout_results_changed": False,
            "External_validation_results_changed": False,
        }
    ]
)
repeated_audit.to_csv(OUTPUT_DIR / "repeated_analysis_audit.csv", index=False)

manifest = {
    "analysis": "Repeated 10 x 5-fold CV robustness analysis",
    "project_directory": str(PROJECT_DIR),
    "output_directory": str(OUTPUT_DIR),
    "development_shape": list(X_DEV.shape),
    "class_counts": {
        str(k): int(v) for k, v in pd.Series(Y_DEV).value_counts().sort_index().items()
    },
    "repeats": REPEATS,
    "folds": FOLDS,
    "expected_outer_folds": TOTAL_FOLDS,
    "completed_outer_folds": complete_fold_count,
    "complete": analysis_complete,
    "k": FINAL_K,
    "methods": METHODS,
    "classifier": CLASSIFIER,
    "decision_threshold": DEFAULT_THRESHOLD,
    "random_state_base": RANDOM_STATE,
    "mrmr_candidate_size": MRMR_CANDIDATE_SIZE,
    "lasso_C": LASSO_C,
    "lasso_solver": LASSO_SOLVER,
    "lasso_max_iter": LASSO_MAX_ITER,
    "checkpoint_resume_enabled": True,
    "scientific_role": "Additional robustness analysis; does not redefine primary frozen panel",
    "output_files": sorted(p.name for p in OUTPUT_DIR.glob("*.csv")),
}
_save_json_atomic(manifest, OUTPUT_DIR / "repeated_run_manifest.json")

try:
    import matplotlib.pyplot as plt

    method_order = METHODS

    figure_data = [
        oof_metrics.loc[oof_metrics["Method"] == method, "ROC_AUC"].to_numpy(float)
        for method in method_order
    ]
    plt.figure(figsize=(7.2, 4.8))
    plt.boxplot(figure_data, tick_labels=method_order, showmeans=True)
    plt.ylabel("OOF ROC-AUC across repeats")
    plt.xlabel("Feature-selection method")
    plt.title("Repeated 10 x 5-fold predictive robustness (LR, k=20)")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "repeated_oof_auc_boxplot.png", dpi=600)
    plt.savefig(FIGURE_DIR / "repeated_oof_auc_boxplot.pdf")
    plt.close()

    stability_data = [
        repeat_stability.loc[
            repeat_stability["Method"] == method, "Mean_Jaccard"
        ].to_numpy(float)
        for method in method_order
    ]
    plt.figure(figsize=(7.2, 4.8))
    plt.boxplot(stability_data, tick_labels=method_order, showmeans=True)
    plt.ylabel("Mean pairwise Jaccard within repeat")
    plt.xlabel("Feature-selection method")
    plt.title("Repeated panel stability (10 repeats, 5 folds, k=20)")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "repeated_stability_boxplot.png", dpi=600)
    plt.savefig(FIGURE_DIR / "repeated_stability_boxplot.pdf")
    plt.close()
except Exception as figure_error:
    warnings.warn(f"Repeated figures were not generated: {figure_error}")

status_path = OUTPUT_DIR / (
    "REPEATED_ANALYSIS_COMPLETE.txt"
    if analysis_complete
    else "REPEATED_ANALYSIS_INCOMPLETE.txt"
)
status_text = (
    f"Completed folds: {complete_fold_count}/{TOTAL_FOLDS}\n"
    f"New folds in this session: {new_fold_count}\n"
    f"Output directory: {OUTPUT_DIR}\n"
)
status_path.write_text(status_text, encoding="utf-8")

# Remove the opposite status marker, if present.
opposite = OUTPUT_DIR / (
    "REPEATED_ANALYSIS_INCOMPLETE.txt"
    if analysis_complete
    else "REPEATED_ANALYSIS_COMPLETE.txt"
)
if opposite.exists():
    opposite.unlink()


# ------------------------------------------------------------
# 11. User-facing completion report
# ------------------------------------------------------------
print("\n" + "=" * 76)
if analysis_complete:
    print("REPEATED 10 x 5-FOLD CV COMPLETED")
else:
    print("REPEATED CV PARTIALLY COMPLETED — SAFE TO RESUME")
print("=" * 76)
print(f"Completed folds : {complete_fold_count}/{TOTAL_FOLDS}")
print(f"New this session: {new_fold_count}")
print("Output          :", OUTPUT_DIR)

print("\nRepeated OOF performance summary:")
display_columns = [
    column
    for column in [
        "Method",
        "ROC_AUC_count",
        "ROC_AUC_mean",
        "ROC_AUC_std",
        "ROC_AUC_median",
        "ROC_AUC_min",
        "ROC_AUC_max",
        "F1_mean",
        "MCC_mean",
        "Brier_score_mean",
    ]
    if column in oof_summary.columns
]
try:
    display(oof_summary[display_columns])
except NameError:
    print(oof_summary[display_columns].to_string(index=False))

print("\nRepeated stability summary:")
try:
    display(stability_summary)
except NameError:
    print(stability_summary.to_string(index=False))

print("\nMost recurrent NIBFS genes:")
nibfs_recurrence = gene_recurrence.loc[gene_recurrence["Method"] == "NIBFS"].head(30)
try:
    display(nibfs_recurrence)
except NameError:
    print(nibfs_recurrence.to_string(index=False))

print("\nGenerated aggregate files:")
for path in sorted(OUTPUT_DIR.glob("*")):
    if path.is_file():
        print(" -", path.name)

if not analysis_complete:
    print(
        "\nTo continue, keep the same project/output folder and rerun:\n"
        "  %run -i src/repeated_10x5_k20_lr_V2.py\n"
        "Completed folds will be skipped automatically."
    )
