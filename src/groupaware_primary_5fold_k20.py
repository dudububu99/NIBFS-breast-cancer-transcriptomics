# ============================================================
# GROUP-AWARE 5-FOLD SENSITIVITY ANALYSIS — NIBFS, k=20
# ============================================================
# Purpose
# -------
# Additional post-harmonization sensitivity analysis only.
# It keeps the original 608-sample development set and the original
# 17,220-gene harmonized matrix, but replaces the sample-level fold
# assignment with a pre-audited subject/group-aware five-fold assignment.
#
# This script DOES NOT overwrite the primary manuscript outputs, frozen
# panel, 152-sample internal assessment, external validations, LOCO, RWR,
# or permutation controls.
#
# Expected project layout
# -----------------------
#   <PROJECT>/src/groupaware_primary_5fold_k20.py
#   <PROJECT>/results/main/tables/harmonized_expression_matrix.csv.gz
#   <PROJECT>/results/main/tables/train_test_split_assignments.csv
#   <PROJECT>/results/main/tables/ppi_degree_table.csv
#   <PROJECT>/results/SUBJECT_GROUP_AUDIT/proposed_primary_groupaware_folds_608.csv
#
# Run from Colab / notebook:
#   GROUPAWARE_PROJECT_DIR = "/content/drive/MyDrive/<project>"
#   GROUPAWARE_DRY_RUN = True
#   %run -i src/groupaware_primary_5fold_k20.py
#
# If dry-run passes:
#   GROUPAWARE_DRY_RUN = False
#   GROUPAWARE_MAX_NEW_FOLDS = None
#   %run -i src/groupaware_primary_5fold_k20.py
#
# Optional:
#   GROUPAWARE_MAX_NEW_FOLDS = 1   # benchmark one fold, then resume later
#   GROUPAWARE_FORCE_RERUN = False
#   GROUPAWARE_FOLD_FILE = "/custom/path/proposed_primary_groupaware_folds_608.csv"
# ============================================================

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from time import perf_counter
import gc
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
except Exception as exc:
    raise ImportError(
        "lightgbm belum tersedia. Jalankan sel instalasi project lama terlebih dahulu."
    ) from exc


# ------------------------------------------------------------
# 0. Locked analysis configuration
# ------------------------------------------------------------

def _resolve_project_dir() -> Path:
    override = globals().get("GROUPAWARE_PROJECT_DIR")
    if override:
        p = Path(str(override)).expanduser().resolve()
        if (p / "src").exists() and (p / "results").exists():
            return p
        raise FileNotFoundError(f"GROUPAWARE_PROJECT_DIR tidak valid: {p}")

    script_file = globals().get("__file__")
    if script_file:
        p = Path(str(script_file)).expanduser().resolve().parent.parent
        if (p / "src").exists() and (p / "results").exists():
            return p

    start = Path.cwd().resolve()
    for p in [start, *start.parents]:
        if (p / "src").exists() and (p / "results").exists():
            return p
    raise FileNotFoundError(
        "Project tidak ditemukan. Tetapkan GROUPAWARE_PROJECT_DIR ke folder utama project."
    )


PROJECT_DIR = _resolve_project_dir()
OUTPUT_DIR = PROJECT_DIR / "results" / "GROUPAWARE_PRIMARY_5FOLD_K20"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

FINAL_K = 20
RANDOM_STATE = int(globals().get("GROUPAWARE_RANDOM_STATE", 42))
DRY_RUN = bool(globals().get("GROUPAWARE_DRY_RUN", False))
FORCE_RERUN = bool(globals().get("GROUPAWARE_FORCE_RERUN", False))
MAX_NEW_FOLDS = globals().get("GROUPAWARE_MAX_NEW_FOLDS", None)
if MAX_NEW_FOLDS is not None:
    MAX_NEW_FOLDS = int(MAX_NEW_FOLDS)

MRMR_CANDIDATE_SIZE = 1000
LASSO_C = 1.0
LASSO_SOLVER = "saga"
LASSO_MAX_ITER = 10000
DEFAULT_THRESHOLD = 0.5

METHODS = ["NIBFS", "DEG-only", "mRMR", "LASSO"]
CLASSIFIERS = ["LR", "RF", "LightGBM"]

# Locked model settings from the manuscript-facing NIBFS pipeline.
RF_N_ESTIMATORS = 500
RF_MAX_FEATURES = "sqrt"
LGBM_N_ESTIMATORS = 500
LGBM_LEARNING_RATE = 0.03
LGBM_NUM_LEAVES = 31
LGBM_SUBSAMPLE = 0.90
LGBM_COLSAMPLE_BYTREE = 0.90


def _latest_path(patterns: list[str]) -> Path | None:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(PROJECT_DIR.rglob(pattern))
    matches = [p for p in matches if p.is_file() and "GROUPAWARE_PRIMARY_5FOLD_K20" not in str(p)]
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lut = {str(c).strip().lower(): str(c) for c in df.columns}
    for c in candidates:
        if c.strip().lower() in lut:
            return lut[c.strip().lower()]
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
        raise ValueError(f"Label tidak dapat dipetakan: {sorted(s[np.isnan(out)].astype(str).unique())}")
    return out.astype(int)


# ------------------------------------------------------------
# 1. Load archived development matrix and audited folds
# ------------------------------------------------------------

expr_path = _latest_path(["harmonized_expression_matrix.csv.gz", "harmonized_expression_matrix.csv"])
split_path = _latest_path(["train_test_split_assignments.csv"])
ppi_path = _latest_path(["ppi_degree_table.csv"])

fold_override = globals().get("GROUPAWARE_FOLD_FILE")
if fold_override:
    fold_path = Path(str(fold_override)).expanduser().resolve()
else:
    preferred = PROJECT_DIR / "results" / "SUBJECT_GROUP_AUDIT" / "proposed_primary_groupaware_folds_608.csv"
    fold_path = preferred if preferred.exists() else _latest_path(["proposed_primary_groupaware_folds_608.csv"])

for label, path in [
    ("harmonized expression", expr_path),
    ("train/test assignment", split_path),
    ("PPI degree table", ppi_path),
    ("group-aware fold assignment", fold_path),
]:
    if path is None or not Path(path).exists():
        raise FileNotFoundError(f"File {label} tidak ditemukan: {path}")

print("Project      :", PROJECT_DIR)
print("Expression   :", expr_path)
print("Split        :", split_path)
print("PPI degree   :", ppi_path)
print("Group folds  :", fold_path)
print("Output       :", OUTPUT_DIR)

expr = pd.read_csv(expr_path)
sample_col = _find_column(expr, ["GSM_ID", "Sample_ID", "sample"])
if sample_col is None:
    sample_col = str(expr.columns[0])
expr[sample_col] = expr[sample_col].astype(str)
expr = expr.set_index(sample_col)
expr.index.name = "GSM_ID"
expr.columns = expr.columns.astype(str)

split = pd.read_csv(split_path)
gsm_col = _find_column(split, ["GSM_ID", "Sample_ID", "sample"])
set_col = _find_column(split, ["Set", "Subset", "Role"])
label_col = _find_column(split, ["Label_binary", "y", "Label"])
geo_col = _find_column(split, ["GEO_ID", "Cohort", "Dataset"])
text_label_col = _find_column(split, ["Label"])
if None in (gsm_col, set_col, label_col):
    raise KeyError(f"Kolom split tidak lengkap: {list(split.columns)}")

is_dev = split[set_col].astype(str).str.lower().str.contains(r"model-development|development|train", regex=True)
dev = split.loc[is_dev].copy()
dev[gsm_col] = dev[gsm_col].astype(str)
if dev[gsm_col].duplicated().any():
    raise RuntimeError("Development split mengandung GSM_ID duplikat.")
dev = dev.set_index(gsm_col)

missing = dev.index.difference(expr.index)
if len(missing):
    raise RuntimeError(f"{len(missing)} development GSM tidak ditemukan pada expression matrix.")

X_DEV = expr.loc[dev.index].copy()
Y_DEV = _labels_to_binary(dev[label_col])
META_DEV = pd.DataFrame(index=X_DEV.index)
META_DEV["GSM_ID"] = X_DEV.index
META_DEV["GEO_ID"] = dev[geo_col].astype(str).to_numpy() if geo_col else "unknown"
if text_label_col:
    META_DEV["Label"] = dev[text_label_col].astype(str).to_numpy()
else:
    META_DEV["Label"] = np.where(Y_DEV == 1, "Cancer", "Normal")

if X_DEV.shape != (608, 17220):
    raise RuntimeError(f"Expected development matrix (608, 17220), found {X_DEV.shape}.")
if dict(pd.Series(Y_DEV).value_counts().sort_index()) != {0: 299, 1: 309}:
    raise RuntimeError(
        "Development class counts berbeda dari manuscript lock: expected Normal=299, Cancer=309."
    )

folds = pd.read_csv(fold_path)
fgsm = _find_column(folds, ["GSM_ID", "Sample_ID"])
ffold = _find_column(folds, ["GroupAware_Validation_fold", "Validation_fold", "Fold"])
fgroup = _find_column(folds, ["Subject_Group", "Group_ID", "Subject_ID"])
if None in (fgsm, ffold, fgroup):
    raise KeyError(f"Kolom group-aware fold file tidak lengkap: {list(folds.columns)}")
folds[fgsm] = folds[fgsm].astype(str)
folds[ffold] = pd.to_numeric(folds[ffold], errors="raise").astype(int)
folds[fgroup] = folds[fgroup].astype(str)

if len(folds) != 608 or folds[fgsm].nunique() != 608:
    raise RuntimeError("Group-aware fold file harus memiliki tepat 608 GSM unik.")
if set(folds[fgsm]) != set(X_DEV.index):
    missing_from_folds = set(X_DEV.index) - set(folds[fgsm])
    extra_in_folds = set(folds[fgsm]) - set(X_DEV.index)
    raise RuntimeError(
        f"Fold IDs tidak identik dengan 608 development samples. missing={len(missing_from_folds)}, extra={len(extra_in_folds)}"
    )
if set(folds[ffold].unique()) != {1, 2, 3, 4, 5}:
    raise RuntimeError(f"Validation folds harus 1..5, found {sorted(folds[ffold].unique())}")

# Core leakage guard: one biological group must occupy exactly one validation fold.
group_fold_n = folds.groupby(fgroup)[ffold].nunique()
if not (group_fold_n == 1).all():
    bad = group_fold_n[group_fold_n > 1]
    raise RuntimeError(f"Subject/group leakage masih ada pada proposed folds: {len(bad)} groups split.")

# Align assignments to X_DEV order.
folds = folds.set_index(fgsm).loc[X_DEV.index].reset_index().rename(
    columns={fgsm: "GSM_ID", ffold: "Validation_fold", fgroup: "Subject_Group"}
)

# Verify class counts per validation fold.
check = folds.set_index("GSM_ID").copy()
check["Label_binary"] = pd.Series(Y_DEV, index=X_DEV.index).loc[check.index].to_numpy()
fold_balance = check.groupby("Validation_fold")["Label_binary"].agg(["size", "sum"])
fold_balance = fold_balance.rename(columns={"sum": "Cancer"})
fold_balance["Normal"] = fold_balance["size"] - fold_balance["Cancer"]
fold_balance = fold_balance.reset_index()
fold_balance.to_csv(OUTPUT_DIR / "groupaware_fold_balance.csv", index=False)

print("Development  :", X_DEV.shape)
print("Class counts : Normal=299, Cancer=309")
print("Group splits :", int((group_fold_n > 1).sum()), "(must be 0)")
print("Fold balance:")
print(fold_balance.to_string(index=False))

# Save exact assignment used by this run.
assignment_export = folds.copy()
assignment_export["GEO_ID"] = META_DEV.loc[assignment_export["GSM_ID"], "GEO_ID"].to_numpy()
assignment_export["Label"] = META_DEV.loc[assignment_export["GSM_ID"], "Label"].to_numpy()
assignment_export["Label_binary"] = pd.Series(Y_DEV, index=X_DEV.index).loc[assignment_export["GSM_ID"]].to_numpy()
assignment_export.to_csv(OUTPUT_DIR / "groupaware_fold_assignments_used.csv", index=False)

if DRY_RUN:
    audit = {
        "status": "DRY_RUN_PASS",
        "development_shape": list(X_DEV.shape),
        "class_counts": {"Normal": 299, "Cancer": 309},
        "unique_samples": int(folds["GSM_ID"].nunique()),
        "unique_groups": int(folds["Subject_Group"].nunique()),
        "groups_split_across_folds": int((group_fold_n > 1).sum()),
        "fold_file": str(fold_path),
    }
    (OUTPUT_DIR / "GROUPAWARE_DRY_RUN_PASS.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("\nDRY RUN PASS. Tidak ada model yang difit.")
    print("Set GROUPAWARE_DRY_RUN=False lalu run ulang untuk analisis penuh.")
    raise SystemExit(0)


# ------------------------------------------------------------
# 2. Feature-selection helpers — matched to archived repeated V2
# ------------------------------------------------------------

def run_limma_rpy2(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.conversion import localconverter
    except Exception as exc:
        raise ImportError("rpy2 / R limma belum siap. Jalankan setup project lama terlebih dahulu.") from exc

    ro.r("suppressPackageStartupMessages(library(limma))")
    expr_matrix = X.to_numpy(dtype=float).T
    y_arr = np.asarray(y, dtype=int)
    genes = X.columns.astype(str).tolist()
    with localconverter(ro.default_converter + numpy2ri.converter):
        ro.globalenv["expr_matrix_groupaware"] = expr_matrix
        ro.globalenv["group_vector_groupaware"] = y_arr
    ro.globalenv["gene_names_groupaware"] = ro.StrVector(genes)
    ro.r(
        """
        rownames(expr_matrix_groupaware) <- gene_names_groupaware
        group_factor_groupaware <- factor(group_vector_groupaware, levels=c(0,1), labels=c('Normal','Cancer'))
        design_groupaware <- model.matrix(~ group_factor_groupaware)
        fit_groupaware <- lmFit(expr_matrix_groupaware, design_groupaware)
        fit_groupaware <- eBayes(fit_groupaware)
        limma_result_groupaware <- topTable(
            fit_groupaware, coef=2, number=Inf, adjust.method='BH', sort.by='none'
        )
        limma_result_groupaware$Gene <- rownames(limma_result_groupaware)
        """
    )
    cols = list(ro.r("colnames(limma_result_groupaware)"))
    out = pd.DataFrame({c: list(ro.r(f"limma_result_groupaware${c}")) for c in cols})
    out = out.rename(columns={"adj.P.Val": "FDR", "P.Value": "P_value"})
    out["Gene"] = out["Gene"].astype(str)
    out["FDR"] = pd.to_numeric(out["FDR"], errors="coerce").fillna(1.0)
    out["logFC"] = pd.to_numeric(out["logFC"], errors="coerce").fillna(0.0)
    out["Stat_score"] = out["logFC"].abs() * (-np.log10(out["FDR"].clip(lower=1e-300)))
    out["Rank_stat"] = out["Stat_score"].rank(method="average", ascending=False)
    return out.sort_values(["Rank_stat", "Gene"]).reset_index(drop=True)


def prepare_ppi_rank_table(ppi_degree: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    gene_col = _find_column(ppi_degree, ["Gene", "Symbol", "Gene_symbol"])
    degree_col = _find_column(ppi_degree, ["Degree", "degree"])
    norm_col = _find_column(ppi_degree, ["Normalized_degree", "normalized_degree"])
    if gene_col is None or degree_col is None:
        raise KeyError(f"PPI table tidak memiliki Gene/Degree: {list(ppi_degree.columns)}")
    ppi = ppi_degree[[gene_col, degree_col] + ([norm_col] if norm_col and norm_col not in [gene_col, degree_col] else [])].copy()
    rename = {gene_col: "Gene", degree_col: "Degree"}
    if norm_col: rename[norm_col] = "Normalized_degree"
    ppi = ppi.rename(columns=rename)
    ppi["Gene"] = ppi["Gene"].astype(str)
    ppi["Degree"] = pd.to_numeric(ppi["Degree"], errors="coerce").fillna(0.0)
    if "Normalized_degree" not in ppi.columns:
        ppi["Normalized_degree"] = ppi["Degree"]
    else:
        ppi["Normalized_degree"] = pd.to_numeric(ppi["Normalized_degree"], errors="coerce").fillna(0.0)
    ppi = ppi.groupby("Gene", as_index=False).agg(Degree=("Degree", "max"), Normalized_degree=("Normalized_degree", "max"))
    out = pd.DataFrame({"Gene": pd.Index(pd.Series(genes).astype(str).drop_duplicates())}).merge(ppi, on="Gene", how="left")
    out[["Degree", "Normalized_degree"]] = out[["Degree", "Normalized_degree"]].fillna(0.0)
    out["Rank_topo"] = out["Normalized_degree"].rank(method="average", ascending=False)
    return out.sort_values(["Rank_topo", "Gene"]).reset_index(drop=True)


def nibfs_rank(limma: pd.DataFrame, ppi_rank: pd.DataFrame, gene_universe: list[str]) -> pd.DataFrame:
    p = len(gene_universe)
    out = limma.merge(ppi_rank[["Gene", "Degree", "Normalized_degree", "Rank_topo"]], on="Gene", how="left")
    out[["Degree", "Normalized_degree"]] = out[["Degree", "Normalized_degree"]].fillna(0.0)
    out["Rank_topo"] = out["Rank_topo"].fillna(float(p))
    out["Borda_stat"] = p - out["Rank_stat"] + 1
    out["Borda_topo"] = p - out["Rank_topo"] + 1
    out["Borda_score"] = out["Borda_stat"] + out["Borda_topo"]
    out = out.sort_values(["Borda_score", "Rank_stat", "Gene"], ascending=[False, True, True]).reset_index(drop=True)
    out["Rank_NIBFS"] = np.arange(1, len(out) + 1)
    return out


def select_mrmr_features(X: pd.DataFrame, y: np.ndarray, k: int, seed: int) -> pd.DataFrame:
    relevance = pd.Series(mutual_info_classif(X, y, random_state=seed), index=X.columns)
    candidates = relevance.sort_values(ascending=False).head(min(MRMR_CANDIDATE_SIZE, X.shape[1])).index.tolist()
    corr = X[candidates].corr().abs().fillna(0.0)
    selected, rows = [], []
    while len(selected) < min(k, len(candidates)):
        best_gene, best_tuple = None, None
        for gene in candidates:
            if gene in selected:
                continue
            redundancy = float(corr.loc[gene, selected].mean()) if selected else 0.0
            score = float(relevance.loc[gene] - redundancy)
            candidate = (score, float(relevance.loc[gene]), -redundancy)
            if best_tuple is None or candidate > best_tuple:
                best_gene, best_tuple = gene, candidate
        if best_gene is None:
            break
        selected.append(str(best_gene))
        rows.append({
            "Gene": str(best_gene),
            "Selection_Order": len(selected),
            "mRMR_score": best_tuple[0],
            "Relevance": best_tuple[1],
            "Redundancy": -best_tuple[2],
        })
    return pd.DataFrame(rows)


def lasso_gene_ranking(X: pd.DataFrame, y: np.ndarray, seed: int) -> tuple[pd.DataFrame, bool]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(
        C=LASSO_C,
        penalty="l1",
        solver=LASSO_SOLVER,
        class_weight="balanced",
        max_iter=LASSO_MAX_ITER,
        n_jobs=-1,
        random_state=seed,
    )
    converged = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X_scaled, y)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    out = pd.DataFrame({"Gene": X.columns.astype(str), "Coefficient": model.coef_.ravel()})
    out["Abs_Coefficient"] = out["Coefficient"].abs()
    out = out.sort_values(["Abs_Coefficient", "Gene"], ascending=[False, True]).reset_index(drop=True)
    out["Rank_LASSO"] = np.arange(1, len(out) + 1)
    return out, converged


def select_top_k(table: pd.DataFrame, rank_col: str) -> list[str]:
    genes = table.sort_values([rank_col, "Gene"]).head(FINAL_K)["Gene"].astype(str).tolist()
    if len(genes) != FINAL_K or len(set(genes)) != FINAL_K:
        raise RuntimeError(f"{rank_col} tidak menghasilkan {FINAL_K} gen unik.")
    return genes


PPI_DEGREE = pd.read_csv(ppi_path)
PPI_RANK = prepare_ppi_rank_table(PPI_DEGREE, X_DEV.columns.tolist())


# ------------------------------------------------------------
# 3. Models and metrics
# ------------------------------------------------------------

def create_models(seed: int) -> dict:
    return {
        "LR": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                penalty="l2", solver="lbfgs", max_iter=5000,
                class_weight="balanced", random_state=seed,
            )),
        ]),
        "RF": RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_features=RF_MAX_FEATURES,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=LGBM_N_ESTIMATORS,
            learning_rate=LGBM_LEARNING_RATE,
            num_leaves=LGBM_NUM_LEAVES,
            subsample=LGBM_SUBSAMPLE,
            colsample_bytree=LGBM_COLSAMPLE_BYTREE,
            objective="binary",
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        ),
    }


def classification_metrics(y_true: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p, dtype=float)
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
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def _save_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _save_json_atomic(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def checkpoint_files(fold: int) -> dict[str, Path]:
    prefix = CHECKPOINT_DIR / f"fold_{fold:02d}"
    return {
        "metrics": prefix.with_name(prefix.name + "_metrics.csv"),
        "predictions": prefix.with_name(prefix.name + "_predictions.csv"),
        "panels": prefix.with_name(prefix.name + "_panels.csv"),
        "lasso": prefix.with_name(prefix.name + "_lasso_audit.csv"),
        "runtime": prefix.with_name(prefix.name + "_runtime.json"),
        "done": prefix.with_name(prefix.name + "_DONE.json"),
    }


def checkpoint_complete(fold: int) -> bool:
    fs = checkpoint_files(fold)
    return all(fs[k].exists() for k in ["metrics", "predictions", "panels", "lasso", "runtime", "done"])


if FORCE_RERUN:
    for p in CHECKPOINT_DIR.glob("*"):
        if p.is_file():
            p.unlink()

# ------------------------------------------------------------
# 4. Execute five preassigned group-aware folds
# ------------------------------------------------------------
new_folds = 0
for fold in range(1, 6):
    if checkpoint_complete(fold):
        print(f"Fold {fold}/5: checkpoint found — skipped")
        continue
    if MAX_NEW_FOLDS is not None and new_folds >= MAX_NEW_FOLDS:
        break

    started = perf_counter()
    fold_seed = RANDOM_STATE + fold
    val_ids = folds.loc[folds["Validation_fold"] == fold, "GSM_ID"].astype(str).tolist()
    fit_ids = folds.loc[folds["Validation_fold"] != fold, "GSM_ID"].astype(str).tolist()

    # Explicit subject-isolation assertion for each train/validation split.
    val_groups = set(folds.loc[folds["Validation_fold"] == fold, "Subject_Group"].astype(str))
    fit_groups = set(folds.loc[folds["Validation_fold"] != fold, "Subject_Group"].astype(str))
    overlap_groups = val_groups & fit_groups
    if overlap_groups:
        raise RuntimeError(f"Fold {fold}: {len(overlap_groups)} Subject_Group overlap train/validation.")

    X_fit, X_val = X_DEV.loc[fit_ids], X_DEV.loc[val_ids]
    y_series = pd.Series(Y_DEV, index=X_DEV.index)
    y_fit = y_series.loc[fit_ids].to_numpy(int)
    y_val = y_series.loc[val_ids].to_numpy(int)

    print("\n" + "=" * 76)
    print(f"GROUP-AWARE FOLD {fold}/5")
    print(f"training={X_fit.shape}, validation={X_val.shape}, subject-group overlap=0")

    print("  [1/5] limma on fold-training samples ...")
    limma = run_limma_rpy2(X_fit, y_fit)
    print("  [2/5] NIBFS rank ...")
    nibfs = nibfs_rank(limma, PPI_RANK, X_fit.columns.tolist())
    print("  [3/5] mRMR ...")
    mrmr = select_mrmr_features(X_fit, y_fit, FINAL_K, fold_seed)
    print("  [4/5] LASSO ...")
    lasso, lasso_converged = lasso_gene_ranking(X_fit, y_fit, fold_seed)

    panels = {
        "NIBFS": select_top_k(nibfs, "Rank_NIBFS"),
        "DEG-only": select_top_k(limma, "Rank_stat"),
        "mRMR": select_top_k(mrmr, "Selection_Order"),
        "LASSO": select_top_k(lasso, "Rank_LASSO"),
    }

    print("  [5/5] LR / RF / LightGBM validation ...")
    metric_rows, prediction_rows, panel_rows = [], [], []
    models = create_models(fold_seed)

    lookup = {
        "NIBFS": nibfs.set_index("Gene"),
        "DEG-only": limma.set_index("Gene"),
        "mRMR": mrmr.set_index("Gene"),
        "LASSO": lasso.set_index("Gene"),
    }

    for method in METHODS:
        genes = panels[method]
        for clf, model in models.items():
            model.fit(X_fit[genes], y_fit)
            prob = model.predict_proba(X_val[genes])[:, 1]
            metric_rows.append({
                "Fold": fold,
                "Method": method,
                "Classifier": clf,
                "k": FINAL_K,
                "Training_samples": len(fit_ids),
                "Validation_samples": len(val_ids),
                "Subject_group_overlap": 0,
                **classification_metrics(y_val, prob, DEFAULT_THRESHOLD),
            })
            for sid, truth, pr in zip(val_ids, y_val, prob):
                prediction_rows.append({
                    "Fold": fold, "Method": method, "Classifier": clf,
                    "k": FINAL_K, "Sample_ID": sid,
                    "True_Label": int(truth), "Probability": float(pr),
                })

        table = lookup[method]
        for rank, gene in enumerate(genes, 1):
            row = {"Fold": fold, "Method": method, "k": FINAL_K, "Selection_rank": rank, "Gene": gene}
            if gene in table.index:
                details = table.loc[gene]
                if isinstance(details, pd.DataFrame): details = details.iloc[0]
                for c in [
                    "logFC", "FDR", "Stat_score", "Rank_stat", "Degree", "Normalized_degree",
                    "Rank_topo", "Borda_score", "Rank_NIBFS", "mRMR_score", "Relevance",
                    "Redundancy", "Coefficient", "Abs_Coefficient", "Rank_LASSO"
                ]:
                    if c in details.index:
                        row[c] = details[c]
            panel_rows.append(row)

    nonzero_count = int((lasso["Abs_Coefficient"] > 0).sum())
    rank20_abs = float(lasso.iloc[FINAL_K - 1]["Abs_Coefficient"])
    lasso_audit = pd.DataFrame([{
        "Fold": fold,
        "Converged": bool(lasso_converged),
        "Nonzero_coefficients": nonzero_count,
        "Rank20_abs_coefficient": rank20_abs,
        "Top20_all_nonzero": bool((lasso.head(FINAL_K)["Abs_Coefficient"] > 0).all()),
    }])

    elapsed = perf_counter() - started
    fs = checkpoint_files(fold)
    _save_csv_atomic(pd.DataFrame(metric_rows), fs["metrics"])
    _save_csv_atomic(pd.DataFrame(prediction_rows), fs["predictions"])
    _save_csv_atomic(pd.DataFrame(panel_rows), fs["panels"])
    _save_csv_atomic(lasso_audit, fs["lasso"])
    _save_json_atomic({
        "Fold": fold, "Elapsed_seconds": elapsed,
        "Training_samples": len(fit_ids), "Validation_samples": len(val_ids),
        "Subject_group_overlap": 0, "LASSO_converged": bool(lasso_converged),
    }, fs["runtime"])
    _save_json_atomic({"status": "complete", "fold": fold}, fs["done"])
    print(f"  completed in {elapsed/60:.2f} minutes")
    new_folds += 1

    del X_fit, X_val, y_fit, y_val, limma, nibfs, mrmr, lasso
    gc.collect()


# ------------------------------------------------------------
# 5. Aggregate completed folds
# ------------------------------------------------------------
completed = [f for f in range(1, 6) if checkpoint_complete(f)]
if not completed:
    raise RuntimeError("Belum ada fold group-aware yang selesai.")

metrics = pd.concat([pd.read_csv(checkpoint_files(f)["metrics"]) for f in completed], ignore_index=True)
predictions = pd.concat([pd.read_csv(checkpoint_files(f)["predictions"]) for f in completed], ignore_index=True)
panels = pd.concat([pd.read_csv(checkpoint_files(f)["panels"]) for f in completed], ignore_index=True)
lasso_audit = pd.concat([pd.read_csv(checkpoint_files(f)["lasso"]) for f in completed], ignore_index=True)

metrics.to_csv(OUTPUT_DIR / "groupaware_fold_metrics.csv", index=False)
predictions.to_csv(OUTPUT_DIR / "groupaware_predictions.csv", index=False)
panels.to_csv(OUTPUT_DIR / "groupaware_selected_panels.csv", index=False)
lasso_audit.to_csv(OUTPUT_DIR / "groupaware_lasso_nonzero_audit.csv", index=False)

# Fold-level descriptive performance summary.
metric_cols = ["ROC_AUC", "Accuracy", "Balanced_accuracy", "Sensitivity", "Specificity", "Precision", "F1", "MCC", "Brier_score"]
summary = metrics.groupby(["Method", "Classifier", "k"])[metric_cols].agg(["count", "mean", "std", "median", "min", "max"]).reset_index()
summary.columns = ["_".join(str(x) for x in c if str(x)) if isinstance(c, tuple) else str(c) for c in summary.columns]
summary.to_csv(OUTPUT_DIR / "groupaware_fold_performance_summary.csv", index=False)

# OOF performance is valid only when all five folds are complete.
if len(completed) == 5:
    oof_rows = []
    for (method, clf), g in predictions.groupby(["Method", "Classifier"]):
        if g["Sample_ID"].nunique() != 608 or len(g) != 608:
            raise RuntimeError(f"OOF coverage invalid for {method}/{clf}: n={len(g)}, unique={g['Sample_ID'].nunique()}")
        oof_rows.append({
            "Method": method, "Classifier": clf, "k": FINAL_K, "Samples": len(g),
            **classification_metrics(g["True_Label"].to_numpy(int), g["Probability"].to_numpy(float), DEFAULT_THRESHOLD),
        })
    pd.DataFrame(oof_rows).sort_values(["Method", "Classifier"]).to_csv(
        OUTPUT_DIR / "groupaware_oof_metrics.csv", index=False
    )

    # Jaccard stability across the five group-aware selected panels.
    pair_rows, stab_rows = [], []
    for method, g in panels.groupby("Method"):
        fold_sets = {
            int(f): set(x.sort_values("Selection_rank")["Gene"].astype(str))
            for f, x in g.groupby("Fold")
        }
        vals = []
        for f1, f2 in combinations(sorted(fold_sets), 2):
            a, b = fold_sets[f1], fold_sets[f2]
            j = len(a & b) / len(a | b)
            vals.append(j)
            pair_rows.append({
                "Method": method, "k": FINAL_K, "Fold_1": f1, "Fold_2": f2,
                "Intersection": len(a & b), "Union": len(a | b), "Jaccard": j,
            })
        stab_rows.append({
            "Method": method, "k": FINAL_K, "Fold_panels": len(fold_sets),
            "Pairwise_comparisons": len(vals), "Mean_Jaccard": float(np.mean(vals)),
            "SD_Jaccard": float(np.std(vals, ddof=1)), "Median_Jaccard": float(np.median(vals)),
            "Minimum_Jaccard": float(np.min(vals)), "Maximum_Jaccard": float(np.max(vals)),
        })
    pairwise = pd.DataFrame(pair_rows).sort_values(["Method", "Fold_1", "Fold_2"])
    stability = pd.DataFrame(stab_rows).sort_values("Method")
    pairwise.to_csv(OUTPUT_DIR / "groupaware_pairwise_jaccard.csv", index=False)
    stability.to_csv(OUTPUT_DIR / "groupaware_stability_summary.csv", index=False)

    # Nogueira stability estimator from the 5 x p selection-indicator matrix.
    # With fixed k, denominator is (k/p)*(1-k/p), and per-feature variance uses ddof=1.
    nog_rows = []
    p = X_DEV.shape[1]
    for method, g in panels.groupby("Method"):
        Z = np.zeros((5, p), dtype=float)
        gene_to_idx = {gene: i for i, gene in enumerate(X_DEV.columns.astype(str))}
        for f in range(1, 6):
            selected = g.loc[g["Fold"] == f, "Gene"].astype(str)
            for gene in selected:
                Z[f - 1, gene_to_idx[gene]] = 1.0
        avg_k = float(Z.sum(axis=1).mean())
        q = avg_k / p
        denom = q * (1.0 - q)
        mean_var = float(np.var(Z, axis=0, ddof=1).mean())
        phi = float(1.0 - mean_var / denom) if denom > 0 else np.nan
        nog_rows.append({
            "Method": method, "Selections_M": 5, "Eligible_features_p": p,
            "Mean_selected_k": avg_k, "Nogueira_stability": phi,
        })
    pd.DataFrame(nog_rows).sort_values("Method").to_csv(
        OUTPUT_DIR / "groupaware_nogueira_stability.csv", index=False
    )

    # Final subject isolation audit.
    per_fold = []
    for fold in range(1, 6):
        vg = set(folds.loc[folds.Validation_fold == fold, "Subject_Group"])
        tg = set(folds.loc[folds.Validation_fold != fold, "Subject_Group"])
        per_fold.append({
            "Fold": fold,
            "Validation_samples": int((folds.Validation_fold == fold).sum()),
            "Validation_groups": len(vg),
            "Training_groups": len(tg),
            "Subject_group_overlap": len(vg & tg),
        })
    pd.DataFrame(per_fold).to_csv(OUTPUT_DIR / "groupaware_subject_isolation_audit.csv", index=False)

complete = len(completed) == 5
manifest = {
    "analysis": "Post-harmonization subject/group-aware 5-fold sensitivity analysis",
    "scientific_role": "Sensitivity analysis only; does not replace locked primary results",
    "development_shape": list(X_DEV.shape),
    "development_class_counts": {"Normal": 299, "Cancer": 309},
    "k": FINAL_K,
    "methods": METHODS,
    "classifiers": CLASSIFIERS,
    "fold_assignment_file": str(fold_path),
    "unique_subject_groups": int(folds.Subject_Group.nunique()),
    "groups_split_across_folds": int((group_fold_n > 1).sum()),
    "completed_folds": len(completed),
    "complete": complete,
    "preprocessing_scope": "Uses archived post-harmonization 608 x 17220 development matrix; preprocessing is not refitted within fold",
    "feature_selection_scope": "limma, NIBFS statistical component, mRMR, and LASSO refitted on each fold-training partition",
    "ppi_scope": "Fixed STRING-derived degree ranking",
    "locked_primary_outputs_modified": False,
}
_save_json_atomic(manifest, OUTPUT_DIR / "groupaware_run_manifest.json")

status = OUTPUT_DIR / ("GROUPAWARE_ANALYSIS_COMPLETE.txt" if complete else "GROUPAWARE_ANALYSIS_INCOMPLETE.txt")
status.write_text(f"Completed folds: {len(completed)}/5\nOutput: {OUTPUT_DIR}\n", encoding="utf-8")

print("\n" + "=" * 76)
print("GROUP-AWARE 5-FOLD STATUS")
print("=" * 76)
print(f"Completed folds: {len(completed)}/5")
print("Output:", OUTPUT_DIR)
if complete:
    print("Subject/group overlap across every train/validation split: 0")
    print("Generated: Jaccard, Nogueira, LR/RF/LightGBM OOF metrics, LASSO audit, and isolation audit.")
else:
    print("Safe to resume by rerunning the same script. Completed folds will be skipped.")
