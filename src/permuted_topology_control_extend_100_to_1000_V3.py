# ============================================================
# GENE-LABEL-PERMUTED STRING-TOPOLOGY CONTROL
# EXTENSION FROM 100 TO 1,000 fixed topology permutations x repeated 10 x 5-fold CV
# k=20, logistic regression
# ============================================================
#
# Scientific question
# -------------------
# Does the observed STRING-derived degree ranking stabilize NIBFS beyond the
# generic effect of adding any fixed ranking to the fold-local statistical
# ranking?
#
# Null control
# ------------
# For each permutation, the observed STRING normalized-degree values are
# randomly reassigned to genes. This preserves the complete degree-value
# distribution (including ties and zeros) while destroying the biological
# gene-to-topology correspondence. One permuted mapping is fixed across all
# 50 folds, so each null control is itself a fixed anchor.
#
# Important
# ---------
# - Uses only the 608-sample model-development set.
# - Uses exactly the same 10 x 5 fold definitions as repeated_10x5_k20_lr_V2.py.
# - Re-fits limma on each fold-training partition.
# - Verifies that reconstructed real-STRING NIBFS top-20 panels exactly match
#   the completed repeated-CV checkpoints before evaluating permutations.
# - Does not redefine the frozen top-20 panel.
# - Does not alter held-out, external-validation, LOCO, or Paper-2 results.
# - Checkpoints permit safe resume.
#
# Run from Colab after copying this file to PROJECT_DIR/src:
#
#   PERMUTED_PROJECT_DIR = str(PROJECT_DIR)
#   PERMUTED_N_PERMUTATIONS = 1000
#   PERMUTED_REUSE_EXISTING_100 = True
#   PERMUTED_MAX_NEW_FOLDS = 1   # first benchmark; then use 4 or 5
#   PERMUTED_FORCE_RERUN = False
#   %run -i src/permuted_topology_control_extend_100_to_1000_V3.py
#
# For a complete uninterrupted run, set PERMUTED_MAX_NEW_FOLDS = None.
# ============================================================

from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path
from time import perf_counter
import gc
import hashlib
import json
import os
import shutil
import warnings

import numpy as np
import pandas as pd
import yaml

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
from scipy.stats import beta as beta_distribution


# ------------------------------------------------------------
# 0. Configuration
# ------------------------------------------------------------

def _perm_resolve_project_dir() -> Path:
    """Resolve the NIBFS project directory."""
    override = globals().get("PERMUTED_PROJECT_DIR")
    if override:
        candidate = Path(str(override)).expanduser().resolve()
        if (candidate / "src").exists() and (candidate / "results").exists():
            return candidate
        raise FileNotFoundError(
            f"PERMUTED_PROJECT_DIR tidak valid: {candidate}"
        )

    script_file = globals().get("__file__")
    if script_file:
        candidate = Path(str(script_file)).expanduser().resolve().parent.parent
        if (candidate / "src").exists() and (candidate / "results").exists():
            return candidate

    start = Path.cwd().resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "src").exists() and (candidate / "results").exists():
            return candidate

    raise FileNotFoundError(
        "Folder proyek tidak dapat ditentukan. Tetapkan PERMUTED_PROJECT_DIR "
        "ke folder utama proyek sebelum menjalankan script."
    )


PROJECT_DIR = _perm_resolve_project_dir()
CONFIG_PATH = PROJECT_DIR / "config.yaml"
CFG: dict = {}
if CONFIG_PATH.exists():
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        CFG = yaml.safe_load(handle) or {}

RANDOM_STATE = int(
    globals().get(
        "PERMUTED_RANDOM_STATE",
        CFG.get("project", {}).get("random_state", 42),
    )
)
FINAL_K = int(
    globals().get(
        "PERMUTED_FINAL_K",
        CFG.get("project", {}).get("final_k", 20),
    )
)
REPEATS = int(globals().get("PERMUTED_REPEATS", 10))
FOLDS = int(globals().get("PERMUTED_FOLDS", 5))
N_PERMUTATIONS = int(globals().get("PERMUTED_N_PERMUTATIONS", 1000))
PERMUTATION_SEED_BASE = int(
    globals().get("PERMUTED_SEED_BASE", 20260730)
)
MAX_NEW_FOLDS = globals().get("PERMUTED_MAX_NEW_FOLDS", None)
FORCE_RERUN = bool(globals().get("PERMUTED_FORCE_RERUN", False))
DEFAULT_THRESHOLD = float(
    globals().get(
        "PERMUTED_THRESHOLD",
        CFG.get("models", {}).get("default_decision_threshold", 0.5),
    )
)
DISPLAY_FINAL_FIGURE = bool(
    globals().get("PERMUTED_DISPLAY_FINAL_FIGURE", True)
)
REUSE_EXISTING_100 = bool(
    globals().get("PERMUTED_REUSE_EXISTING_100", True)
)
BASE_PERMUTATIONS = 100

if FINAL_K != 20:
    raise ValueError(f"Script ini dikunci untuk k=20, ditemukan k={FINAL_K}.")
if REPEATS != 10 or FOLDS != 5:
    raise ValueError(
        f"Kontrol paper harus memakai 10 x 5, ditemukan {REPEATS} x {FOLDS}."
    )
if N_PERMUTATIONS != 1000:
    raise ValueError(
        "Extension V3 ini dikunci untuk total 1,000 permutasi. "
        f"Ditemukan PERMUTED_N_PERMUTATIONS={N_PERMUTATIONS}."
    )
if MAX_NEW_FOLDS is not None:
    MAX_NEW_FOLDS = int(MAX_NEW_FOLDS)
    if MAX_NEW_FOLDS < 1:
        raise ValueError("PERMUTED_MAX_NEW_FOLDS harus None atau >= 1.")

REPEATED_DIR = PROJECT_DIR / "results" / "REPEATED_10X5_K20_LR"
OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / f"PERMUTED_TOPOLOGY_CONTROL_{N_PERMUTATIONS}X_10X5_K20_LR"
)
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FIGURE_DIR = OUTPUT_DIR / "figures"
LOCK_FILE = OUTPUT_DIR / "permuted_control_configuration_lock.json"
RANK_MATRIX_FILE = OUTPUT_DIR / "permuted_topology_rank_matrix.npz"
PERMUTATION_MANIFEST_FILE = OUTPUT_DIR / "permutation_manifest.csv"
BASE_100_DIR = (
    PROJECT_DIR
    / "results"
    / "PERMUTED_TOPOLOGY_CONTROL_100X_10X5_K20_LR"
)
BASE_100_CHECKPOINT_DIR = BASE_100_DIR / "checkpoints"
BASE_100_MANIFEST_FILE = BASE_100_DIR / "permutation_manifest.csv"

if FORCE_RERUN and OUTPUT_DIR.exists():
    print(f"FORCE_RERUN=True: removing only {OUTPUT_DIR}", flush=True)
    shutil.rmtree(OUTPUT_DIR)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 1. Generic helpers and data recovery
# ------------------------------------------------------------

def _perm_latest_path(patterns: list[str]) -> Path | None:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(PROJECT_DIR.rglob(pattern))
    matches = [p for p in matches if p.is_file()]
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _perm_find_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {str(c).strip().lower(): str(c) for c in frame.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lookup:
            return lookup[key]
    return None


def _perm_labels_to_binary(values) -> np.ndarray:
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


def _perm_load_development_from_disk() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    expression_path = _perm_latest_path(
        ["harmonized_expression_matrix.csv.gz", "harmonized_expression_matrix.csv"]
    )
    split_path = _perm_latest_path(
        ["train_test_split_assignments.csv", "discovery_train_test_assignments.csv"]
    )
    if expression_path is None or split_path is None:
        raise RuntimeError(
            "Development matrix tidak aktif dan file harmonized-expression/split "
            "tidak ditemukan. Jalankan hanya sel load hasil utama; jangan rerun "
            "seluruh pipeline."
        )

    print("Loading development matrix:", expression_path, flush=True)
    expression = pd.read_csv(expression_path)
    sample_col = _perm_find_column(expression, ["GSM_ID", "Sample_ID", "sample"])
    if sample_col is None:
        sample_col = str(expression.columns[0])
    expression[sample_col] = expression[sample_col].astype(str)
    expression = expression.set_index(sample_col)

    split = pd.read_csv(split_path)
    gsm_col = _perm_find_column(split, ["GSM_ID", "Sample_ID", "sample"])
    set_col = _perm_find_column(split, ["Set", "Subset", "Role"])
    label_col = _perm_find_column(split, ["Label_binary", "y", "Label"])
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

    missing = dev.index.difference(expression.index)
    if len(missing):
        raise RuntimeError(
            f"{len(missing)} development sample tidak ditemukan pada matrix ekspresi."
        )

    X = expression.loc[dev.index].copy()
    y = _perm_labels_to_binary(dev[label_col])
    metadata = dev.reset_index().rename(columns={gsm_col: "GSM_ID"})
    metadata.index = X.index
    return X, y, metadata


def _perm_load_ppi_degree_from_disk() -> pd.DataFrame:
    path = _perm_latest_path(["ppi_degree_table.csv"])
    if path is None:
        raise RuntimeError("ppi_degree_table.csv tidak ditemukan.")
    print("Loading PPI degree table:", path, flush=True)
    return pd.read_csv(path)


def _perm_hash_text_sequence(values) -> str:
    payload = "\n".join(map(str, values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _perm_hash_float_array(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype="<f8")
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def _perm_save_json_atomic(data: dict, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def _perm_save_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".gz" or path.name.endswith(".csv.gz"):
        frame.to_csv(temp, index=False, compression="gzip")
    else:
        frame.to_csv(temp, index=False)
    temp.replace(path)


def _perm_save_npz_atomic(path: Path, **arrays) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temp.replace(path)


if "X_train" in globals() and "y_train" in globals():
    X_DEV = pd.DataFrame(globals()["X_train"]).copy()
    Y_DEV = np.asarray(globals()["y_train"], dtype=int)
    if "metadata_train" in globals():
        META_DEV = pd.DataFrame(globals()["metadata_train"]).copy()
    else:
        META_DEV = pd.DataFrame(index=X_DEV.index)
else:
    X_DEV, Y_DEV, META_DEV = _perm_load_development_from_disk()

if "ppi_degree" in globals():
    PPI_DEGREE = pd.DataFrame(globals()["ppi_degree"]).copy()
else:
    PPI_DEGREE = _perm_load_ppi_degree_from_disk()

X_DEV.index = X_DEV.index.astype(str)
X_DEV.columns = X_DEV.columns.astype(str)
Y_DEV = np.asarray(Y_DEV, dtype=int)

if X_DEV.shape != (608, 17220):
    raise RuntimeError(
        "Permuted control harus memakai development matrix (608, 17220), "
        f"tetapi ditemukan {X_DEV.shape}."
    )
if len(Y_DEV) != len(X_DEV) or set(np.unique(Y_DEV)) != {0, 1}:
    raise RuntimeError("Development labels tidak valid.")
if X_DEV.index.duplicated().any() or X_DEV.columns.duplicated().any():
    raise RuntimeError("Sample ID atau gene symbol tidak unik.")

GENES = X_DEV.columns.to_numpy(dtype=str)
N_GENES = len(GENES)
GENE_HASH = _perm_hash_text_sequence(GENES)

print("PROJECT_DIR          :", PROJECT_DIR)
print("REPEATED_DIR         :", REPEATED_DIR)
print("OUTPUT_DIR           :", OUTPUT_DIR)
print("Development data     :", X_DEV.shape)
print("Class counts         :", dict(pd.Series(Y_DEV).value_counts().sort_index()))
print("Permutations         :", N_PERMUTATIONS)
print("Repeated design      :", f"{REPEATS} x {FOLDS} = {REPEATS * FOLDS} folds")
print("Final k / classifier :", FINAL_K, "/ LR")


# ------------------------------------------------------------
# 2. Load completed repeated-CV reference
# ------------------------------------------------------------

REQUIRED_REPEATED_FILES = {
    "panels": REPEATED_DIR / "repeated_selected_panels.csv",
    "oof": REPEATED_DIR / "repeated_oof_metrics_by_repeat.csv",
    "stability": REPEATED_DIR / "repeated_stability_by_repeat.csv",
    "assignments": REPEATED_DIR / "repeated_fold_assignments.csv",
}
missing_repeated = [str(p) for p in REQUIRED_REPEATED_FILES.values() if not p.exists()]
if missing_repeated:
    raise FileNotFoundError(
        "Repeated 10x5 reference belum lengkap. File yang hilang:\n- "
        + "\n- ".join(missing_repeated)
    )

REPEATED_PANELS = pd.read_csv(REQUIRED_REPEATED_FILES["panels"])
REPEATED_OOF = pd.read_csv(REQUIRED_REPEATED_FILES["oof"])
REPEATED_STABILITY = pd.read_csv(REQUIRED_REPEATED_FILES["stability"])
REPEATED_ASSIGNMENTS = pd.read_csv(REQUIRED_REPEATED_FILES["assignments"])

expected_fold_keys = {
    (repeat, fold)
    for repeat in range(1, REPEATS + 1)
    for fold in range(1, FOLDS + 1)
}
observed_fold_keys = set(
    REPEATED_PANELS.loc[
        REPEATED_PANELS["Method"].astype(str) == "NIBFS", ["Repeat", "Fold"]
    ].itertuples(index=False, name=None)
)
if observed_fold_keys != expected_fold_keys:
    raise RuntimeError(
        "Repeated NIBFS reference bukan 50 fold lengkap. "
        f"Found={len(observed_fold_keys)}, expected=50."
    )

OBSERVED_PANEL_LOOKUP: dict[tuple[int, int], list[str]] = {}
for (repeat, fold), group in REPEATED_PANELS.loc[
    REPEATED_PANELS["Method"].astype(str) == "NIBFS"
].groupby(["Repeat", "Fold"], sort=True):
    genes = (
        group.sort_values("Selection_rank")["Gene"]
        .astype(str)
        .tolist()
    )
    if len(genes) != FINAL_K or len(set(genes)) != FINAL_K:
        raise RuntimeError(
            f"Repeated reference panel repeat={repeat}, fold={fold} tidak valid."
        )
    OBSERVED_PANEL_LOOKUP[(int(repeat), int(fold))] = genes


# ------------------------------------------------------------
# 3. Fold-local limma and topology helpers
# ------------------------------------------------------------

def _perm_run_limma_rpy2(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.conversion import localconverter
    except Exception as exc:
        raise ImportError(
            "Permuted control requires rpy2 and R limma. Jalankan instalasi "
            "R/limma seperti pada repeated analysis."
        ) from exc

    ro.r("suppressPackageStartupMessages(library(limma))")
    genes = X.columns.astype(str).tolist()
    expr = X.to_numpy(dtype=float).T
    y_arr = np.asarray(y, dtype=int)

    with localconverter(ro.default_converter + numpy2ri.converter):
        ro.globalenv["expr_matrix_permctrl"] = expr
        ro.globalenv["group_vector_permctrl"] = y_arr
    ro.globalenv["gene_names_permctrl"] = ro.StrVector(genes)

    ro.r(
        """
        rownames(expr_matrix_permctrl) <- gene_names_permctrl
        group_factor_permctrl <- factor(
            group_vector_permctrl,
            levels=c(0,1),
            labels=c("Normal","Cancer")
        )
        design_permctrl <- model.matrix(~ group_factor_permctrl)
        fit_permctrl <- lmFit(expr_matrix_permctrl, design_permctrl)
        fit_permctrl <- eBayes(fit_permctrl)
        limma_result_permctrl <- topTable(
            fit_permctrl,
            coef=2,
            number=Inf,
            adjust.method="BH",
            sort.by="none"
        )
        limma_result_permctrl$Gene <- rownames(limma_result_permctrl)
        """
    )

    cols = list(ro.r("colnames(limma_result_permctrl)"))
    result = pd.DataFrame(
        {c: list(ro.r(f"limma_result_permctrl${c}")) for c in cols}
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
    return result


def _perm_prepare_real_topology(
    ppi_degree: pd.DataFrame,
    genes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    gene_col = _perm_find_column(ppi_degree, ["Gene", "Symbol", "Gene_symbol"])
    degree_col = _perm_find_column(ppi_degree, ["Degree", "degree"])
    normalized_col = _perm_find_column(
        ppi_degree, ["Normalized_degree", "normalized_degree"]
    )
    if gene_col is None or degree_col is None:
        raise KeyError(
            f"PPI table lacks Gene/Degree columns: {list(ppi_degree.columns)}"
        )

    ppi = pd.DataFrame(
        {
            "Gene": ppi_degree[gene_col].astype(str),
            "Degree": pd.to_numeric(
                ppi_degree[degree_col], errors="coerce"
            ).fillna(0.0),
        }
    )
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
    aligned = pd.DataFrame({"Gene": genes}).merge(ppi, on="Gene", how="left")
    aligned[["Degree", "Normalized_degree"]] = aligned[
        ["Degree", "Normalized_degree"]
    ].fillna(0.0)
    aligned["Rank_topo"] = aligned["Normalized_degree"].rank(
        method="average", ascending=False
    )
    return (
        aligned["Normalized_degree"].to_numpy(dtype=float),
        aligned["Rank_topo"].to_numpy(dtype=float),
        aligned,
    )


def _perm_fast_top_k_indices(
    rank_stat: np.ndarray,
    rank_topo: np.ndarray,
    genes: np.ndarray,
    k: int,
) -> np.ndarray:
    """Exact NIBFS order: rank_sum, then Rank_stat, then Gene."""
    rank_sum = np.asarray(rank_stat, dtype=float) + np.asarray(rank_topo, dtype=float)
    order = np.lexsort((genes, rank_stat, rank_sum))
    return order[:k]


def _perm_create_lr(seed: int) -> Pipeline:
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


def _perm_classification_metrics(
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


REAL_NORMALIZED_DEGREE, REAL_TOPO_RANK, REAL_TOPOLOGY_TABLE = (
    _perm_prepare_real_topology(PPI_DEGREE, GENES)
)
PPI_VECTOR_HASH = _perm_hash_float_array(REAL_NORMALIZED_DEGREE)


# ------------------------------------------------------------
# 4. Create/load fixed permuted topology-rank matrix
# ------------------------------------------------------------

configuration_lock = {
    "analysis": "Gene-label-permuted fixed STRING-topology control extension to 1,000",
    "script_version": "V2 extension 100-to-1000",
    "reuse_existing_100": REUSE_EXISTING_100,
    "development_shape": [int(X_DEV.shape[0]), int(X_DEV.shape[1])],
    "gene_order_sha256": GENE_HASH,
    "real_normalized_degree_sha256": PPI_VECTOR_HASH,
    "random_state": RANDOM_STATE,
    "repeats": REPEATS,
    "folds": FOLDS,
    "k": FINAL_K,
    "n_permutations": N_PERMUTATIONS,
    "permutation_seed_base": PERMUTATION_SEED_BASE,
    "decision_threshold": DEFAULT_THRESHOLD,
    "null_definition": (
        "Observed STRING normalized-degree values randomly reassigned to genes; "
        "each reassignment fixed across all repeated folds"
    ),
}

if LOCK_FILE.exists():
    existing_lock = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    if existing_lock != configuration_lock:
        raise RuntimeError(
            "Konfigurasi sekarang berbeda dari checkpoint yang sudah ada. "
            "Jangan mengubah jumlah permutasi/seed/data ketika resume. Gunakan "
            "folder output baru atau PERMUTED_FORCE_RERUN=True hanya bila memang "
            "ingin menghapus kontrol permutasi lama."
        )
else:
    _perm_save_json_atomic(configuration_lock, LOCK_FILE)

if RANK_MATRIX_FILE.exists() and PERMUTATION_MANIFEST_FILE.exists():
    stored = np.load(RANK_MATRIX_FILE, allow_pickle=False)
    stored_genes = stored["genes"].astype(str)
    PERMUTED_RANK_MATRIX = stored["rank_matrix"].astype(np.float64)
    permutation_seeds = stored["seeds"].astype(np.int64)
    if not np.array_equal(stored_genes, GENES):
        raise RuntimeError("Gene order pada saved permutation matrix berubah.")
    if PERMUTED_RANK_MATRIX.shape != (N_GENES, N_PERMUTATIONS):
        raise RuntimeError(
            f"Saved rank matrix shape salah: {PERMUTED_RANK_MATRIX.shape}."
        )
    if len(permutation_seeds) != N_PERMUTATIONS:
        raise RuntimeError("Saved permutation seed count tidak cocok.")
    PERMUTATION_MANIFEST = pd.read_csv(PERMUTATION_MANIFEST_FILE)
    print("Loaded fixed permuted topology matrix:", RANK_MATRIX_FILE, flush=True)
else:
    if RANK_MATRIX_FILE.exists() != PERMUTATION_MANIFEST_FILE.exists():
        raise RuntimeError(
            "Permutation matrix/manifest tidak lengkap. Hapus folder kontrol yang "
            "belum pernah selesai dibuat atau gunakan PERMUTED_FORCE_RERUN=True."
        )

    print(
        f"Generating {N_PERMUTATIONS} fixed gene-label permutations ...",
        flush=True,
    )
    PERMUTED_RANK_MATRIX = np.empty(
        (N_GENES, N_PERMUTATIONS), dtype=np.float32
    )
    permutation_seeds = np.empty(N_PERMUTATIONS, dtype=np.int64)
    manifest_rows: list[dict] = []

    for permutation_id in range(1, N_PERMUTATIONS + 1):
        seed = PERMUTATION_SEED_BASE + permutation_id
        rng = np.random.default_rng(seed)
        permuted_degree = rng.permutation(REAL_NORMALIZED_DEGREE)
        permuted_rank = pd.Series(permuted_degree).rank(
            method="average", ascending=False
        ).to_numpy(dtype=np.float32)

        PERMUTED_RANK_MATRIX[:, permutation_id - 1] = permuted_rank
        permutation_seeds[permutation_id - 1] = seed
        manifest_rows.append(
            {
                "Permutation_ID": permutation_id,
                "Seed": seed,
                "Degree_distribution_preserved": bool(
                    np.array_equal(
                        np.sort(permuted_degree),
                        np.sort(REAL_NORMALIZED_DEGREE),
                    )
                ),
                "Permuted_degree_assignment_SHA256": _perm_hash_float_array(
                    permuted_degree
                ),
                "Permuted_rank_SHA256": _perm_hash_float_array(permuted_rank),
            }
        )

    PERMUTATION_MANIFEST = pd.DataFrame(manifest_rows)
    if not PERMUTATION_MANIFEST["Degree_distribution_preserved"].all():
        raise RuntimeError("Degree distribution was not preserved in a permutation.")

    _perm_save_npz_atomic(
        RANK_MATRIX_FILE,
        genes=GENES.astype("U"),
        rank_matrix=PERMUTED_RANK_MATRIX,
        seeds=permutation_seeds,
    )
    _perm_save_csv_atomic(PERMUTATION_MANIFEST, PERMUTATION_MANIFEST_FILE)
    PERMUTED_RANK_MATRIX = PERMUTED_RANK_MATRIX.astype(np.float64)

print("Permutation rank matrix:", PERMUTED_RANK_MATRIX.shape)


# ------------------------------------------------------------
# 5. Fold checkpoints
# ------------------------------------------------------------

def _perm_checkpoint_prefix(repeat: int, fold: int) -> Path:
    return CHECKPOINT_DIR / f"repeat_{repeat:02d}_fold_{fold:02d}"


def _perm_checkpoint_files(repeat: int, fold: int) -> dict[str, Path]:
    prefix = _perm_checkpoint_prefix(repeat, fold)
    return {
        "metrics": prefix.with_name(prefix.name + "_metrics.csv"),
        "predictions": prefix.with_name(prefix.name + "_predictions.csv.gz"),
        "panels": prefix.with_name(prefix.name + "_panels.csv.gz"),
        "verification": prefix.with_name(prefix.name + "_verification.csv"),
        "runtime": prefix.with_name(prefix.name + "_runtime.json"),
        "done": prefix.with_name(prefix.name + "_DONE.json"),
    }


def _perm_checkpoint_complete(repeat: int, fold: int) -> bool:
    files = _perm_checkpoint_files(repeat, fold)
    return all(path.exists() for path in files.values())



def _base100_checkpoint_files(repeat: int, fold: int) -> dict[str, Path]:
    prefix = BASE_100_CHECKPOINT_DIR / f"repeat_{repeat:02d}_fold_{fold:02d}"
    return {
        "metrics": prefix.with_name(prefix.name + "_metrics.csv"),
        "predictions": prefix.with_name(prefix.name + "_predictions.csv.gz"),
        "panels": prefix.with_name(prefix.name + "_panels.csv.gz"),
        "verification": prefix.with_name(prefix.name + "_verification.csv"),
        "runtime": prefix.with_name(prefix.name + "_runtime.json"),
        "done": prefix.with_name(prefix.name + "_DONE.json"),
    }


def _base100_checkpoint_complete(repeat: int, fold: int) -> bool:
    files = _base100_checkpoint_files(repeat, fold)
    return all(path.exists() for path in files.values())


_BASE100_AGGREGATE_CACHE: dict[str, pd.DataFrame] | None = None
_BASE100_AGGREGATE_PATHS: dict[str, Path] | None = None


def _read_csv_columns(path: Path) -> set[str]:
    try:
        return set(pd.read_csv(path, nrows=3).columns.astype(str))
    except Exception:
        return set()


def _find_base100_aggregate_file(
    preferred_names: list[str],
    required_columns: set[str],
) -> Path | None:
    """Find a completed base-100 aggregate CSV even when an older script used
    slightly different output names.
    """
    candidates: list[Path] = []

    for name in preferred_names:
        candidate = BASE_100_DIR / name
        if candidate.exists():
            candidates.append(candidate)

    for pattern in ("*.csv", "*.csv.gz"):
        for candidate in BASE_100_DIR.glob(pattern):
            if candidate not in candidates:
                candidates.append(candidate)

    valid: list[Path] = []
    for candidate in candidates:
        columns = _read_csv_columns(candidate)
        if required_columns.issubset(columns):
            valid.append(candidate)

    if not valid:
        return None

    # Prefer root-level exact/preferred names, then the largest aggregate file.
    preferred_rank = {name: i for i, name in enumerate(preferred_names)}
    valid.sort(
        key=lambda p: (
            preferred_rank.get(p.name, len(preferred_names)),
            -p.stat().st_size,
            p.name,
        )
    )
    return valid[0]


def _load_base100_aggregate_cache() -> tuple[dict[str, pd.DataFrame], dict[str, Path]]:
    global _BASE100_AGGREGATE_CACHE, _BASE100_AGGREGATE_PATHS

    if _BASE100_AGGREGATE_CACHE is not None and _BASE100_AGGREGATE_PATHS is not None:
        return _BASE100_AGGREGATE_CACHE, _BASE100_AGGREGATE_PATHS

    metrics_path = _find_base100_aggregate_file(
        [
            "permuted_fold_metrics.csv.gz",
            "permuted_fold_metrics.csv",
            "permuted_metrics.csv.gz",
            "permuted_metrics.csv",
        ],
        {
            "Repeat",
            "Fold",
            "Permutation_ID",
            "Permutation_seed",
            "Validation_samples",
        },
    )
    predictions_path = _find_base100_aggregate_file(
        [
            "permuted_predictions.csv.gz",
            "permuted_predictions.csv",
            "permutation_predictions.csv.gz",
            "permutation_predictions.csv",
        ],
        {
            "Repeat",
            "Fold",
            "Permutation_ID",
            "Sample_ID",
            "True_Label",
            "Probability",
        },
    )
    panels_path = _find_base100_aggregate_file(
        [
            "permuted_selected_panels.csv.gz",
            "permuted_selected_panels.csv",
            "permuted_panels.csv.gz",
            "permuted_panels.csv",
        ],
        {
            "Repeat",
            "Fold",
            "Permutation_ID",
            "Selection_rank",
            "Gene",
        },
    )

    missing = [
        label
        for label, path in [
            ("metrics", metrics_path),
            ("predictions", predictions_path),
            ("panels", panels_path),
        ]
        if path is None
    ]
    if missing:
        raise FileNotFoundError(
            "Base 100-permutation fold checkpoints were not found, and the "
            "completed aggregate outputs could not be identified for: "
            + ", ".join(missing)
            + f". Searched in {BASE_100_DIR}"
        )

    frames = {
        "metrics": pd.read_csv(metrics_path),
        "predictions": pd.read_csv(predictions_path),
        "panels": pd.read_csv(panels_path),
    }
    paths = {
        "metrics": metrics_path,
        "predictions": predictions_path,
        "panels": panels_path,
    }

    expected_ids = set(range(1, BASE_PERMUTATIONS + 1))
    expected_fold_keys = {
        (repeat, fold)
        for repeat in range(1, REPEATS + 1)
        for fold in range(1, FOLDS + 1)
    }

    for name, frame in frames.items():
        frame["Repeat"] = pd.to_numeric(frame["Repeat"], errors="raise").astype(int)
        frame["Fold"] = pd.to_numeric(frame["Fold"], errors="raise").astype(int)
        frame["Permutation_ID"] = pd.to_numeric(
            frame["Permutation_ID"], errors="raise"
        ).astype(int)

        fold_keys = set(zip(frame["Repeat"], frame["Fold"]))
        if fold_keys != expected_fold_keys:
            raise RuntimeError(
                f"Base-100 aggregate {name} does not contain exactly all 50 folds. "
                f"Found {len(fold_keys)} fold keys."
            )

        for (repeat, fold), group in frame.groupby(["Repeat", "Fold"], sort=False):
            ids = set(group["Permutation_ID"].astype(int))
            if ids != expected_ids:
                raise RuntimeError(
                    f"Base-100 aggregate {name} has unexpected permutation IDs "
                    f"at repeat={repeat}, fold={fold}."
                )

    metrics = frames["metrics"]
    predictions = frames["predictions"]
    panels = frames["panels"]

    for (repeat, fold), group in metrics.groupby(["Repeat", "Fold"], sort=False):
        if len(group) != BASE_PERMUTATIONS:
            raise RuntimeError(
                f"Base-100 aggregate metrics expected 100 rows at "
                f"repeat={repeat}, fold={fold}; found {len(group)}."
            )
        if "Permutation_seed" not in group.columns:
            raise RuntimeError("Base-100 aggregate metrics lacks Permutation_seed.")
        group_sorted = group.sort_values("Permutation_ID")
        expected_seeds = PERMUTATION_SEED_BASE + np.arange(
            1, BASE_PERMUTATIONS + 1
        )
        if not np.array_equal(
            group_sorted["Permutation_seed"].to_numpy(dtype=np.int64),
            expected_seeds.astype(np.int64),
        ):
            raise RuntimeError(
                f"Base-100 aggregate permutation seeds mismatch at "
                f"repeat={repeat}, fold={fold}."
            )

        validation_samples = int(group["Validation_samples"].iloc[0])
        pred_group = predictions[
            (predictions["Repeat"] == repeat)
            & (predictions["Fold"] == fold)
        ]
        expected_predictions = BASE_PERMUTATIONS * validation_samples
        if len(pred_group) != expected_predictions:
            raise RuntimeError(
                f"Base-100 aggregate predictions expected {expected_predictions} "
                f"rows at repeat={repeat}, fold={fold}; found {len(pred_group)}."
            )

        panel_group = panels[
            (panels["Repeat"] == repeat)
            & (panels["Fold"] == fold)
        ]
        if len(panel_group) != BASE_PERMUTATIONS * FINAL_K:
            raise RuntimeError(
                f"Base-100 aggregate panels expected "
                f"{BASE_PERMUTATIONS * FINAL_K} rows at repeat={repeat}, "
                f"fold={fold}; found {len(panel_group)}."
            )

    _BASE100_AGGREGATE_CACHE = frames
    _BASE100_AGGREGATE_PATHS = paths
    return frames, paths


def _load_base100_rows(
    repeat: int,
    fold: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Load and strictly validate completed 100-permutation output.

    Preferred source is the original per-fold checkpoint bundle. If an older
    completed run retained only aggregate CSV outputs, use those instead.
    """
    if _base100_checkpoint_complete(repeat, fold):
        files = _base100_checkpoint_files(repeat, fold)
        metrics = pd.read_csv(files["metrics"])
        predictions = pd.read_csv(files["predictions"])
        panels = pd.read_csv(files["panels"])
    else:
        frames, _ = _load_base100_aggregate_cache()
        metrics = frames["metrics"][
            (frames["metrics"]["Repeat"] == repeat)
            & (frames["metrics"]["Fold"] == fold)
        ].copy()
        predictions = frames["predictions"][
            (frames["predictions"]["Repeat"] == repeat)
            & (frames["predictions"]["Fold"] == fold)
        ].copy()
        panels = frames["panels"][
            (frames["panels"]["Repeat"] == repeat)
            & (frames["panels"]["Fold"] == fold)
        ].copy()

    expected_ids = set(range(1, BASE_PERMUTATIONS + 1))
    for name, frame in [
        ("metrics", metrics),
        ("predictions", predictions),
        ("panels", panels),
    ]:
        ids = set(pd.to_numeric(frame["Permutation_ID"], errors="raise").astype(int))
        if ids != expected_ids:
            raise RuntimeError(
                f"Base-100 {name} has unexpected permutation IDs at "
                f"repeat={repeat}, fold={fold}."
            )
        if set(pd.to_numeric(frame["Repeat"], errors="raise").astype(int)) != {repeat}:
            raise RuntimeError(f"Base-100 {name} repeat mismatch.")
        if set(pd.to_numeric(frame["Fold"], errors="raise").astype(int)) != {fold}:
            raise RuntimeError(f"Base-100 {name} fold mismatch.")

    if len(metrics) != BASE_PERMUTATIONS:
        raise RuntimeError(
            f"Base-100 metrics expected {BASE_PERMUTATIONS} rows; found {len(metrics)}."
        )
    if "Permutation_seed" not in metrics.columns:
        raise RuntimeError("Base-100 metrics lacks Permutation_seed for audit.")
    seed_table = metrics.sort_values("Permutation_ID")
    expected_seeds = PERMUTATION_SEED_BASE + np.arange(
        1, BASE_PERMUTATIONS + 1
    )
    if not np.array_equal(
        seed_table["Permutation_seed"].to_numpy(dtype=np.int64),
        expected_seeds.astype(np.int64),
    ):
        raise RuntimeError(
            f"Base-100 permutation seeds mismatch at repeat={repeat}, fold={fold}."
        )
    expected_predictions = BASE_PERMUTATIONS * int(
        metrics["Validation_samples"].iloc[0]
    )
    if len(predictions) != expected_predictions:
        raise RuntimeError(
            f"Base-100 predictions expected {expected_predictions} rows; "
            f"found {len(predictions)}."
        )
    if len(panels) != BASE_PERMUTATIONS * FINAL_K:
        raise RuntimeError(
            f"Base-100 panels expected {BASE_PERMUTATIONS * FINAL_K} rows; "
            f"found {len(panels)}."
        )
    return (
        metrics.to_dict("records"),
        predictions.to_dict("records"),
        panels.to_dict("records"),
    )


TOTAL_FOLDS = REPEATS * FOLDS
completed_before = sum(
    _perm_checkpoint_complete(repeat, fold)
    for repeat in range(1, REPEATS + 1)
    for fold in range(1, FOLDS + 1)
)
print(f"Existing complete checkpoints: {completed_before}/{TOTAL_FOLDS}")

base100_checkpoint_complete_folds = sum(
    _base100_checkpoint_complete(repeat, fold)
    for repeat in range(1, REPEATS + 1)
    for fold in range(1, FOLDS + 1)
)

if REUSE_EXISTING_100:
    if base100_checkpoint_complete_folds == TOTAL_FOLDS:
        base100_source_note = "50/50 original fold checkpoint bundles"
    else:
        try:
            _, aggregate_paths = _load_base100_aggregate_cache()
            base100_source_note = (
                "completed aggregate outputs "
                f"(metrics={aggregate_paths['metrics'].name}, "
                f"predictions={aggregate_paths['predictions'].name}, "
                f"panels={aggregate_paths['panels'].name})"
            )
        except Exception as exc:
            raise RuntimeError(
                "PERMUTED_REUSE_EXISTING_100=True, tetapi script tidak menemukan "
                "50 bundle checkpoint lama maupun aggregate outputs lengkap dari "
                "kontrol 100-permutasi. Hasil 100-permutasi tidak perlu dihitung "
                "ulang; gunakan script V3 ini setelah memastikan tiga aggregate "
                "files (metrics, predictions, selected panels) masih ada di "
                f"{BASE_100_DIR}. Detail: {exc}"
            ) from exc

    if BASE_100_MANIFEST_FILE.exists():
        base_manifest = pd.read_csv(BASE_100_MANIFEST_FILE)
        if set(base_manifest["Permutation_ID"].astype(int)) != set(range(1, 101)):
            raise RuntimeError("Manifest kontrol 100-permutasi tidak berisi ID 1--100.")
        expected_base_seeds = PERMUTATION_SEED_BASE + np.arange(1, 101)
        if not np.array_equal(
            base_manifest.sort_values("Permutation_ID")["Seed"].to_numpy(dtype=np.int64),
            expected_base_seeds.astype(np.int64),
        ):
            raise RuntimeError("Seed base-100 berbeda dari extension 1,000-permutasi.")
        manifest_note = "manifest verified"
    else:
        warnings.warn(
            "Base-100 permutation manifest tidak ditemukan; seed divalidasi "
            "langsung dari aggregate/per-fold metrics."
        )
        manifest_note = "manifest absent; metrics seed validation enabled"

    print(
        "Base 100-permutation control verified: 50/50 folds; "
        f"source={base100_source_note}; permutations 1--100 will be reused "
        f"({manifest_note}).",
        flush=True,
    )
else:
    print("Base-100 reuse disabled; all 1,000 permutations will be recomputed.")


# ------------------------------------------------------------
# 6. Execute/resume permuted controls
# ------------------------------------------------------------

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
        global_position = (repeat - 1) * FOLDS + fold
        files = _perm_checkpoint_files(repeat, fold)

        if _perm_checkpoint_complete(repeat, fold):
            print(
                f"[{global_position:02d}/{TOTAL_FOLDS}] repeat {repeat}, fold {fold}: "
                "checkpoint found — skipped",
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

        print("\n" + "=" * 78, flush=True)
        print(
            f"[{global_position:02d}/{TOTAL_FOLDS}] PERMUTED CONTROL — "
            f"REPEAT {repeat}/{REPEATS}, FOLD {fold}/{FOLDS}",
            flush=True,
        )
        print(
            f"training={X_fit.shape}, validation={X_val.shape}, "
            f"permutations={N_PERMUTATIONS}",
            flush=True,
        )

        print("  [1/3] Fold-local R limma ...", flush=True)
        limma = _perm_run_limma_rpy2(X_fit, y_fit)
        limma_aligned = limma.set_index("Gene").reindex(GENES)
        if limma_aligned["Rank_stat"].isna().any():
            missing_genes = limma_aligned.index[
                limma_aligned["Rank_stat"].isna()
            ].tolist()[:10]
            raise RuntimeError(
                f"limma did not return all eligible genes; examples={missing_genes}"
            )
        rank_stat = limma_aligned["Rank_stat"].to_numpy(dtype=float)

        print("  [2/3] Verify real STRING panel against repeated checkpoints ...", flush=True)
        observed_idx = _perm_fast_top_k_indices(
            rank_stat, REAL_TOPO_RANK, GENES, FINAL_K
        )
        reconstructed_observed_panel = GENES[observed_idx].tolist()
        expected_observed_panel = OBSERVED_PANEL_LOOKUP[(repeat, fold)]
        exact_match = reconstructed_observed_panel == expected_observed_panel
        set_match = set(reconstructed_observed_panel) == set(expected_observed_panel)
        if not exact_match:
            raise RuntimeError(
                "Real STRING reconstruction did not exactly match repeated checkpoint "
                f"at repeat={repeat}, fold={fold}. set_match={set_match}. "
                f"Reconstructed={reconstructed_observed_panel}; "
                f"Expected={expected_observed_panel}"
            )

        verification = pd.DataFrame(
            [
                {
                    "Repeat": repeat,
                    "Fold": fold,
                    "Real_STRING_panel_exact_order_match": exact_match,
                    "Real_STRING_panel_set_match": set_match,
                    "Expected_panel": "|".join(expected_observed_panel),
                    "Reconstructed_panel": "|".join(reconstructed_observed_panel),
                }
            ]
        )

        print("  [3/3] Fixed permuted anchors + LR validation ...", flush=True)
        metric_rows: list[dict] = []
        prediction_rows: list[dict] = []
        panel_rows: list[dict] = []
        observed_set = set(expected_observed_panel)

        first_new_permutation = 1
        if REUSE_EXISTING_100:
            (
                metric_rows,
                prediction_rows,
                panel_rows,
            ) = _load_base100_rows(repeat, fold)
            first_new_permutation = BASE_PERMUTATIONS + 1
            print(
                "    reused completed permutations 1--100; "
                "computing 101--1000 only",
                flush=True,
            )

        for permutation_id in range(first_new_permutation, N_PERMUTATIONS + 1):
            rank_topo_perm = PERMUTED_RANK_MATRIX[:, permutation_id - 1]
            selected_idx = _perm_fast_top_k_indices(
                rank_stat, rank_topo_perm, GENES, FINAL_K
            )
            selected_genes = GENES[selected_idx].tolist()
            selected_set = set(selected_genes)

            model_seed = (
                RANDOM_STATE
                + global_position * 100000
                + permutation_id
            )
            model = _perm_create_lr(model_seed)
            model.fit(X_fit[selected_genes], y_fit)
            probability = model.predict_proba(X_val[selected_genes])[:, 1]
            metric = _perm_classification_metrics(
                y_val, probability, DEFAULT_THRESHOLD
            )

            intersection = len(selected_set & observed_set)
            union = len(selected_set | observed_set)
            metric_rows.append(
                {
                    "Repeat": repeat,
                    "Fold": fold,
                    "Permutation_ID": permutation_id,
                    "Permutation_seed": int(permutation_seeds[permutation_id - 1]),
                    "Control": "Fixed gene-label-permuted STRING degree",
                    "Classifier": "LR",
                    "k": FINAL_K,
                    "Training_samples": len(fit_idx),
                    "Validation_samples": len(val_idx),
                    "Overlap_with_real_STRING_panel": intersection,
                    "Jaccard_with_real_STRING_panel": (
                        intersection / union if union else np.nan
                    ),
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
                        "Permutation_ID": permutation_id,
                        "Sample_ID": sample_id,
                        "True_Label": int(true_label),
                        "Probability": float(prob),
                    }
                )

            for selection_rank, gene_index in enumerate(selected_idx, 1):
                gene = str(GENES[gene_index])
                panel_rows.append(
                    {
                        "Repeat": repeat,
                        "Fold": fold,
                        "Permutation_ID": permutation_id,
                        "Selection_rank": selection_rank,
                        "Gene": gene,
                        "Rank_stat": float(rank_stat[gene_index]),
                        "Permuted_Rank_topo": float(rank_topo_perm[gene_index]),
                        "Rank_sum": float(
                            rank_stat[gene_index] + rank_topo_perm[gene_index]
                        ),
                        "In_real_STRING_fold_panel": gene in observed_set,
                    }
                )

            if permutation_id % max(1, N_PERMUTATIONS // 10) == 0:
                print(
                    f"    completed {permutation_id}/{N_PERMUTATIONS} permutations",
                    flush=True,
                )

        elapsed = perf_counter() - fold_start
        runtime = {
            "Repeat": repeat,
            "Fold": fold,
            "Elapsed_seconds": elapsed,
            "Elapsed_minutes": elapsed / 60.0,
            "N_permutations": N_PERMUTATIONS,
            "Training_samples": len(fit_idx),
            "Validation_samples": len(val_idx),
            "Real_STRING_panel_verified": True,
            "Completed": True,
        }

        _perm_save_csv_atomic(pd.DataFrame(metric_rows), files["metrics"])
        _perm_save_csv_atomic(pd.DataFrame(prediction_rows), files["predictions"])
        _perm_save_csv_atomic(pd.DataFrame(panel_rows), files["panels"])
        _perm_save_csv_atomic(verification, files["verification"])
        _perm_save_json_atomic(runtime, files["runtime"])
        _perm_save_json_atomic(
            {
                "status": "complete",
                "repeat": repeat,
                "fold": fold,
                "n_permutations": N_PERMUTATIONS,
                "elapsed_seconds": elapsed,
            },
            files["done"],
        )

        new_fold_count += 1
        session_fold_times.append(elapsed)
        complete_now = sum(
            _perm_checkpoint_complete(r, f)
            for r in range(1, REPEATS + 1)
            for f in range(1, FOLDS + 1)
        )
        remaining = TOTAL_FOLDS - complete_now
        eta_minutes = (
            remaining * float(np.mean(session_fold_times)) / 60.0
            if session_fold_times
            else np.nan
        )
        print(
            f"  completed in {elapsed / 60.0:.2f} minutes | "
            f"checkpoints={complete_now}/{TOTAL_FOLDS} | "
            f"estimated remaining={eta_minutes:.1f} minutes",
            flush=True,
        )

        del X_fit, X_val, y_fit, y_val, limma, limma_aligned
        del metric_rows, prediction_rows, panel_rows
        gc.collect()

    if stop_requested:
        break


# ------------------------------------------------------------
# 7. Aggregate all complete checkpoints safely
# ------------------------------------------------------------

metric_parts: list[pd.DataFrame] = []
prediction_parts: list[pd.DataFrame] = []
panel_parts: list[pd.DataFrame] = []
verification_parts: list[pd.DataFrame] = []
runtime_rows: list[dict] = []
complete_fold_keys: set[tuple[int, int]] = set()

for repeat in range(1, REPEATS + 1):
    for fold in range(1, FOLDS + 1):
        if not _perm_checkpoint_complete(repeat, fold):
            continue
        files = _perm_checkpoint_files(repeat, fold)
        metric_parts.append(pd.read_csv(files["metrics"]))
        prediction_parts.append(pd.read_csv(files["predictions"]))
        panel_parts.append(pd.read_csv(files["panels"]))
        verification_parts.append(pd.read_csv(files["verification"]))
        runtime_rows.append(
            json.loads(files["runtime"].read_text(encoding="utf-8"))
        )
        complete_fold_keys.add((repeat, fold))

if not metric_parts:
    raise RuntimeError("Belum ada permuted-control fold yang selesai.")

fold_metrics = pd.concat(metric_parts, ignore_index=True).sort_values(
    ["Repeat", "Fold", "Permutation_ID"]
)
predictions = pd.concat(prediction_parts, ignore_index=True).sort_values(
    ["Repeat", "Fold", "Permutation_ID", "Sample_ID"]
)
selected_panels = pd.concat(panel_parts, ignore_index=True).sort_values(
    ["Repeat", "Fold", "Permutation_ID", "Selection_rank"]
)
verification_table = pd.concat(verification_parts, ignore_index=True).sort_values(
    ["Repeat", "Fold"]
)
runtime_table = pd.DataFrame(runtime_rows).sort_values(["Repeat", "Fold"])

_perm_save_csv_atomic(
    fold_metrics,
    OUTPUT_DIR / "permuted_fold_metrics.csv.gz",
)
_perm_save_csv_atomic(
    predictions,
    OUTPUT_DIR / "permuted_predictions.csv.gz",
)
_perm_save_csv_atomic(
    selected_panels,
    OUTPUT_DIR / "permuted_selected_panels.csv.gz",
)
_perm_save_csv_atomic(
    verification_table,
    OUTPUT_DIR / "real_STRING_panel_reconstruction_verification.csv",
)
_perm_save_csv_atomic(runtime_table, OUTPUT_DIR / "permuted_runtime.csv")

if not verification_table[
    "Real_STRING_panel_exact_order_match"
].astype(bool).all():
    raise RuntimeError("At least one real STRING panel verification failed.")

complete_repeats = [
    repeat
    for repeat in range(1, REPEATS + 1)
    if all((repeat, fold) in complete_fold_keys for fold in range(1, FOLDS + 1))
]


# ------------------------------------------------------------
# 8. OOF and stability summaries for complete repeats only
# ------------------------------------------------------------

OOF_METRICS_FILE = OUTPUT_DIR / "permuted_oof_metrics_by_repeat.csv"
STABILITY_FILE = OUTPUT_DIR / "permuted_stability_by_repeat.csv"
NULL_SUMMARY_FILE = OUTPUT_DIR / "permuted_null_summary_by_permutation.csv"
GENE_FREQUENCY_FILE = OUTPUT_DIR / "permuted_gene_selection_frequency.csv.gz"
EMPIRICAL_TESTS_FILE = OUTPUT_DIR / "permuted_empirical_tests.csv"

if complete_repeats:
    oof_rows: list[dict] = []
    for repeat in complete_repeats:
        repeat_pred = predictions.loc[predictions["Repeat"] == repeat]
        for permutation_id, group in repeat_pred.groupby(
            "Permutation_ID", sort=True
        ):
            if len(group) != len(X_DEV):
                raise RuntimeError(
                    f"Repeat {repeat}, permutation {permutation_id}: "
                    f"expected {len(X_DEV)} OOF predictions, found {len(group)}."
                )
            if group["Sample_ID"].duplicated().any():
                raise RuntimeError(
                    f"Repeat {repeat}, permutation {permutation_id}: duplicate OOF sample."
                )
            metrics = _perm_classification_metrics(
                group["True_Label"].to_numpy(int),
                group["Probability"].to_numpy(float),
                DEFAULT_THRESHOLD,
            )
            oof_rows.append(
                {
                    "Repeat": repeat,
                    "Permutation_ID": int(permutation_id),
                    "Samples": len(group),
                    **metrics,
                }
            )

    oof_metrics = pd.DataFrame(oof_rows).sort_values(
        ["Permutation_ID", "Repeat"]
    )
    _perm_save_csv_atomic(oof_metrics, OOF_METRICS_FILE)

    stability_rows: list[dict] = []
    for repeat in complete_repeats:
        repeat_panels = selected_panels.loc[selected_panels["Repeat"] == repeat]
        for permutation_id, group in repeat_panels.groupby(
            "Permutation_ID", sort=True
        ):
            panels_by_fold = {
                int(fold): set(
                    fold_group.sort_values("Selection_rank")["Gene"].astype(str)
                )
                for fold, fold_group in group.groupby("Fold", sort=True)
            }
            if len(panels_by_fold) != FOLDS:
                raise RuntimeError(
                    f"Repeat {repeat}, permutation {permutation_id}: "
                    f"expected {FOLDS} panels, found {len(panels_by_fold)}."
                )
            values: list[float] = []
            for fold_1, fold_2 in combinations(sorted(panels_by_fold), 2):
                a = panels_by_fold[fold_1]
                b = panels_by_fold[fold_2]
                values.append(len(a & b) / len(a | b))
            stability_rows.append(
                {
                    "Repeat": repeat,
                    "Permutation_ID": int(permutation_id),
                    "Fold_panels": len(panels_by_fold),
                    "Pairwise_comparisons": len(values),
                    "Mean_Jaccard": float(np.mean(values)),
                    "SD_Jaccard": float(np.std(values, ddof=1)),
                    "Median_Jaccard": float(np.median(values)),
                    "Minimum_Jaccard": float(np.min(values)),
                    "Maximum_Jaccard": float(np.max(values)),
                }
            )

    stability = pd.DataFrame(stability_rows).sort_values(
        ["Permutation_ID", "Repeat"]
    )
    _perm_save_csv_atomic(stability, STABILITY_FILE)

    null_summary = (
        stability.groupby("Permutation_ID")["Mean_Jaccard"]
        .agg(
            Stability_repeats="count",
            Mean_Jaccard="mean",
            SD_Jaccard_across_repeats="std",
            Minimum_repeat_Jaccard="min",
            Maximum_repeat_Jaccard="max",
        )
        .reset_index()
    )

    oof_summary = (
        oof_metrics.groupby("Permutation_ID")[
            ["ROC_AUC", "F1", "MCC", "Brier_score"]
        ]
        .mean()
        .reset_index()
        .rename(
            columns={
                "ROC_AUC": "Mean_OOF_ROC_AUC",
                "F1": "Mean_OOF_F1",
                "MCC": "Mean_OOF_MCC",
                "Brier_score": "Mean_OOF_Brier_score",
            }
        )
    )
    null_summary = null_summary.merge(oof_summary, on="Permutation_ID")
    null_summary = null_summary.merge(
        PERMUTATION_MANIFEST[["Permutation_ID", "Seed"]],
        on="Permutation_ID",
        how="left",
    )
    _perm_save_csv_atomic(null_summary, NULL_SUMMARY_FILE)

    # Frequency across all currently complete fold panels.
    frequency_rows: list[dict] = []
    complete_panel_data = selected_panels.loc[
        selected_panels["Repeat"].isin(complete_repeats)
    ]
    total_panels_per_permutation = len(complete_repeats) * FOLDS
    for permutation_id, group in complete_panel_data.groupby(
        "Permutation_ID", sort=True
    ):
        counts = Counter(group["Gene"].astype(str))
        mean_rank = group.groupby("Gene")["Selection_rank"].mean()
        for gene, frequency in counts.items():
            frequency_rows.append(
                {
                    "Permutation_ID": int(permutation_id),
                    "Gene": gene,
                    "Selection_frequency": int(frequency),
                    "Total_fold_panels": total_panels_per_permutation,
                    "Selection_proportion": float(
                        frequency / total_panels_per_permutation
                    ),
                    "Mean_selection_rank": float(mean_rank.loc[gene]),
                }
            )
    gene_frequency = pd.DataFrame(frequency_rows).sort_values(
        ["Permutation_ID", "Selection_frequency", "Mean_selection_rank", "Gene"],
        ascending=[True, False, True, True],
    )
    _perm_save_csv_atomic(gene_frequency, GENE_FREQUENCY_FILE)
else:
    oof_metrics = pd.DataFrame()
    stability = pd.DataFrame()
    null_summary = pd.DataFrame()


# ------------------------------------------------------------
# 9. Final empirical null tests when 50/50 folds are complete
# ------------------------------------------------------------

analysis_complete = len(complete_fold_keys) == TOTAL_FOLDS
empirical_tests = pd.DataFrame()
observed_reference = pd.DataFrame()

if analysis_complete:
    if len(complete_repeats) != REPEATS:
        raise RuntimeError("50 fold complete but complete repeat count is not 10.")
    if null_summary["Permutation_ID"].nunique() != N_PERMUTATIONS:
        raise RuntimeError("Null summary does not contain every permutation.")

    observed_oof = REPEATED_OOF.loc[
        REPEATED_OOF["Method"].astype(str).isin(["NIBFS", "DEG-only"])
    ].copy()
    observed_stability = REPEATED_STABILITY.loc[
        REPEATED_STABILITY["Method"].astype(str).isin(["NIBFS", "DEG-only"])
    ].copy()

    if (
        observed_oof.groupby("Method")["Repeat"].nunique().min() != REPEATS
        or observed_stability.groupby("Method")["Repeat"].nunique().min() != REPEATS
    ):
        raise RuntimeError("Observed repeated reference is not complete for 10 repeats.")

    observed_rows: list[dict] = []
    for method in ["NIBFS", "DEG-only"]:
        oof_group = observed_oof.loc[observed_oof["Method"] == method]
        stab_group = observed_stability.loc[
            observed_stability["Method"] == method
        ]
        observed_rows.append(
            {
                "Reference": "Real STRING NIBFS" if method == "NIBFS" else "DEG-only",
                "Method": method,
                "Repeats": REPEATS,
                "Mean_Jaccard": float(stab_group["Mean_Jaccard"].mean()),
                "Mean_OOF_ROC_AUC": float(oof_group["ROC_AUC"].mean()),
                "Mean_OOF_F1": float(oof_group["F1"].mean()),
                "Mean_OOF_MCC": float(oof_group["MCC"].mean()),
                "Mean_OOF_Brier_score": float(oof_group["Brier_score"].mean()),
            }
        )
    observed_reference = pd.DataFrame(observed_rows)
    _perm_save_csv_atomic(
        observed_reference,
        OUTPUT_DIR / "observed_STRING_and_DEG_reference.csv",
    )

    metric_specs = [
        ("Mean_Jaccard", "greater", "Panel stability"),
        ("Mean_OOF_ROC_AUC", "greater", "OOF ROC-AUC"),
        ("Mean_OOF_F1", "greater", "OOF F1"),
        ("Mean_OOF_MCC", "greater", "OOF MCC"),
        ("Mean_OOF_Brier_score", "less", "OOF Brier score"),
    ]
    real_string_row = observed_reference.loc[
        observed_reference["Method"] == "NIBFS"
    ].iloc[0]

    test_rows: list[dict] = []
    for column, direction, label in metric_specs:
        observed_value = float(real_string_row[column])
        null_values = null_summary[column].to_numpy(dtype=float)
        if direction == "greater":
            extreme_count = int(np.sum(null_values >= observed_value))
            percentile = float(
                100.0
                * (
                    np.sum(null_values < observed_value)
                    + 0.5 * np.sum(null_values == observed_value)
                )
                / len(null_values)
            )
        else:
            extreme_count = int(np.sum(null_values <= observed_value))
            percentile = float(
                100.0
                * (
                    np.sum(null_values > observed_value)
                    + 0.5 * np.sum(null_values == observed_value)
                )
                / len(null_values)
            )

        empirical_p = (1 + extreme_count) / (N_PERMUTATIONS + 1)
        raw_tail_probability = extreme_count / N_PERMUTATIONS
        if extreme_count == 0:
            ci_low = 0.0
        else:
            ci_low = float(
                beta_distribution.ppf(
                    0.025, extreme_count, N_PERMUTATIONS - extreme_count + 1
                )
            )
        if extreme_count == N_PERMUTATIONS:
            ci_high = 1.0
        else:
            ci_high = float(
                beta_distribution.ppf(
                    0.975, extreme_count + 1, N_PERMUTATIONS - extreme_count
                )
            )
        monte_carlo_se = float(
            np.sqrt(
                max(raw_tail_probability * (1.0 - raw_tail_probability), 0.0)
                / N_PERMUTATIONS
            )
        )
        test_rows.append(
            {
                "Outcome": label,
                "Column": column,
                "Alternative_for_real_STRING": direction,
                "Observed_real_STRING": observed_value,
                "Null_permutations": N_PERMUTATIONS,
                "Null_mean": float(np.mean(null_values)),
                "Null_SD": float(np.std(null_values, ddof=1)),
                "Null_median": float(np.median(null_values)),
                "Null_2.5_percentile": float(np.quantile(null_values, 0.025)),
                "Null_97.5_percentile": float(np.quantile(null_values, 0.975)),
                "Observed_minus_null_mean": float(
                    observed_value - np.mean(null_values)
                ),
                "Extreme_null_count": extreme_count,
                "Empirical_p_value": float(empirical_p),
                "Raw_extreme_fraction": float(raw_tail_probability),
                "Tail_probability_Clopper_Pearson_95CI_low": ci_low,
                "Tail_probability_Clopper_Pearson_95CI_high": ci_high,
                "Monte_Carlo_SE_raw_tail_probability": monte_carlo_se,
                "Favorable_percentile_of_real_STRING": percentile,
            }
        )

    empirical_tests = pd.DataFrame(test_rows)
    _perm_save_csv_atomic(empirical_tests, EMPIRICAL_TESTS_FILE)


# ------------------------------------------------------------
# 10. Figure, audit, manifest, and status
# ------------------------------------------------------------

FIGURE_PNG = FIGURE_DIR / "permuted_topology_control_observed_vs_null.png"
FIGURE_PDF = FIGURE_DIR / "permuted_topology_control_observed_vs_null.pdf"

if analysis_complete:
    try:
        import matplotlib.pyplot as plt

        real = observed_reference.set_index("Method").loc["NIBFS"]
        deg = observed_reference.set_index("Method").loc["DEG-only"]

        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

        axes[0].hist(null_summary["Mean_Jaccard"], bins=15, edgecolor="black")
        axes[0].axvline(
            real["Mean_Jaccard"],
            linestyle="--",
            linewidth=2,
            label="Real STRING NIBFS",
        )
        axes[0].axvline(
            deg["Mean_Jaccard"],
            linestyle=":",
            linewidth=2,
            label="DEG-only",
        )
        axes[0].set_xlabel("Mean repeat-level Jaccard")
        axes[0].set_ylabel("Number of fixed permutations")
        axes[0].set_title("A. Panel stability")
        axes[0].legend(frameon=False)

        axes[1].hist(null_summary["Mean_OOF_ROC_AUC"], bins=15, edgecolor="black")
        axes[1].axvline(
            real["Mean_OOF_ROC_AUC"],
            linestyle="--",
            linewidth=2,
            label="Real STRING NIBFS",
        )
        axes[1].axvline(
            deg["Mean_OOF_ROC_AUC"],
            linestyle=":",
            linewidth=2,
            label="DEG-only",
        )
        axes[1].set_xlabel("Mean repeated OOF ROC-AUC")
        axes[1].set_ylabel("Number of fixed permutations")
        axes[1].set_title("B. Predictive performance")
        axes[1].legend(frameon=False)

        fig.suptitle(
            "Real STRING topology versus fixed gene-label-permuted anchors"
        )
        fig.tight_layout()
        fig.savefig(FIGURE_PNG, dpi=600, bbox_inches="tight")
        fig.savefig(FIGURE_PDF, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        warnings.warn(f"Permuted-control figure was not generated: {exc}")

complete_fold_count = len(complete_fold_keys)

audit = pd.DataFrame(
    [
        {
            "Analysis": "Fixed gene-label-permuted STRING-topology control",
            "Development_samples": len(X_DEV),
            "Eligible_genes": X_DEV.shape[1],
            "Locked_internal_test_used": False,
            "Repeats": REPEATS,
            "Folds_per_repeat": FOLDS,
            "Expected_validation_folds": TOTAL_FOLDS,
            "Completed_validation_folds": complete_fold_count,
            "Complete": analysis_complete,
            "Permutations": N_PERMUTATIONS,
            "Reused_existing_permutations": BASE_PERMUTATIONS if REUSE_EXISTING_100 else 0,
            "New_permutations_computed_per_fold": (N_PERMUTATIONS - BASE_PERMUTATIONS) if REUSE_EXISTING_100 else N_PERMUTATIONS,
            "k": FINAL_K,
            "Classifier": "LR",
            "Limma_scope": "Refitted on each fold-training partition only",
            "Null_topology": (
                "STRING normalized-degree values reassigned to genes; fixed across folds"
            ),
            "Degree_value_distribution_preserved": True,
            "Real_STRING_panels_reconstructed_and_verified": bool(
                verification_table[
                    "Real_STRING_panel_exact_order_match"
                ].astype(bool).all()
            ),
            "Same_repeated_fold_seeds": True,
            "Frozen_primary_panel_changed": False,
            "Main_heldout_results_changed": False,
            "External_validation_results_changed": False,
            "LOCO_results_changed": False,
        }
    ]
)
_perm_save_csv_atomic(audit, OUTPUT_DIR / "permuted_analysis_audit.csv")

manifest = {
    **configuration_lock,
    "project_directory": str(PROJECT_DIR),
    "repeated_reference_directory": str(REPEATED_DIR),
    "output_directory": str(OUTPUT_DIR),
    "completed_validation_folds": complete_fold_count,
    "complete_repeats": complete_repeats,
    "complete": analysis_complete,
    "checkpoint_resume_enabled": True,
    "reused_base_100_control": REUSE_EXISTING_100,
    "base_100_directory": str(BASE_100_DIR),
    "rank_matrix_file": RANK_MATRIX_FILE.name,
    "real_STRING_panel_verification_passed": bool(
        verification_table[
            "Real_STRING_panel_exact_order_match"
        ].astype(bool).all()
    ),
    "scientific_role": (
        "Tests whether STRING-specific topology stabilizes NIBFS beyond an "
        "arbitrary fixed ranking with the same degree-value distribution"
    ),
    "output_files": sorted(
        str(path.relative_to(OUTPUT_DIR))
        for path in OUTPUT_DIR.rglob("*")
        if path.is_file()
    ),
}
_perm_save_json_atomic(manifest, OUTPUT_DIR / "permuted_run_manifest.json")

status_complete = OUTPUT_DIR / "PERMUTED_CONTROL_COMPLETE.txt"
status_incomplete = OUTPUT_DIR / "PERMUTED_CONTROL_INCOMPLETE.txt"
status_text = (
    f"Completed folds: {complete_fold_count}/{TOTAL_FOLDS}\n"
    f"Complete repeats: {len(complete_repeats)}/{REPEATS}\n"
    f"Permutations: {N_PERMUTATIONS}\n"
    f"New folds in this session: {new_fold_count}\n"
    f"Output directory: {OUTPUT_DIR}\n"
)
if analysis_complete:
    status_complete.write_text(status_text, encoding="utf-8")
    if status_incomplete.exists():
        status_incomplete.unlink()
else:
    status_incomplete.write_text(status_text, encoding="utf-8")
    if status_complete.exists():
        status_complete.unlink()


# ------------------------------------------------------------
# 11. User-facing report and inline figure display
# ------------------------------------------------------------

print("\n" + "=" * 78)
if analysis_complete:
    print("PERMUTED STRING-TOPOLOGY CONTROL COMPLETED")
else:
    print("PERMUTED STRING-TOPOLOGY CONTROL PARTIALLY COMPLETED — SAFE TO RESUME")
print("=" * 78)
print(f"Completed folds : {complete_fold_count}/{TOTAL_FOLDS}")
print(f"Complete repeats: {len(complete_repeats)}/{REPEATS}")
print(f"Permutations    : {N_PERMUTATIONS}")
print(f"New this session: {new_fold_count}")
print("Output          :", OUTPUT_DIR)
print(
    "Real panel verification:",
    "PASSED" if verification_table[
        "Real_STRING_panel_exact_order_match"
    ].astype(bool).all() else "FAILED",
)

if not null_summary.empty:
    print("\nCurrent null summary across complete repeats:")
    print(
        null_summary[
            [
                "Permutation_ID",
                "Mean_Jaccard",
                "Mean_OOF_ROC_AUC",
                "Mean_OOF_F1",
                "Mean_OOF_MCC",
                "Mean_OOF_Brier_score",
            ]
        ].describe().round(6)
    )

if analysis_complete and not empirical_tests.empty:
    print("\nFinal empirical tests:")
    print(
        empirical_tests[
            [
                "Outcome",
                "Observed_real_STRING",
                "Null_mean",
                "Null_2.5_percentile",
                "Null_97.5_percentile",
                "Empirical_p_value",
                "Tail_probability_Clopper_Pearson_95CI_low",
                "Tail_probability_Clopper_Pearson_95CI_high",
                "Favorable_percentile_of_real_STRING",
            ]
        ].round(6).to_string(index=False)
    )

if analysis_complete and DISPLAY_FINAL_FIGURE and FIGURE_PNG.exists():
    try:
        from IPython.display import Image as IPythonImage
        from IPython.display import Markdown, display

        display(Markdown("## Permuted-network control: observed versus null"))
        display(IPythonImage(filename=str(FIGURE_PNG), width=1200))
    except Exception as exc:
        warnings.warn(f"Final figure exists but could not be displayed inline: {exc}")
