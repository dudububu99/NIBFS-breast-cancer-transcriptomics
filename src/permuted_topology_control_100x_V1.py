# ============================================================
# PERMUTED-TOPOLOGY CONTROL
# 100 fixed gene-label permutations × repeated 10×5-fold CV
# k=20, logistic regression
# ============================================================
#
# Scientific question:
# Does the real STRING-derived topology stabilize NIBFS beyond the generic
# effect of adding any fixed ranking?
#
# Null control:
# - Keep the complete empirical degree-value distribution unchanged.
# - Randomly permute degree values across gene labels.
# - For each permutation, keep that permuted gene-to-degree assignment fixed
#   across all 50 validation folds.
# - Refit limma on each fold-training partition.
# - Combine fold-local statistical rank with the fixed permuted topology rank.
# - Fit LR on the selected top-20 genes and predict the fold-validation samples.
#
# This script:
# - uses only the 608-sample development set;
# - excludes the locked 152-sample internal test set;
# - does not modify the frozen top-20 panel;
# - does not alter repeated, LOCO, held-out, external, or Paper-2 results;
# - supports checkpoint resume by fold;
# - reads the completed real-STRING repeated analysis as the observed reference.
#
# Run from the executed notebook:
#
#   PERMUTED_PROJECT_DIR = str(PROJECT_DIR)
#   PERMUTED_N_PERMUTATIONS = 100
#   PERMUTED_MAX_NEW_FOLDS = 5       # None = all remaining folds
#   PERMUTED_FORCE_RERUN = False
#   %run -i "{PROJECT_DIR}/src/permuted_topology_control_100x_V1.py"
#
# The first processed fold can be slower because limma is run once for that
# fold. Every completed fold contains all requested permutations.
# ============================================================

from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path
from time import perf_counter
import gc
import hashlib
import json
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


# ------------------------------------------------------------
# 0. Configuration
# ------------------------------------------------------------

def _resolve_project_dir() -> Path:
    override = globals().get("PERMUTED_PROJECT_DIR")
    if override:
        candidate = Path(str(override)).expanduser().resolve()
        if (candidate / "src").exists():
            return candidate
        raise FileNotFoundError(f"PERMUTED_PROJECT_DIR tidak valid: {candidate}")

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
        "Folder proyek tidak ditemukan. Tetapkan PERMUTED_PROJECT_DIR "
        "ke folder utama proyek."
    )


PROJECT_DIR = _resolve_project_dir()
CONFIG_PATH = PROJECT_DIR / "config.yaml"
CFG = {}
if CONFIG_PATH.exists():
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        CFG = yaml.safe_load(handle) or {}

RANDOM_STATE = int(
    globals().get(
        "PERMUTED_RANDOM_STATE",
        CFG.get("project", {}).get("random_state", 42),
    )
)
N_PERMUTATIONS = int(globals().get("PERMUTED_N_PERMUTATIONS", 100))
REPEATS = int(globals().get("PERMUTED_REPEATS", 10))
FOLDS = int(globals().get("PERMUTED_FOLDS", 5))
FINAL_K = int(globals().get("PERMUTED_FINAL_K", 20))
MAX_NEW_FOLDS = globals().get("PERMUTED_MAX_NEW_FOLDS", None)
FORCE_RERUN = bool(globals().get("PERMUTED_FORCE_RERUN", False))
THRESHOLD = float(globals().get("PERMUTED_THRESHOLD", 0.5))
PERMUTATION_SEED_BASE = int(
    globals().get("PERMUTED_SEED_BASE", RANDOM_STATE + 100_000)
)

if N_PERMUTATIONS < 20:
    warnings.warn(
        f"N_PERMUTATIONS={N_PERMUTATIONS}. Untuk paper disarankan minimal 100.",
        RuntimeWarning,
    )
if REPEATS != 10 or FOLDS != 5 or FINAL_K != 20:
    warnings.warn(
        f"Desain aktif {REPEATS}×{FOLDS}, k={FINAL_K}; "
        "desain paper adalah 10×5, k=20.",
        RuntimeWarning,
    )
if MAX_NEW_FOLDS is not None:
    MAX_NEW_FOLDS = int(MAX_NEW_FOLDS)
    if MAX_NEW_FOLDS < 1:
        raise ValueError("PERMUTED_MAX_NEW_FOLDS harus None atau >=1.")

OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / f"PERMUTED_TOPOLOGY_CONTROL_{N_PERMUTATIONS}X_10X5_K20_LR"
)
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FIGURE_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

OBSERVED_DIR = PROJECT_DIR / "results" / "REPEATED_10X5_K20_LR"


# ------------------------------------------------------------
# 1. General helpers and data recovery
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
        ["train_test_split_assignments.csv", "discovery_train_test_assignments.csv"]
    )
    if expression_path is None or split_path is None:
        raise RuntimeError(
            "Matriks ekspresi atau pembagian train/test tidak ditemukan."
        )

    print("Loading development matrix:", expression_path, flush=True)
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
            f"Kolom split tidak lengkap: {list(split.columns)}"
        )

    is_dev = split[set_col].astype(str).str.lower().str.contains(
        r"model-development|development|train", regex=True
    )
    dev = split.loc[is_dev].copy()
    dev[gsm_col] = dev[gsm_col].astype(str)
    dev = dev.drop_duplicates(gsm_col).set_index(gsm_col)

    missing = dev.index.difference(expression.index)
    if len(missing):
        raise RuntimeError(
            f"{len(missing)} development sample tidak ditemukan di matriks ekspresi."
        )

    X = expression.loc[dev.index].copy()
    y = _labels_to_binary(dev[label_col])
    metadata = dev.copy()
    metadata.index = X.index
    return X, y, metadata


def _load_ppi_degree_from_disk() -> pd.DataFrame:
    path = _latest_path(["ppi_degree_table.csv"])
    if path is None:
        raise RuntimeError("ppi_degree_table.csv tidak ditemukan.")
    print("Loading PPI degree table:", path, flush=True)
    return pd.read_csv(path)


if "X_train" in globals() and "y_train" in globals():
    X_DEV = pd.DataFrame(globals()["X_train"]).copy()
    Y_DEV = np.asarray(globals()["y_train"], dtype=int)
    META_DEV = pd.DataFrame(
        globals().get("metadata_train", pd.DataFrame(index=X_DEV.index))
    ).copy()
else:
    X_DEV, Y_DEV, META_DEV = _load_development_from_disk()

if "ppi_degree" in globals():
    PPI_DEGREE = pd.DataFrame(globals()["ppi_degree"]).copy()
else:
    PPI_DEGREE = _load_ppi_degree_from_disk()

X_DEV.index = X_DEV.index.astype(str)
X_DEV.columns = X_DEV.columns.astype(str)
Y_DEV = np.asarray(Y_DEV, dtype=int)

if X_DEV.shape != (608, 17220):
    raise RuntimeError(
        "Kontrol permutasi harus memakai development matrix (608, 17220), "
        f"tetapi ditemukan {X_DEV.shape}."
    )
if set(np.unique(Y_DEV)) != {0, 1}:
    raise RuntimeError("Label development harus berisi kelas 0 dan 1.")
if X_DEV.index.duplicated().any():
    raise RuntimeError("Sample ID pada development set tidak unik.")


# ------------------------------------------------------------
# 2. limma, ranking, and prediction
# ------------------------------------------------------------

def run_limma_rpy2(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.conversion import localconverter
    except Exception as exc:
        raise ImportError(
            "Script membutuhkan rpy2 dan R package limma."
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
    return result.sort_values(["Rank_stat", "Gene"]).reset_index(drop=True)


def prepare_real_ppi_table(
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
            f"PPI table tidak memiliki Gene/Degree: {list(ppi_degree.columns)}"
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

    out = pd.DataFrame({"Gene": genes}).merge(ppi, on="Gene", how="left")
    out[["Degree", "Normalized_degree"]] = out[
        ["Degree", "Normalized_degree"]
    ].fillna(0.0)
    out["Rank_topo"] = out["Normalized_degree"].rank(
        method="average", ascending=False
    )
    return out.sort_values("Gene").reset_index(drop=True)


def make_permuted_topology(
    real_ppi: pd.DataFrame,
    permutation_id: int,
) -> pd.DataFrame:
    # Fixed across all folds for this permutation.
    rng = np.random.default_rng(PERMUTATION_SEED_BASE + permutation_id)
    values = real_ppi["Normalized_degree"].to_numpy(float).copy()
    permuted_values = rng.permutation(values)

    out = pd.DataFrame(
        {
            "Gene": real_ppi["Gene"].astype(str).to_numpy(),
            "Permuted_normalized_degree": permuted_values,
        }
    )
    out["Permuted_rank_topo"] = out["Permuted_normalized_degree"].rank(
        method="average", ascending=False
    )
    return out


def select_permuted_nibfs_panel(
    limma: pd.DataFrame,
    permuted_topology: pd.DataFrame,
    k: int,
) -> tuple[list[str], pd.DataFrame]:
    p = len(limma)
    out = limma.merge(
        permuted_topology[
            ["Gene", "Permuted_normalized_degree", "Permuted_rank_topo"]
        ],
        on="Gene",
        how="left",
        validate="one_to_one",
    )
    if out["Permuted_rank_topo"].isna().any():
        raise RuntimeError("Ada gen limma yang tidak memiliki rank topologi permutasi.")

    out["Borda_stat"] = p - out["Rank_stat"] + 1
    out["Borda_topo_permuted"] = p - out["Permuted_rank_topo"] + 1
    out["Borda_score_permuted"] = (
        out["Borda_stat"] + out["Borda_topo_permuted"]
    )
    out = out.sort_values(
        ["Borda_score_permuted", "Rank_stat", "Gene"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    out["Rank_permuted_NIBFS"] = np.arange(1, len(out) + 1)
    genes = out.head(k)["Gene"].astype(str).tolist()
    if len(genes) != k or len(set(genes)) != k:
        raise RuntimeError(f"Panel permutasi tidak menghasilkan {k} gen unik.")
    return genes, out.head(k).copy()


REAL_PPI = prepare_real_ppi_table(PPI_DEGREE, X_DEV.columns.tolist())
PERMUTED_TOPOLOGIES = {
    permutation_id: make_permuted_topology(REAL_PPI, permutation_id)
    for permutation_id in range(1, N_PERMUTATIONS + 1)
}


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
        "ROC_AUC": float(roc_auc_score(y, p)),
        "Accuracy": float(accuracy_score(y, pred)),
        "Balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "Sensitivity": float(recall_score(y, pred, zero_division=0)),
        "Specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y, pred)),
        "Brier_score": float(brier_score_loss(y, p)),
    }


# ------------------------------------------------------------
# 3. Checkpoint helpers
# ------------------------------------------------------------

def _prefix(repeat: int, fold: int) -> Path:
    return CHECKPOINT_DIR / f"repeat_{repeat:02d}_fold_{fold:02d}"


def _files(repeat: int, fold: int) -> dict[str, Path]:
    prefix = _prefix(repeat, fold)
    return {
        "predictions": prefix.with_name(prefix.name + "_predictions.csv.gz"),
        "panels": prefix.with_name(prefix.name + "_panels.csv.gz"),
        "limma": prefix.with_name(prefix.name + "_limma.csv.gz"),
        "runtime": prefix.with_name(prefix.name + "_runtime.json"),
        "done": prefix.with_name(prefix.name + "_DONE.json"),
    }


def _complete(repeat: int, fold: int) -> bool:
    paths = _files(repeat, fold)
    return all(paths[key].exists() for key in paths)


def _remove(repeat: int, fold: int) -> None:
    for path in _files(repeat, fold).values():
        if path.exists():
            path.unlink()


def _save_json_atomic(data: dict, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temp.replace(path)


def _save_csv_gz_atomic(frame: pd.DataFrame, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, compression="gzip")
    temp.replace(path)


# ------------------------------------------------------------
# 4. Observed real-STRING reference
# ------------------------------------------------------------

required_observed = {
    "performance": OBSERVED_DIR / "repeated_oof_performance_summary.csv",
    "stability": OBSERVED_DIR / "repeated_stability_summary.csv",
    "by_repeat_perf": OBSERVED_DIR / "repeated_oof_metrics_by_repeat.csv",
    "by_repeat_stab": OBSERVED_DIR / "repeated_stability_by_repeat.csv",
}
missing_observed = [str(p) for p in required_observed.values() if not p.exists()]
if missing_observed:
    raise FileNotFoundError(
        "Observed repeated real-STRING output belum lengkap:\n"
        + "\n".join(missing_observed)
    )

observed_performance_summary = pd.read_csv(required_observed["performance"])
observed_stability_summary = pd.read_csv(required_observed["stability"])
observed_perf_by_repeat = pd.read_csv(required_observed["by_repeat_perf"])
observed_stab_by_repeat = pd.read_csv(required_observed["by_repeat_stab"])

obs_perf_row = observed_performance_summary.loc[
    observed_performance_summary["Method"].astype(str) == "NIBFS"
]
obs_stab_row = observed_stability_summary.loc[
    observed_stability_summary["Method"].astype(str) == "NIBFS"
]
if len(obs_perf_row) != 1 or len(obs_stab_row) != 1:
    raise RuntimeError("Baris observed NIBFS tidak unik pada output repeated.")

OBSERVED_AUC = float(obs_perf_row.iloc[0]["ROC_AUC_mean"])
OBSERVED_F1 = float(obs_perf_row.iloc[0]["F1_mean"])
OBSERVED_MCC = float(obs_perf_row.iloc[0]["MCC_mean"])
OBSERVED_BRIER = float(obs_perf_row.iloc[0]["Brier_score_mean"])
OBSERVED_JACCARD = float(
    obs_stab_row.iloc[0]["Mean_of_repeat_mean_Jaccard"]
)

if int(obs_perf_row.iloc[0]["ROC_AUC_count"]) != REPEATS:
    raise RuntimeError("Observed repeated NIBFS belum memiliki 10 repeat.")
if int(obs_stab_row.iloc[0]["Repeats"]) != REPEATS:
    raise RuntimeError("Observed repeated stability belum memiliki 10 repeat.")


# ------------------------------------------------------------
# 5. Execute/resume fold-major null control
# ------------------------------------------------------------

TOTAL_FOLDS = REPEATS * FOLDS
if FORCE_RERUN:
    print("PERMUTED_FORCE_RERUN=True: menghapus checkpoint kontrol permutasi.")
    for repeat in range(1, REPEATS + 1):
        for fold in range(1, FOLDS + 1):
            _remove(repeat, fold)

completed_before = sum(
    _complete(repeat, fold)
    for repeat in range(1, REPEATS + 1)
    for fold in range(1, FOLDS + 1)
)

print("=" * 78)
print("PERMUTED-TOPOLOGY CONTROL")
print("=" * 78)
print("PROJECT_DIR          :", PROJECT_DIR)
print("OUTPUT_DIR           :", OUTPUT_DIR)
print("Development data     :", X_DEV.shape)
print("Permutations         :", N_PERMUTATIONS)
print("Design               :", f"{REPEATS}×{FOLDS} = {TOTAL_FOLDS} folds")
print("Observed NIBFS AUC   :", f"{OBSERVED_AUC:.6f}")
print("Observed NIBFS Jaccard:", f"{OBSERVED_JACCARD:.6f}")
print("Existing checkpoints :", f"{completed_before}/{TOTAL_FOLDS}")

new_fold_count = 0
session_times: list[float] = []
stop_requested = False

for repeat in range(1, REPEATS + 1):
    cv = StratifiedKFold(
        n_splits=FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE + repeat - 1,
    )

    for fold, (fit_idx, val_idx) in enumerate(cv.split(X_DEV, Y_DEV), 1):
        global_position = (repeat - 1) * FOLDS + fold
        paths = _files(repeat, fold)

        if _complete(repeat, fold):
            print(
                f"[{global_position:02d}/{TOTAL_FOLDS}] repeat {repeat}, "
                f"fold {fold}: checkpoint found — skipped",
                flush=True,
            )
            continue

        if MAX_NEW_FOLDS is not None and new_fold_count >= MAX_NEW_FOLDS:
            stop_requested = True
            break

        start = perf_counter()
        fold_seed = RANDOM_STATE + (repeat - 1) * FOLDS + fold
        X_fit = X_DEV.iloc[fit_idx]
        y_fit = Y_DEV[fit_idx]
        X_val = X_DEV.iloc[val_idx]
        y_val = Y_DEV[val_idx]

        print("\n" + "-" * 78)
        print(
            f"[{global_position:02d}/{TOTAL_FOLDS}] REPEAT {repeat}/"
            f"{REPEATS} — FOLD {fold}/{FOLDS}"
        )
        print("  [1/3] Fold-local limma ...", flush=True)
        limma_table = run_limma_rpy2(X_fit, y_fit)

        print(
            f"  [2/3] {N_PERMUTATIONS} fixed gene-label topology permutations ...",
            flush=True,
        )
        prediction_rows: list[dict] = []
        panel_rows: list[dict] = []

        for permutation_id in range(1, N_PERMUTATIONS + 1):
            genes, panel_detail = select_permuted_nibfs_panel(
                limma_table,
                PERMUTED_TOPOLOGIES[permutation_id],
                FINAL_K,
            )

            model = create_lr(fold_seed)
            model.fit(X_fit[genes], y_fit)
            probability = model.predict_proba(X_val[genes])[:, 1]

            for sample_id, true_label, prob in zip(
                X_val.index.astype(str), y_val, probability
            ):
                prediction_rows.append(
                    {
                        "Permutation": permutation_id,
                        "Repeat": repeat,
                        "Fold": fold,
                        "Sample_ID": sample_id,
                        "True_Label": int(true_label),
                        "Probability": float(prob),
                    }
                )

            for rank, row in panel_detail.reset_index(drop=True).iterrows():
                panel_rows.append(
                    {
                        "Permutation": permutation_id,
                        "Repeat": repeat,
                        "Fold": fold,
                        "Selection_rank": rank + 1,
                        "Gene": str(row["Gene"]),
                        "Rank_stat": float(row["Rank_stat"]),
                        "Permuted_rank_topo": float(
                            row["Permuted_rank_topo"]
                        ),
                        "Borda_score_permuted": float(
                            row["Borda_score_permuted"]
                        ),
                    }
                )

            if permutation_id % 10 == 0 or permutation_id == N_PERMUTATIONS:
                print(
                    f"      completed {permutation_id}/{N_PERMUTATIONS}",
                    flush=True,
                )

        print("  [3/3] Saving fold checkpoint ...", flush=True)
        elapsed = perf_counter() - start

        _save_csv_gz_atomic(
            pd.DataFrame(prediction_rows), paths["predictions"]
        )
        _save_csv_gz_atomic(pd.DataFrame(panel_rows), paths["panels"])
        _save_csv_gz_atomic(limma_table, paths["limma"])
        _save_json_atomic(
            {
                "Repeat": repeat,
                "Fold": fold,
                "Elapsed_seconds": elapsed,
                "Elapsed_minutes": elapsed / 60.0,
                "Permutations": N_PERMUTATIONS,
                "Training_samples": len(fit_idx),
                "Validation_samples": len(val_idx),
                "Fold_seed": fold_seed,
                "Completed": True,
            },
            paths["runtime"],
        )
        _save_json_atomic(
            {
                "status": "complete",
                "repeat": repeat,
                "fold": fold,
                "permutations": N_PERMUTATIONS,
            },
            paths["done"],
        )

        new_fold_count += 1
        session_times.append(elapsed)
        complete_now = sum(
            _complete(r, f)
            for r in range(1, REPEATS + 1)
            for f in range(1, FOLDS + 1)
        )
        remaining = TOTAL_FOLDS - complete_now
        eta = (
            remaining * float(np.mean(session_times)) / 60.0
            if session_times else np.nan
        )
        print(
            f"  completed in {elapsed/60.0:.2f} min | "
            f"checkpoints={complete_now}/{TOTAL_FOLDS} | "
            f"estimated remaining={eta:.1f} min",
            flush=True,
        )

        del X_fit, X_val, y_fit, y_val, limma_table
        del prediction_rows, panel_rows
        gc.collect()

    if stop_requested:
        break


# ------------------------------------------------------------
# 6. Partial-status report; aggregate only when 50/50 complete
# ------------------------------------------------------------

complete_fold_count = sum(
    _complete(repeat, fold)
    for repeat in range(1, REPEATS + 1)
    for fold in range(1, FOLDS + 1)
)

status = {
    "Analysis": "Fixed gene-label-permuted topology control",
    "Development_shape": list(X_DEV.shape),
    "N_permutations": N_PERMUTATIONS,
    "Repeats": REPEATS,
    "Folds": FOLDS,
    "Expected_folds": TOTAL_FOLDS,
    "Completed_folds": complete_fold_count,
    "Complete": complete_fold_count == TOTAL_FOLDS,
    "Observed_real_STRING_AUC": OBSERVED_AUC,
    "Observed_real_STRING_Jaccard": OBSERVED_JACCARD,
    "Frozen_panel_changed": False,
    "Locked_test_used": False,
}
_save_json_atomic(status, OUTPUT_DIR / "permuted_control_status.json")

print("\n" + "=" * 78)
print("PERMUTED CONTROL STATUS")
print("=" * 78)
print(f"Completed folds : {complete_fold_count}/{TOTAL_FOLDS}")
print(f"New this session: {new_fold_count}")

if complete_fold_count < TOTAL_FOLDS:
    print(
        "Checkpoint tersimpan. Ringkasan final dibuat otomatis setelah 50/50 fold."
    )
else:
    print("All folds complete. Aggregating final null distribution ...")

    prediction_parts = []
    panel_parts = []
    runtime_rows = []
    for repeat in range(1, REPEATS + 1):
        for fold in range(1, FOLDS + 1):
            paths = _files(repeat, fold)
            prediction_parts.append(pd.read_csv(paths["predictions"]))
            panel_parts.append(pd.read_csv(paths["panels"]))
            runtime_rows.append(
                json.loads(paths["runtime"].read_text(encoding="utf-8"))
            )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    panels = pd.concat(panel_parts, ignore_index=True)
    runtime_table = pd.DataFrame(runtime_rows)

    predictions.to_csv(
        OUTPUT_DIR / "permuted_all_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    panels.to_csv(
        OUTPUT_DIR / "permuted_all_selected_panels.csv.gz",
        index=False,
        compression="gzip",
    )
    runtime_table.to_csv(
        OUTPUT_DIR / "permuted_fold_runtime.csv",
        index=False,
    )

    # OOF performance for every permutation and repeat.
    perf_rows = []
    for (permutation_id, repeat), group in predictions.groupby(
        ["Permutation", "Repeat"], sort=True
    ):
        if len(group) != len(X_DEV):
            raise RuntimeError(
                f"Permutation {permutation_id}, repeat {repeat}: "
                f"expected {len(X_DEV)} OOF predictions, found {len(group)}."
            )
        if group["Sample_ID"].nunique() != len(X_DEV):
            raise RuntimeError(
                f"Permutation {permutation_id}, repeat {repeat}: "
                "OOF sample IDs are not unique."
            )
        metrics = classification_metrics(
            group["True_Label"].to_numpy(int),
            group["Probability"].to_numpy(float),
            THRESHOLD,
        )
        perf_rows.append(
            {
                "Permutation": int(permutation_id),
                "Repeat": int(repeat),
                **metrics,
            }
        )

    perf_by_repeat = pd.DataFrame(perf_rows).sort_values(
        ["Permutation", "Repeat"]
    )
    perf_by_repeat.to_csv(
        OUTPUT_DIR / "permuted_oof_metrics_by_repeat.csv",
        index=False,
    )

    # Within-repeat panel stability.
    stability_rows = []
    pairwise_rows = []
    for (permutation_id, repeat), group in panels.groupby(
        ["Permutation", "Repeat"], sort=True
    ):
        panel_by_fold = {
            int(fold): set(fold_group["Gene"].astype(str))
            for fold, fold_group in group.groupby("Fold")
        }
        if len(panel_by_fold) != FOLDS:
            raise RuntimeError(
                f"Permutation {permutation_id}, repeat {repeat}: "
                f"expected {FOLDS} panels, found {len(panel_by_fold)}."
            )

        values = []
        for fold_1, fold_2 in combinations(sorted(panel_by_fold), 2):
            a = panel_by_fold[fold_1]
            b = panel_by_fold[fold_2]
            jaccard = len(a & b) / len(a | b)
            values.append(jaccard)
            pairwise_rows.append(
                {
                    "Permutation": int(permutation_id),
                    "Repeat": int(repeat),
                    "Fold_1": fold_1,
                    "Fold_2": fold_2,
                    "Jaccard": float(jaccard),
                }
            )

        stability_rows.append(
            {
                "Permutation": int(permutation_id),
                "Repeat": int(repeat),
                "Mean_Jaccard": float(np.mean(values)),
                "SD_Jaccard": float(np.std(values, ddof=1)),
                "Median_Jaccard": float(np.median(values)),
                "Minimum_Jaccard": float(np.min(values)),
                "Maximum_Jaccard": float(np.max(values)),
            }
        )

    stability_by_repeat = pd.DataFrame(stability_rows).sort_values(
        ["Permutation", "Repeat"]
    )
    pairwise_jaccard = pd.DataFrame(pairwise_rows).sort_values(
        ["Permutation", "Repeat", "Fold_1", "Fold_2"]
    )
    stability_by_repeat.to_csv(
        OUTPUT_DIR / "permuted_stability_by_repeat.csv",
        index=False,
    )
    pairwise_jaccard.to_csv(
        OUTPUT_DIR / "permuted_pairwise_jaccard.csv.gz",
        index=False,
        compression="gzip",
    )

    # One null statistic per permutation, averaged across the same 10 repeats.
    null_summary = (
        perf_by_repeat.groupby("Permutation")
        .agg(
            Null_ROC_AUC=("ROC_AUC", "mean"),
            Null_F1=("F1", "mean"),
            Null_MCC=("MCC", "mean"),
            Null_Brier_score=("Brier_score", "mean"),
        )
        .join(
            stability_by_repeat.groupby("Permutation").agg(
                Null_Mean_Jaccard=("Mean_Jaccard", "mean")
            )
        )
        .reset_index()
        .sort_values("Permutation")
    )
    null_summary.to_csv(
        OUTPUT_DIR / "permuted_null_summary_by_permutation.csv",
        index=False,
    )

    def empirical_p_greater(observed: float, null_values) -> float:
        values = np.asarray(null_values, dtype=float)
        return float((1 + np.sum(values >= observed)) / (len(values) + 1))

    def empirical_p_lower(observed: float, null_values) -> float:
        values = np.asarray(null_values, dtype=float)
        return float((1 + np.sum(values <= observed)) / (len(values) + 1))

    comparison_rows = []
    metric_specs = [
        ("Mean_Jaccard", OBSERVED_JACCARD, "Null_Mean_Jaccard", "greater"),
        ("ROC_AUC", OBSERVED_AUC, "Null_ROC_AUC", "greater"),
        ("F1", OBSERVED_F1, "Null_F1", "greater"),
        ("MCC", OBSERVED_MCC, "Null_MCC", "greater"),
        ("Brier_score", OBSERVED_BRIER, "Null_Brier_score", "lower"),
    ]

    for metric, observed, null_col, direction in metric_specs:
        null_values = null_summary[null_col].to_numpy(float)
        p_value = (
            empirical_p_greater(observed, null_values)
            if direction == "greater"
            else empirical_p_lower(observed, null_values)
        )
        percentile = float(
            100.0 * np.mean(null_values <= observed)
            if direction == "greater"
            else 100.0 * np.mean(null_values >= observed)
        )
        comparison_rows.append(
            {
                "Metric": metric,
                "Observed_real_STRING": observed,
                "Null_mean": float(np.mean(null_values)),
                "Null_SD": float(np.std(null_values, ddof=1)),
                "Null_median": float(np.median(null_values)),
                "Null_minimum": float(np.min(null_values)),
                "Null_maximum": float(np.max(null_values)),
                "Expected_direction": direction,
                "Observed_percentile_in_favorable_direction": percentile,
                "Empirical_p_value": p_value,
                "N_permutations": len(null_values),
            }
        )

    observed_vs_null = pd.DataFrame(comparison_rows)
    observed_vs_null.to_csv(
        OUTPUT_DIR / "observed_STRING_vs_permuted_topology.csv",
        index=False,
    )

    # Gene recurrence across all null panels, provided for audit only.
    recurrence_rows = []
    total_null_panels = N_PERMUTATIONS * TOTAL_FOLDS
    for gene, group in panels.groupby("Gene"):
        recurrence_rows.append(
            {
                "Gene": str(gene),
                "Null_selection_frequency": int(len(group)),
                "Total_null_panels": int(total_null_panels),
                "Null_selection_proportion": float(
                    len(group) / total_null_panels
                ),
                "Mean_selection_rank": float(
                    group["Selection_rank"].mean()
                ),
            }
        )
    pd.DataFrame(recurrence_rows).sort_values(
        ["Null_selection_frequency", "Mean_selection_rank", "Gene"],
        ascending=[False, True, True],
    ).to_csv(
        OUTPUT_DIR / "permuted_gene_selection_frequency.csv",
        index=False,
    )

    # Figures generated only from final run outputs.
    try:
        import matplotlib.pyplot as plt

        # Stability null distribution.
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.hist(
            null_summary["Null_Mean_Jaccard"],
            bins=20,
            edgecolor="black",
            alpha=0.8,
        )
        ax.axvline(
            OBSERVED_JACCARD,
            linestyle="--",
            linewidth=2,
            label=f"Real STRING = {OBSERVED_JACCARD:.4f}",
        )
        ax.set_xlabel("Mean Jaccard across 10 repeats")
        ax.set_ylabel("Number of topology permutations")
        ax.set_title("Real STRING topology versus fixed permuted rankings")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            FIGURE_DIR / "permuted_topology_stability_null.pdf"
        )
        fig.savefig(
            FIGURE_DIR / "permuted_topology_stability_null.png",
            dpi=600,
        )
        plt.close(fig)

        # Predictive null distribution.
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.hist(
            null_summary["Null_ROC_AUC"],
            bins=20,
            edgecolor="black",
            alpha=0.8,
        )
        ax.axvline(
            OBSERVED_AUC,
            linestyle="--",
            linewidth=2,
            label=f"Real STRING = {OBSERVED_AUC:.4f}",
        )
        ax.set_xlabel("Mean repeated OOF ROC-AUC")
        ax.set_ylabel("Number of topology permutations")
        ax.set_title("Predictive performance under topology-label permutation")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            FIGURE_DIR / "permuted_topology_auc_null.pdf"
        )
        fig.savefig(
            FIGURE_DIR / "permuted_topology_auc_null.png",
            dpi=600,
        )
        plt.close(fig)
    except Exception as figure_error:
        warnings.warn(f"Figure generation failed: {figure_error}")

    # Reproducibility manifest.
    topology_hash = hashlib.sha256(
        REAL_PPI[
            ["Gene", "Degree", "Normalized_degree"]
        ].to_csv(index=False).encode("utf-8")
    ).hexdigest()

    manifest = {
        "analysis": "Fixed gene-label-permuted topology control",
        "scientific_question": (
            "Does real STRING topology stabilize selection beyond adding "
            "an arbitrary fixed ranking?"
        ),
        "project_directory": str(PROJECT_DIR),
        "output_directory": str(OUTPUT_DIR),
        "development_shape": list(X_DEV.shape),
        "locked_internal_test_used": False,
        "frozen_panel_changed": False,
        "n_permutations": N_PERMUTATIONS,
        "repeats": REPEATS,
        "folds": FOLDS,
        "completed_folds": complete_fold_count,
        "k": FINAL_K,
        "classifier": "logistic regression",
        "threshold": THRESHOLD,
        "random_state_base": RANDOM_STATE,
        "permutation_seed_base": PERMUTATION_SEED_BASE,
        "null_design": (
            "Empirical normalized-degree values permuted across gene labels; "
            "one fixed assignment per permutation across all 50 folds."
        ),
        "degree_distribution_preserved": True,
        "node_specific_degree_preserved": False,
        "observed_reference_directory": str(OBSERVED_DIR),
        "real_topology_sha256": topology_hash,
        "complete": True,
    }
    _save_json_atomic(
        manifest, OUTPUT_DIR / "permuted_control_manifest.json"
    )

    (OUTPUT_DIR / "PERMUTED_TOPOLOGY_CONTROL_COMPLETE.txt").write_text(
        "PERMUTED TOPOLOGY CONTROL COMPLETE\n"
        f"Permutations: {N_PERMUTATIONS}\n"
        f"Repeated folds: {TOTAL_FOLDS}\n"
        f"Observed STRING Jaccard: {OBSERVED_JACCARD:.8f}\n"
        f"Observed STRING ROC-AUC: {OBSERVED_AUC:.8f}\n",
        encoding="utf-8",
    )

    print("\nFINAL OBSERVED-VS-NULL COMPARISON")
    print(observed_vs_null.to_string(index=False))
    print("\nOutput:", OUTPUT_DIR)
    print("PERMUTED TOPOLOGY CONTROL COMPLETE")
