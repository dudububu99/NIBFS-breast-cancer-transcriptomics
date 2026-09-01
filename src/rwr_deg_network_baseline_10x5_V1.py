# ============================================================
# RWR-DEG NETWORK BASELINE — REPEATED 10 x 5-FOLD, K=20, LR
# ============================================================
#
# Scientific role
# ---------------
# This script adds one apples-to-apples network-aware baseline to the
# completed NIBFS repeated analysis:
#
#   fold-training samples
#       -> fold-local limma statistical scores
#       -> weighted Random Walk with Restart (RWR) on the fixed
#          high-confidence STRING graph
#       -> top-20 genes
#       -> logistic-regression validation on the held-out fold
#
# Prespecified design
# -------------
# - development set only: 608 samples x 17,220 eligible genes
# - exact 10 x 5 partitions reused from REPEATED_10X5_K20_LR
# - main restart probability: 0.50
# - sensitivity restart probabilities: 0.30 and 0.70
# - k = 20
# - classifier: LR with the same specification as repeated V2
# - threshold: 0.50
# - no change to the frozen top-20 NIBFS panel
# - no change to held-out or external-validation results
# - checkpoint/resume enabled
#
# Recommended Colab usage
# ------------------------
#   RWR_PROJECT_DIR = str(PROJECT_DIR)
#   RWR_MAX_NEW_FOLDS = 1      # benchmark first
#   RWR_FORCE_RERUN = False
#   %run -i src/rwr_deg_network_baseline_10x5_V1.py
#
# Then complete repeat 1 with RWR_MAX_NEW_FOLDS = 4, and use 5-fold
# blocks afterwards.
# ============================================================

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from time import perf_counter
import gc
import json
import math
import warnings

import numpy as np
import pandas as pd
import yaml

from scipy import sparse
from scipy.stats import wilcoxon
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------
# 0. Configuration
# ------------------------------------------------------------

def _resolve_project_dir() -> Path:
    override = globals().get("RWR_PROJECT_DIR")
    if override:
        candidate = Path(str(override)).expanduser().resolve()
        if (candidate / "src").exists():
            return candidate
        raise FileNotFoundError(f"RWR_PROJECT_DIR tidak valid: {candidate}")

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
        "Folder proyek tidak dapat ditentukan. Tetapkan RWR_PROJECT_DIR "
        "ke folder utama proyek sebelum menjalankan script."
    )


PROJECT_DIR = _resolve_project_dir()
CONFIG_PATH = PROJECT_DIR / "config.yaml"
CFG: dict = {}
if CONFIG_PATH.exists():
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        CFG = yaml.safe_load(handle) or {}

RANDOM_STATE = int(
    globals().get(
        "RWR_RANDOM_STATE",
        CFG.get("project", {}).get("random_state", 42),
    )
)
FINAL_K = int(
    globals().get(
        "RWR_FINAL_K",
        CFG.get("project", {}).get("final_k", 20),
    )
)
REPEATS = int(globals().get("RWR_REPEATS", 10))
FOLDS = int(globals().get("RWR_FOLDS", 5))
MAX_NEW_FOLDS = globals().get("RWR_MAX_NEW_FOLDS", None)
FORCE_RERUN = bool(globals().get("RWR_FORCE_RERUN", False))
DISPLAY_FINAL_FIGURE = bool(globals().get("RWR_DISPLAY_FINAL_FIGURE", True))
DEFAULT_THRESHOLD = float(globals().get("RWR_THRESHOLD", 0.5))

MAIN_RESTART = float(globals().get("RWR_MAIN_RESTART", 0.50))
RESTART_VALUES = tuple(
    float(x)
    for x in globals().get("RWR_RESTART_VALUES", (0.30, 0.50, 0.70))
)
RWR_TOLERANCE = float(globals().get("RWR_TOLERANCE", 1e-10))
RWR_MAX_ITER = int(globals().get("RWR_MAX_ITER", 1000))
USE_STRING_WEIGHTS = bool(globals().get("RWR_USE_STRING_WEIGHTS", True))

if MAX_NEW_FOLDS is not None:
    MAX_NEW_FOLDS = int(MAX_NEW_FOLDS)
    if MAX_NEW_FOLDS < 1:
        raise ValueError("RWR_MAX_NEW_FOLDS harus None atau bilangan >= 1.")
if FINAL_K != 20:
    raise ValueError(f"Script dikunci untuk k=20, tetapi ditemukan k={FINAL_K}.")
if REPEATS != 10 or FOLDS != 5:
    warnings.warn(
        f"Desain aktif {REPEATS} x {FOLDS}; desain paper adalah 10 x 5.",
        RuntimeWarning,
    )
if MAIN_RESTART not in RESTART_VALUES:
    raise ValueError("RWR_MAIN_RESTART harus terdapat dalam RWR_RESTART_VALUES.")
if len(set(RESTART_VALUES)) != len(RESTART_VALUES):
    raise ValueError("RWR_RESTART_VALUES tidak boleh mengandung duplikat.")
if any(not (0.0 < r < 1.0) for r in RESTART_VALUES):
    raise ValueError("Semua restart probability harus berada di antara 0 dan 1.")

OUTPUT_DIR = PROJECT_DIR / "results" / "RWR_DEG_NETWORK_BASELINE_10X5_K20_LR"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FIGURE_DIR = OUTPUT_DIR / "figures"
CACHE_DIR = OUTPUT_DIR / "fold_limma_cache"
for directory in [OUTPUT_DIR, CHECKPOINT_DIR, FIGURE_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

REPEATED_DIR = PROJECT_DIR / "results" / "REPEATED_10X5_K20_LR"
REPEATED_ASSIGNMENTS_FILE = REPEATED_DIR / "repeated_fold_assignments.csv"


# ------------------------------------------------------------
# 1. Generic helpers and data recovery
# ------------------------------------------------------------

def _latest_path(patterns: list[str]) -> Path | None:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(PROJECT_DIR.rglob(pattern))
    matches = [path for path in matches if path.is_file()]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def _find_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {str(column).strip().lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lookup:
            return lookup[key]
    return None


def _labels_to_binary(values) -> np.ndarray:
    series = pd.Series(values)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all() and set(numeric.astype(int).unique()).issubset({0, 1}):
        return numeric.astype(int).to_numpy()
    text = series.astype(str).str.strip().str.lower()
    out = np.full(len(text), np.nan)
    out[text.str.contains(r"cancer|tumou?r|malignan|carcinoma", regex=True)] = 1
    out[text.str.contains(r"normal|control|healthy", regex=True)] = 0
    if np.isnan(out).any():
        unknown = sorted(series[np.isnan(out)].astype(str).unique())
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
            "Matriks development atau file split tidak ditemukan. Jalankan sel load "
            "hasil utama sampai X_train/y_train aktif; analisis utama tidak perlu diulang."
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
        raise KeyError(f"Kolom split tidak lengkap: {list(split.columns)}")

    is_development = split[set_col].astype(str).str.lower().str.contains(
        r"model-development|development|train", regex=True
    )
    dev = split.loc[is_development].copy()
    dev[gsm_col] = dev[gsm_col].astype(str)
    dev = dev.drop_duplicates(gsm_col).set_index(gsm_col)

    missing = dev.index.difference(expression.index)
    if len(missing):
        raise RuntimeError(
            f"{len(missing)} sampel development tidak ditemukan dalam matriks ekspresi."
        )

    X = expression.loc[dev.index].copy()
    y = _labels_to_binary(dev[label_col])
    metadata = dev.reset_index().rename(columns={gsm_col: "GSM_ID"})
    metadata.index = X.index
    return X, y, metadata


if "X_train" in globals() and "y_train" in globals():
    X_DEV = pd.DataFrame(globals()["X_train"]).copy()
    Y_DEV = np.asarray(globals()["y_train"], dtype=int)
    if "metadata_train" in globals():
        META_DEV = pd.DataFrame(globals()["metadata_train"]).copy()
    else:
        META_DEV = pd.DataFrame(index=X_DEV.index)
else:
    X_DEV, Y_DEV, META_DEV = _load_development_from_disk()

X_DEV.index = X_DEV.index.astype(str)
X_DEV.columns = X_DEV.columns.astype(str)
Y_DEV = np.asarray(Y_DEV, dtype=int)

if X_DEV.shape != (608, 17220):
    raise RuntimeError(
        "RWR-DEG harus memakai development matrix (608, 17220), "
        f"tetapi ditemukan {X_DEV.shape}."
    )
if len(X_DEV) != len(Y_DEV):
    raise ValueError("Jumlah baris X_DEV tidak sama dengan panjang Y_DEV.")
if X_DEV.index.duplicated().any():
    raise ValueError("Sample ID development tidak unik.")
if set(np.unique(Y_DEV)) != {0, 1}:
    raise ValueError("Label development harus mengandung kelas 0 dan 1.")
if not np.isfinite(X_DEV.to_numpy(dtype=float)).all():
    raise ValueError("Matriks development mengandung nilai non-finite.")

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


# ------------------------------------------------------------
# 2. Reuse the exact repeated 10 x 5 fold assignments
# ------------------------------------------------------------

def _load_repeated_assignments() -> pd.DataFrame:
    if not REPEATED_ASSIGNMENTS_FILE.exists():
        raise FileNotFoundError(
            "Exact repeated assignments tidak ditemukan:\n"
            f"{REPEATED_ASSIGNMENTS_FILE}\n"
            "Selesaikan repeated_10x5_k20_lr_V2.py terlebih dahulu."
        )
    assignments = pd.read_csv(REPEATED_ASSIGNMENTS_FILE)
    required = {"Repeat", "Fold", "Sample_ID"}
    missing = required.difference(assignments.columns)
    if missing:
        raise KeyError(
            f"Kolom assignment repeated tidak lengkap: {sorted(missing)}"
        )
    assignments["Repeat"] = pd.to_numeric(assignments["Repeat"], errors="raise").astype(int)
    assignments["Fold"] = pd.to_numeric(assignments["Fold"], errors="raise").astype(int)
    assignments["Sample_ID"] = assignments["Sample_ID"].astype(str)
    assignments = assignments.drop_duplicates(["Repeat", "Sample_ID"])

    if set(assignments["Repeat"].unique()) != set(range(1, REPEATS + 1)):
        raise RuntimeError("Repeated assignments tidak berisi repeat 1 sampai 10.")
    if set(assignments["Fold"].unique()) != set(range(1, FOLDS + 1)):
        raise RuntimeError("Repeated assignments tidak berisi fold 1 sampai 5.")

    counts = assignments.groupby("Repeat")["Sample_ID"].nunique()
    if not (counts == len(X_DEV)).all():
        raise RuntimeError(
            "Setiap repeat harus memuat tepat 608 validation assignments."
        )
    unknown = set(assignments["Sample_ID"]).difference(set(X_DEV.index))
    missing_samples = set(X_DEV.index).difference(set(assignments["Sample_ID"]))
    if unknown or missing_samples:
        raise RuntimeError(
            f"Assignment/sample mismatch: unknown={len(unknown)}, "
            f"missing={len(missing_samples)}."
        )

    per_fold = assignments.groupby(["Repeat", "Fold"])["Sample_ID"].nunique()
    if len(per_fold) != REPEATS * FOLDS or (per_fold <= 0).any():
        raise RuntimeError("Repeated assignments tidak membentuk tepat 50 fold non-kosong.")
    return assignments.sort_values(["Repeat", "Fold", "Sample_ID"]).reset_index(drop=True)


FOLD_ASSIGNMENTS = _load_repeated_assignments()


# ------------------------------------------------------------
# 3. Load and lock the high-confidence STRING graph
# ------------------------------------------------------------

def _load_string_edges() -> tuple[pd.DataFrame, Path]:
    override = globals().get("RWR_EDGE_FILE")
    if override:
        path = Path(str(override)).expanduser().resolve()
    else:
        path = _latest_path(["STRING_gene_edges_eligible_genes.csv"])
    if path is None or not path.exists():
        raise FileNotFoundError(
            "STRING_gene_edges_eligible_genes.csv tidak ditemukan dalam main run."
        )

    edges = pd.read_csv(path)
    gene1_col = _find_column(edges, ["Gene1", "gene1", "Source", "Node1"])
    gene2_col = _find_column(edges, ["Gene2", "gene2", "Target", "Node2"])
    score_col = _find_column(
        edges,
        ["combined_score", "Combined_score", "Score", "STRING_score", "weight"],
    )
    if gene1_col is None or gene2_col is None:
        raise KeyError(f"Kolom endpoint STRING tidak ditemukan: {list(edges.columns)}")

    keep_columns = [gene1_col, gene2_col] + ([score_col] if score_col else [])
    edges = edges[keep_columns].copy()
    rename = {gene1_col: "Gene1", gene2_col: "Gene2"}
    if score_col:
        rename[score_col] = "combined_score"
    edges = edges.rename(columns=rename)
    edges["Gene1"] = edges["Gene1"].astype(str)
    edges["Gene2"] = edges["Gene2"].astype(str)
    if "combined_score" not in edges.columns:
        edges["combined_score"] = 1.0
    edges["combined_score"] = pd.to_numeric(
        edges["combined_score"], errors="coerce"
    ).fillna(0.0)

    eligible = set(X_DEV.columns)
    edges = edges[
        edges["Gene1"].isin(eligible)
        & edges["Gene2"].isin(eligible)
        & (edges["Gene1"] != edges["Gene2"])
    ].copy()
    if edges.empty:
        raise RuntimeError("Tidak ada edge STRING yang tersisa setelah alignment.")

    # Canonicalize undirected pairs and keep the strongest duplicate score.
    left = edges[["Gene1", "Gene2"]].min(axis=1)
    right = edges[["Gene1", "Gene2"]].max(axis=1)
    edges["Gene1"] = left
    edges["Gene2"] = right
    edges = (
        edges.groupby(["Gene1", "Gene2"], as_index=False)["combined_score"]
        .max()
        .sort_values(["Gene1", "Gene2"])
        .reset_index(drop=True)
    )
    return edges, path


def _build_transition_matrix(
    edges: pd.DataFrame,
    genes: list[str],
    use_weights: bool,
) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    gene_to_index = {gene: idx for idx, gene in enumerate(genes)}
    row = np.concatenate(
        [
            edges["Gene1"].map(gene_to_index).to_numpy(dtype=int),
            edges["Gene2"].map(gene_to_index).to_numpy(dtype=int),
        ]
    )
    col = np.concatenate(
        [
            edges["Gene2"].map(gene_to_index).to_numpy(dtype=int),
            edges["Gene1"].map(gene_to_index).to_numpy(dtype=int),
        ]
    )

    if use_weights:
        raw_score = edges["combined_score"].to_numpy(dtype=float)
        # STRING combined scores are ordinarily 0--1000. Scaling does not
        # change row-normalized transition probabilities, but improves auditability.
        scale = 1000.0 if np.nanmax(raw_score) > 1.0 else 1.0
        edge_weight = np.clip(raw_score / scale, 0.0, None)
        if not np.any(edge_weight > 0):
            raise RuntimeError("Semua STRING edge weights bernilai nol.")
    else:
        edge_weight = np.ones(len(edges), dtype=float)
    data = np.concatenate([edge_weight, edge_weight])

    adjacency = sparse.csr_matrix(
        (data, (row, col)),
        shape=(len(genes), len(genes)),
        dtype=float,
    )
    adjacency.sum_duplicates()

    weighted_degree = np.asarray(adjacency.sum(axis=1)).ravel()
    unweighted_degree = np.asarray((adjacency > 0).sum(axis=1)).ravel()
    isolated = weighted_degree <= 0
    if isolated.any():
        adjacency = adjacency + sparse.diags(isolated.astype(float), format="csr")
        weighted_degree = np.asarray(adjacency.sum(axis=1)).ravel()

    row_inverse = np.divide(
        1.0,
        weighted_degree,
        out=np.zeros_like(weighted_degree, dtype=float),
        where=weighted_degree > 0,
    )
    transition_row = sparse.diags(row_inverse, format="csr") @ adjacency
    propagation = transition_row.transpose().tocsr()

    row_sums = np.asarray(transition_row.sum(axis=1)).ravel()
    if not np.allclose(row_sums, 1.0, atol=1e-12):
        raise RuntimeError("Transition matrix tidak row-stochastic.")

    audit = pd.DataFrame(
        {
            "Gene": genes,
            "Unweighted_degree": unweighted_degree.astype(int),
            "Weighted_degree_before_isolate_self_loop": np.where(
                isolated, 0.0, weighted_degree
            ),
            "Isolated_before_self_loop": isolated,
        }
    )
    return propagation, audit


STRING_EDGES, STRING_EDGE_PATH = _load_string_edges()
GENES = X_DEV.columns.astype(str).tolist()
PROPAGATION_MATRIX, NETWORK_GENE_AUDIT = _build_transition_matrix(
    STRING_EDGES,
    GENES,
    USE_STRING_WEIGHTS,
)
NETWORK_GENE_AUDIT.to_csv(OUTPUT_DIR / "rwr_deg_network_gene_audit.csv", index=False)

network_summary = pd.DataFrame(
    [
        {
            "Edge_source": str(STRING_EDGE_PATH),
            "Eligible_genes": len(GENES),
            "Unique_undirected_edges": len(STRING_EDGES),
            "Genes_with_at_least_one_edge": int(
                (~NETWORK_GENE_AUDIT["Isolated_before_self_loop"]).sum()
            ),
            "Isolated_genes": int(
                NETWORK_GENE_AUDIT["Isolated_before_self_loop"].sum()
            ),
            "Weighted_by_STRING_combined_score": USE_STRING_WEIGHTS,
            "Isolate_handling": "self-loop only for isolated genes",
            "Transition_normalization": "row-stochastic",
        }
    ]
)
network_summary.to_csv(OUTPUT_DIR / "rwr_deg_network_summary.csv", index=False)


# ------------------------------------------------------------
# 4. Fold-local limma and RWR ranking
# ------------------------------------------------------------

def run_limma_rpy2(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.conversion import localconverter
    except Exception as exc:
        raise ImportError(
            "RWR-DEG membutuhkan rpy2 dan R limma. Jalankan sel instalasi "
            "R/limma terlebih dahulu."
        ) from exc

    ro.r("suppressPackageStartupMessages(library(limma))")
    genes = X.columns.astype(str).tolist()
    expr = X.to_numpy(dtype=float).T
    y_arr = np.asarray(y, dtype=int)

    with localconverter(ro.default_converter + numpy2ri.converter):
        ro.globalenv["expr_matrix_rwr"] = expr
        ro.globalenv["group_vector_rwr"] = y_arr
    ro.globalenv["gene_names_rwr"] = ro.StrVector(genes)

    ro.r(
        """
        rownames(expr_matrix_rwr) <- gene_names_rwr
        group_factor_rwr <- factor(
            group_vector_rwr,
            levels=c(0,1),
            labels=c("Normal","Cancer")
        )
        design_rwr <- model.matrix(~ group_factor_rwr)
        fit_rwr <- lmFit(expr_matrix_rwr, design_rwr)
        fit_rwr <- eBayes(fit_rwr)
        limma_result_rwr <- topTable(
            fit_rwr,
            coef=2,
            number=Inf,
            adjust.method="BH",
            sort.by="none"
        )
        limma_result_rwr$Gene <- rownames(limma_result_rwr)
        """
    )

    columns = list(ro.r("colnames(limma_result_rwr)"))
    result = pd.DataFrame(
        {column: list(ro.r(f"limma_result_rwr${column}")) for column in columns}
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


def _seed_vector_from_limma(limma_table: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    required = {"Gene", "Stat_score", "Rank_stat"}
    missing = required.difference(limma_table.columns)
    if missing:
        raise KeyError(f"Limma table kehilangan kolom: {sorted(missing)}")

    aligned = pd.DataFrame({"Gene": GENES}).merge(
        limma_table,
        on="Gene",
        how="left",
        validate="one_to_one",
    )
    aligned["Stat_score"] = pd.to_numeric(
        aligned["Stat_score"], errors="coerce"
    ).fillna(0.0)
    aligned["Rank_stat"] = pd.to_numeric(
        aligned["Rank_stat"], errors="coerce"
    ).fillna(float(len(GENES)))
    score = np.clip(aligned["Stat_score"].to_numpy(dtype=float), 0.0, None)
    total = float(score.sum())
    if total <= 0 or not np.isfinite(total):
        # Deterministic fallback, expected never to be needed in this dataset.
        rank = aligned["Rank_stat"].to_numpy(dtype=float)
        score = 1.0 / np.clip(rank, 1.0, None)
        total = float(score.sum())
        aligned["Seed_fallback"] = "inverse_statistical_rank"
    else:
        aligned["Seed_fallback"] = "none"
    seed = score / total
    if not np.isclose(seed.sum(), 1.0, atol=1e-12):
        raise RuntimeError("Restart seed vector tidak berjumlah satu.")
    aligned["Restart_seed_probability"] = seed
    return seed, aligned


def random_walk_with_restart(
    propagation: sparse.csr_matrix,
    restart_seed: np.ndarray,
    restart_probability: float,
    tolerance: float,
    max_iter: int,
) -> tuple[np.ndarray, int, float]:
    p0 = np.asarray(restart_seed, dtype=float)
    p = p0.copy()
    final_delta = math.inf
    for iteration in range(1, max_iter + 1):
        updated = (
            (1.0 - restart_probability) * propagation.dot(p)
            + restart_probability * p0
        )
        final_delta = float(np.abs(updated - p).sum())
        p = updated
        if final_delta <= tolerance:
            break
    else:
        warnings.warn(
            f"RWR belum konvergen setelah {max_iter} iterasi; "
            f"L1 delta={final_delta:.3e}.",
            RuntimeWarning,
        )

    p = np.clip(p, 0.0, None)
    total = float(p.sum())
    if total <= 0 or not np.isfinite(total):
        raise RuntimeError("RWR menghasilkan skor tidak valid.")
    p /= total
    return p, iteration, final_delta


def rwr_rank_table(
    limma_table: pd.DataFrame,
    restart_probability: float,
) -> tuple[pd.DataFrame, dict]:
    seed, aligned = _seed_vector_from_limma(limma_table)
    score, iterations, final_delta = random_walk_with_restart(
        PROPAGATION_MATRIX,
        seed,
        restart_probability,
        RWR_TOLERANCE,
        RWR_MAX_ITER,
    )
    aligned["RWR_score"] = score
    aligned["Restart_probability"] = restart_probability
    aligned = aligned.sort_values(
        ["RWR_score", "Rank_stat", "Gene"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    aligned["Rank_RWR"] = np.arange(1, len(aligned) + 1)
    diagnostics = {
        "Restart_probability": float(restart_probability),
        "RWR_iterations": int(iterations),
        "RWR_final_L1_delta": float(final_delta),
        "Seed_nonzero_genes": int((seed > 0).sum()),
        "Seed_max_probability": float(seed.max()),
        "RWR_max_probability": float(score.max()),
    }
    return aligned, diagnostics


def select_top_k(table: pd.DataFrame, k: int) -> list[str]:
    genes = table.sort_values(["Rank_RWR", "Gene"]).head(k)["Gene"].astype(str).tolist()
    if len(genes) != k or len(set(genes)) != k:
        raise RuntimeError(f"RWR panel tidak menghasilkan {k} gen unik: {genes}")
    return genes


# ------------------------------------------------------------
# 5. Prediction and metric helpers
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


def _safe_wilcoxon(a, b) -> tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if len(a) < 2:
        return np.nan, np.nan
    difference = a - b
    if np.allclose(difference, 0.0):
        return 0.0, 1.0
    try:
        result = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
        return float(result.statistic), float(result.pvalue)
    except ValueError:
        return np.nan, np.nan


# ------------------------------------------------------------
# 6. Checkpoint helpers
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
        "diagnostics": prefix.with_name(prefix.name + "_rwr_diagnostics.csv"),
        "runtime": prefix.with_name(prefix.name + "_runtime.json"),
        "done": prefix.with_name(prefix.name + "_DONE.json"),
        "limma": CACHE_DIR / f"repeat_{repeat:02d}_fold_{fold:02d}_limma.csv.gz",
    }


def _checkpoint_complete(repeat: int, fold: int) -> bool:
    files = _checkpoint_files(repeat, fold)
    required = [
        "metrics",
        "predictions",
        "panels",
        "assignments",
        "diagnostics",
        "runtime",
        "done",
    ]
    return all(files[key].exists() for key in required)


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


def _save_csv_gz_atomic(frame: pd.DataFrame, path: Path) -> None:
    temp = Path(str(path) + ".tmp")
    frame.to_csv(temp, index=False, compression="gzip")
    temp.replace(path)


# ------------------------------------------------------------
# 7. Execute or resume the 50 folds
# ------------------------------------------------------------
TOTAL_FOLDS = REPEATS * FOLDS
completed_before = sum(
    _checkpoint_complete(repeat, fold)
    for repeat in range(1, REPEATS + 1)
    for fold in range(1, FOLDS + 1)
)

if FORCE_RERUN:
    print("RWR_FORCE_RERUN=True: removing all RWR-DEG checkpoints.", flush=True)
    for repeat in range(1, REPEATS + 1):
        for fold in range(1, FOLDS + 1):
            _remove_checkpoint(repeat, fold)
    completed_before = 0

print("=" * 82)
print("RWR-DEG NETWORK BASELINE — REPEATED 10 x 5-FOLD")
print("=" * 82)
print("Project                 :", PROJECT_DIR)
print("Output                  :", OUTPUT_DIR)
print("Development data        :", X_DEV.shape)
print("Class counts            :", dict(pd.Series(Y_DEV).value_counts().sort_index()))
print("Exact assignment source :", REPEATED_ASSIGNMENTS_FILE)
print("STRING edge source      :", STRING_EDGE_PATH)
print("STRING edges            :", len(STRING_EDGES))
print("Main restart            :", MAIN_RESTART)
print("Restart sensitivity     :", RESTART_VALUES)
print("Weighted STRING graph   :", USE_STRING_WEIGHTS)
print("Existing checkpoints    :", f"{completed_before}/{TOTAL_FOLDS}")

new_fold_count = 0
session_fold_times: list[float] = []
stop_requested = False

for repeat in range(1, REPEATS + 1):
    for fold in range(1, FOLDS + 1):
        global_position = (repeat - 1) * FOLDS + fold
        files = _checkpoint_files(repeat, fold)

        if _checkpoint_complete(repeat, fold):
            print(
                f"[{global_position:02d}/{TOTAL_FOLDS}] repeat {repeat}/{REPEATS}, "
                f"fold {fold}/{FOLDS}: checkpoint found — skipped",
                flush=True,
            )
            continue

        if MAX_NEW_FOLDS is not None and new_fold_count >= MAX_NEW_FOLDS:
            stop_requested = True
            break

        fold_start = perf_counter()
        fold_seed = RANDOM_STATE + (repeat - 1) * FOLDS + fold

        validation_ids = FOLD_ASSIGNMENTS.loc[
            (FOLD_ASSIGNMENTS["Repeat"] == repeat)
            & (FOLD_ASSIGNMENTS["Fold"] == fold),
            "Sample_ID",
        ].astype(str).tolist()
        if not validation_ids:
            raise RuntimeError(f"Validation IDs kosong untuk repeat={repeat}, fold={fold}.")
        validation_set = set(validation_ids)
        fit_ids = [sample_id for sample_id in X_DEV.index if sample_id not in validation_set]
        if len(fit_ids) + len(validation_ids) != len(X_DEV):
            raise RuntimeError("Fold assignment overlap atau missing sample terdeteksi.")

        X_fit = X_DEV.loc[fit_ids]
        y_fit = Y_DEV[X_DEV.index.get_indexer(fit_ids)]
        X_val = X_DEV.loc[validation_ids]
        y_val = Y_DEV[X_DEV.index.get_indexer(validation_ids)]

        print("\n" + "=" * 82, flush=True)
        print(
            f"[{global_position:02d}/{TOTAL_FOLDS}] REPEAT {repeat}/{REPEATS} — "
            f"FOLD {fold}/{FOLDS}",
            flush=True,
        )
        print(
            f"training={X_fit.shape}, validation={X_val.shape}, "
            f"train classes={dict(pd.Series(y_fit).value_counts().sort_index())}",
            flush=True,
        )

        limma_cache_used = files["limma"].exists()
        if limma_cache_used:
            print("  [1/4] Loading cached fold-local limma scores ...", flush=True)
            limma_table = pd.read_csv(files["limma"])
        else:
            print("  [1/4] R limma on fold-training samples ...", flush=True)
            limma_table = run_limma_rpy2(X_fit, y_fit)
            _save_csv_gz_atomic(limma_table, files["limma"])

        print(
            "  [2/4] Weighted RWR-DEG ranking for restart probabilities "
            + ", ".join(f"{r:.2f}" for r in RESTART_VALUES)
            + " ...",
            flush=True,
        )
        ranking_tables: dict[float, pd.DataFrame] = {}
        diagnostics_rows: list[dict] = []
        panels: dict[float, list[str]] = {}
        for restart in RESTART_VALUES:
            ranking, diagnostics = rwr_rank_table(limma_table, restart)
            ranking_tables[restart] = ranking
            panels[restart] = select_top_k(ranking, FINAL_K)
            diagnostics_rows.append(
                {
                    "Repeat": repeat,
                    "Fold": fold,
                    **diagnostics,
                }
            )

        print("  [3/4] Logistic-regression validation for all restart values ...", flush=True)
        metrics_rows: list[dict] = []
        prediction_rows: list[dict] = []
        panel_rows: list[dict] = []

        for restart in RESTART_VALUES:
            genes = panels[restart]
            model = create_lr(fold_seed)
            model.fit(X_fit[genes], y_fit)
            probability = model.predict_proba(X_val[genes])[:, 1]
            metric = classification_metrics(y_val, probability, DEFAULT_THRESHOLD)
            variant = "main" if math.isclose(restart, MAIN_RESTART) else "sensitivity"
            method_label = "RWR-DEG" if variant == "main" else f"RWR-DEG r={restart:.2f}"

            metrics_rows.append(
                {
                    "Repeat": repeat,
                    "Fold": fold,
                    "Method": method_label,
                    "Variant": variant,
                    "Restart_probability": restart,
                    "Classifier": "LR",
                    "k": FINAL_K,
                    "Training_samples": len(fit_ids),
                    "Validation_samples": len(validation_ids),
                    **metric,
                }
            )

            for sample_id, true_label, prob in zip(validation_ids, y_val, probability):
                prediction_rows.append(
                    {
                        "Repeat": repeat,
                        "Fold": fold,
                        "Method": method_label,
                        "Variant": variant,
                        "Restart_probability": restart,
                        "Classifier": "LR",
                        "k": FINAL_K,
                        "Sample_ID": sample_id,
                        "True_Label": int(true_label),
                        "Probability": float(prob),
                    }
                )

            detail = ranking_tables[restart].set_index("Gene")
            for selection_rank, gene in enumerate(genes, 1):
                row = {
                    "Repeat": repeat,
                    "Fold": fold,
                    "Method": method_label,
                    "Variant": variant,
                    "Restart_probability": restart,
                    "k": FINAL_K,
                    "Selection_rank": selection_rank,
                    "Gene": gene,
                }
                values = detail.loc[gene]
                for column in [
                    "logFC",
                    "FDR",
                    "Stat_score",
                    "Rank_stat",
                    "Restart_seed_probability",
                    "RWR_score",
                    "Rank_RWR",
                ]:
                    if column in values.index:
                        row[column] = values[column]
                panel_rows.append(row)

        print("  [4/4] Saving checkpoint and audit records ...", flush=True)
        assignment_rows: list[dict] = []
        for sample_id in validation_ids:
            local_idx = X_DEV.index.get_loc(sample_id)
            meta_row = META_DEV.loc[sample_id]
            assignment_rows.append(
                {
                    "Repeat": repeat,
                    "Fold": fold,
                    "Sample_ID": sample_id,
                    "GEO_ID": str(meta_row.get("GEO_ID", "unknown")),
                    "Label": str(
                        meta_row.get(
                            "Label",
                            "Cancer" if Y_DEV[local_idx] == 1 else "Normal",
                        )
                    ),
                    "Label_binary": int(Y_DEV[local_idx]),
                    "Subset": "Validation",
                    "Assignment_source": str(REPEATED_ASSIGNMENTS_FILE),
                }
            )

        elapsed = perf_counter() - fold_start
        runtime_data = {
            "Repeat": repeat,
            "Fold": fold,
            "Elapsed_seconds": elapsed,
            "Elapsed_minutes": elapsed / 60.0,
            "Random_state": fold_seed,
            "Training_samples": len(fit_ids),
            "Validation_samples": len(validation_ids),
            "Limma_cache_used": bool(limma_cache_used),
            "Completed": True,
        }

        _save_csv_atomic(pd.DataFrame(metrics_rows), files["metrics"])
        _save_csv_atomic(pd.DataFrame(prediction_rows), files["predictions"])
        _save_csv_atomic(pd.DataFrame(panel_rows), files["panels"])
        _save_csv_atomic(pd.DataFrame(assignment_rows), files["assignments"])
        _save_csv_atomic(pd.DataFrame(diagnostics_rows), files["diagnostics"])
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
        eta_minutes = remaining * float(np.mean(session_fold_times)) / 60.0
        print(
            f"  completed in {elapsed / 60.0:.2f} minutes | "
            f"checkpoints={complete_now}/{TOTAL_FOLDS} | "
            f"estimated remaining={eta_minutes:.1f} minutes",
            flush=True,
        )
        print(
            "  Main RWR-DEG panel: " + ", ".join(panels[MAIN_RESTART][:10]) + ", ...",
            flush=True,
        )

        del X_fit, X_val, y_fit, y_val, limma_table, ranking_tables, panels
        gc.collect()

    if stop_requested:
        break


# ------------------------------------------------------------
# 8. Aggregate every complete checkpoint (partial-repeat safe)
# ------------------------------------------------------------
metrics_parts: list[pd.DataFrame] = []
prediction_parts: list[pd.DataFrame] = []
panel_parts: list[pd.DataFrame] = []
assignment_parts: list[pd.DataFrame] = []
diagnostic_parts: list[pd.DataFrame] = []
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
        diagnostic_parts.append(pd.read_csv(files["diagnostics"]))
        runtime_rows.append(json.loads(files["runtime"].read_text(encoding="utf-8")))

if not metrics_parts:
    raise RuntimeError("Belum ada fold RWR-DEG yang selesai.")

fold_metrics = pd.concat(metrics_parts, ignore_index=True).sort_values(
    ["Repeat", "Fold", "Restart_probability"]
)
predictions = pd.concat(prediction_parts, ignore_index=True).sort_values(
    ["Repeat", "Fold", "Restart_probability", "Sample_ID"]
)
selected_panels = pd.concat(panel_parts, ignore_index=True).sort_values(
    ["Repeat", "Fold", "Restart_probability", "Selection_rank"]
)
fold_assignments = pd.concat(assignment_parts, ignore_index=True).sort_values(
    ["Repeat", "Fold", "Sample_ID"]
)
rwr_diagnostics = pd.concat(diagnostic_parts, ignore_index=True).sort_values(
    ["Repeat", "Fold", "Restart_probability"]
)
runtime_table = pd.DataFrame(runtime_rows).sort_values(["Repeat", "Fold"])

fold_metrics.to_csv(OUTPUT_DIR / "rwr_deg_fold_metrics.csv", index=False)
predictions.to_csv(OUTPUT_DIR / "rwr_deg_predictions.csv", index=False)
selected_panels.to_csv(OUTPUT_DIR / "rwr_deg_selected_panels.csv", index=False)
fold_assignments.to_csv(OUTPUT_DIR / "rwr_deg_fold_assignments.csv", index=False)
rwr_diagnostics.to_csv(OUTPUT_DIR / "rwr_deg_convergence_diagnostics.csv", index=False)
runtime_table.to_csv(OUTPUT_DIR / "rwr_deg_runtime.csv", index=False)

fold_summary = (
    fold_metrics.groupby(["Method", "Restart_probability"], as_index=False)
    .agg(
        Completed_folds=("Fold", "count"),
        ROC_AUC_mean=("ROC_AUC", "mean"),
        ROC_AUC_std=("ROC_AUC", "std"),
        F1_mean=("F1", "mean"),
        MCC_mean=("MCC", "mean"),
        Brier_score_mean=("Brier_score", "mean"),
    )
)
fold_summary.to_csv(OUTPUT_DIR / "rwr_deg_fold_performance_summary.csv", index=False)

# Complete repeat = exactly 608 OOF probabilities for every restart value.
complete_repeats: list[int] = []
for repeat in range(1, REPEATS + 1):
    subset = predictions.loc[predictions["Repeat"] == repeat]
    valid = True
    for restart in RESTART_VALUES:
        group = subset.loc[np.isclose(subset["Restart_probability"], restart)]
        if len(group) != len(X_DEV) or group["Sample_ID"].nunique() != len(X_DEV):
            valid = False
            break
    if valid:
        complete_repeats.append(repeat)

# OOF metrics by complete repeat.
oof_rows: list[dict] = []
for repeat in complete_repeats:
    for restart in RESTART_VALUES:
        group = predictions.loc[
            (predictions["Repeat"] == repeat)
            & np.isclose(predictions["Restart_probability"], restart)
        ].sort_values("Sample_ID")
        method = "RWR-DEG" if math.isclose(restart, MAIN_RESTART) else f"RWR-DEG r={restart:.2f}"
        metric = classification_metrics(
            group["True_Label"].to_numpy(int),
            group["Probability"].to_numpy(float),
            DEFAULT_THRESHOLD,
        )
        oof_rows.append(
            {
                "Repeat": repeat,
                "Method": method,
                "Restart_probability": restart,
                "k": FINAL_K,
                **metric,
            }
        )

oof_metrics = pd.DataFrame(oof_rows)
if not oof_metrics.empty:
    oof_metrics = oof_metrics.sort_values(["Repeat", "Restart_probability"])
oof_metrics.to_csv(OUTPUT_DIR / "rwr_deg_oof_metrics_by_repeat.csv", index=False)

if not oof_metrics.empty:
    oof_summary = (
        oof_metrics.groupby(["Method", "Restart_probability"], as_index=False)
        .agg(
            ROC_AUC_count=("ROC_AUC", "count"),
            ROC_AUC_mean=("ROC_AUC", "mean"),
            ROC_AUC_std=("ROC_AUC", "std"),
            ROC_AUC_median=("ROC_AUC", "median"),
            ROC_AUC_min=("ROC_AUC", "min"),
            ROC_AUC_max=("ROC_AUC", "max"),
            F1_mean=("F1", "mean"),
            MCC_mean=("MCC", "mean"),
            Brier_score_mean=("Brier_score", "mean"),
        )
    )
else:
    oof_summary = pd.DataFrame()
oof_summary.to_csv(OUTPUT_DIR / "rwr_deg_oof_performance_summary.csv", index=False)

# Within-repeat pairwise Jaccard, only for complete repeats.
pairwise_rows: list[dict] = []
stability_rows: list[dict] = []
for repeat in complete_repeats:
    for restart in RESTART_VALUES:
        panel_group = selected_panels.loc[
            (selected_panels["Repeat"] == repeat)
            & np.isclose(selected_panels["Restart_probability"], restart)
        ]
        panels = {
            int(fold): set(frame["Gene"].astype(str))
            for fold, frame in panel_group.groupby("Fold")
        }
        if set(panels) != set(range(1, FOLDS + 1)):
            continue
        values: list[float] = []
        for fold_a, fold_b in combinations(range(1, FOLDS + 1), 2):
            a = panels[fold_a]
            b = panels[fold_b]
            jaccard = len(a & b) / len(a | b)
            values.append(jaccard)
            pairwise_rows.append(
                {
                    "Repeat": repeat,
                    "Fold_A": fold_a,
                    "Fold_B": fold_b,
                    "Restart_probability": restart,
                    "Jaccard": jaccard,
                    "Intersection_size": len(a & b),
                    "Union_size": len(a | b),
                }
            )
        method = "RWR-DEG" if math.isclose(restart, MAIN_RESTART) else f"RWR-DEG r={restart:.2f}"
        stability_rows.append(
            {
                "Repeat": repeat,
                "Method": method,
                "Restart_probability": restart,
                "Mean_Jaccard": float(np.mean(values)),
                "SD_Jaccard": float(np.std(values, ddof=1)),
                "Median_Jaccard": float(np.median(values)),
                "Minimum_Jaccard": float(np.min(values)),
                "Maximum_Jaccard": float(np.max(values)),
                "Pair_count": len(values),
            }
        )

pairwise_jaccard = pd.DataFrame(pairwise_rows)
repeat_stability = pd.DataFrame(stability_rows)
pairwise_jaccard.to_csv(
    OUTPUT_DIR / "rwr_deg_within_repeat_pairwise_jaccard.csv", index=False
)
repeat_stability.to_csv(OUTPUT_DIR / "rwr_deg_stability_by_repeat.csv", index=False)

if not repeat_stability.empty:
    stability_summary = (
        repeat_stability.groupby(["Method", "Restart_probability"], as_index=False)
        .agg(
            Repeats=("Repeat", "count"),
            Mean_of_repeat_mean_Jaccard=("Mean_Jaccard", "mean"),
            SD_of_repeat_mean_Jaccard=("Mean_Jaccard", "std"),
            Median_of_repeat_mean_Jaccard=("Mean_Jaccard", "median"),
            Minimum_repeat_mean_Jaccard=("Mean_Jaccard", "min"),
            Maximum_repeat_mean_Jaccard=("Mean_Jaccard", "max"),
        )
    )
else:
    stability_summary = pd.DataFrame()
stability_summary.to_csv(OUTPUT_DIR / "rwr_deg_stability_summary.csv", index=False)

# Gene recurrence for every completed fold panel.
recurrence_rows: list[dict] = []
for restart in RESTART_VALUES:
    subset = selected_panels.loc[
        np.isclose(selected_panels["Restart_probability"], restart)
    ]
    total_panels = subset[["Repeat", "Fold"]].drop_duplicates().shape[0]
    if total_panels == 0:
        continue
    frequency = subset.groupby("Gene").size()
    mean_rank = subset.groupby("Gene")["Selection_rank"].mean()
    method = "RWR-DEG" if math.isclose(restart, MAIN_RESTART) else f"RWR-DEG r={restart:.2f}"
    for gene, count in frequency.items():
        recurrence_rows.append(
            {
                "Method": method,
                "Restart_probability": restart,
                "Gene": gene,
                "Selection_frequency": int(count),
                "Total_fold_panels": int(total_panels),
                "Selection_proportion": float(count / total_panels),
                "Mean_selection_rank": float(mean_rank.loc[gene]),
            }
        )

gene_recurrence = pd.DataFrame(recurrence_rows)
if not gene_recurrence.empty:
    gene_recurrence = gene_recurrence.sort_values(
        ["Restart_probability", "Selection_frequency", "Mean_selection_rank", "Gene"],
        ascending=[True, False, True, True],
    )
gene_recurrence.to_csv(OUTPUT_DIR / "rwr_deg_gene_selection_frequency.csv", index=False)

# Combined restart-sensitivity summary.
if not oof_summary.empty or not stability_summary.empty:
    restart_sensitivity = pd.merge(
        oof_summary,
        stability_summary,
        on=["Method", "Restart_probability"],
        how="outer",
    )
else:
    restart_sensitivity = pd.DataFrame()
restart_sensitivity.to_csv(
    OUTPUT_DIR / "rwr_deg_restart_sensitivity_summary.csv", index=False
)


# ------------------------------------------------------------
# 9. Compare main RWR-DEG against completed existing methods
# ------------------------------------------------------------
combined_summary = pd.DataFrame()
paired_tests_rows: list[dict] = []
existing_oof_file = REPEATED_DIR / "repeated_oof_metrics_by_repeat.csv"
existing_stability_file = REPEATED_DIR / "repeated_stability_by_repeat.csv"
existing_oof_summary_file = REPEATED_DIR / "repeated_oof_performance_summary.csv"
existing_stability_summary_file = REPEATED_DIR / "repeated_stability_summary.csv"

if (
    existing_oof_file.exists()
    and existing_stability_file.exists()
    and not oof_metrics.empty
    and not repeat_stability.empty
):
    existing_oof = pd.read_csv(existing_oof_file)
    existing_stability = pd.read_csv(existing_stability_file)
    rwr_main_oof = oof_metrics.loc[
        np.isclose(oof_metrics["Restart_probability"], MAIN_RESTART)
    ][["Repeat", "ROC_AUC"]].rename(columns={"ROC_AUC": "RWR_DEG"})
    rwr_main_stability = repeat_stability.loc[
        np.isclose(repeat_stability["Restart_probability"], MAIN_RESTART)
    ][["Repeat", "Mean_Jaccard"]].rename(columns={"Mean_Jaccard": "RWR_DEG"})

    nibfs_oof = existing_oof.loc[existing_oof["Method"] == "NIBFS", ["Repeat", "ROC_AUC"]].rename(
        columns={"ROC_AUC": "NIBFS"}
    )
    nibfs_stability = existing_stability.loc[
        existing_stability["Method"] == "NIBFS", ["Repeat", "Mean_Jaccard"]
    ].rename(columns={"Mean_Jaccard": "NIBFS"})

    for analysis_name, merged, value_label in [
        (
            "Repeated OOF ROC-AUC",
            nibfs_oof.merge(rwr_main_oof, on="Repeat", how="inner"),
            "ROC_AUC",
        ),
        (
            "Repeated panel stability",
            nibfs_stability.merge(rwr_main_stability, on="Repeat", how="inner"),
            "Mean_Jaccard",
        ),
    ]:
        statistic, p_value = _safe_wilcoxon(merged["NIBFS"], merged["RWR_DEG"])
        difference = merged["NIBFS"] - merged["RWR_DEG"]
        paired_tests_rows.append(
            {
                "Analysis": analysis_name,
                "Metric": value_label,
                "Test": "Paired Wilcoxon",
                "Alternative": "two-sided",
                "Proposed": "NIBFS",
                "Comparator": "RWR-DEG",
                "N_repeats": len(merged),
                "Statistic": statistic,
                "P_value": p_value,
                "Mean_NIBFS_minus_RWR": float(difference.mean()) if len(difference) else np.nan,
                "Median_NIBFS_minus_RWR": float(difference.median()) if len(difference) else np.nan,
            }
        )

if paired_tests_rows:
    paired_tests = pd.DataFrame(paired_tests_rows)
else:
    paired_tests = pd.DataFrame(
        columns=[
            "Analysis",
            "Metric",
            "Test",
            "Alternative",
            "Proposed",
            "Comparator",
            "N_repeats",
            "Statistic",
            "P_value",
            "Mean_NIBFS_minus_RWR",
            "Median_NIBFS_minus_RWR",
        ]
    )
paired_tests.to_csv(OUTPUT_DIR / "rwr_deg_vs_nibfs_paired_tests.csv", index=False)

# Publication-friendly method summary, when the completed repeated summaries exist.
if existing_oof_summary_file.exists() and existing_stability_summary_file.exists():
    existing_perf_summary = pd.read_csv(existing_oof_summary_file)
    existing_stab_summary = pd.read_csv(existing_stability_summary_file)
    existing_combined = existing_perf_summary.merge(
        existing_stab_summary,
        on="Method",
        how="outer",
    )
    rwr_perf = oof_summary.loc[
        np.isclose(oof_summary.get("Restart_probability", np.nan), MAIN_RESTART)
    ].copy() if not oof_summary.empty else pd.DataFrame()
    rwr_stab = stability_summary.loc[
        np.isclose(stability_summary.get("Restart_probability", np.nan), MAIN_RESTART)
    ].copy() if not stability_summary.empty else pd.DataFrame()
    if not rwr_perf.empty and not rwr_stab.empty:
        rwr_combined = rwr_perf.merge(
            rwr_stab,
            on=["Method", "Restart_probability"],
            how="outer",
        )
        combined_summary = pd.concat(
            [existing_combined, rwr_combined],
            ignore_index=True,
            sort=False,
        )
    else:
        combined_summary = existing_combined
combined_summary.to_csv(
    OUTPUT_DIR / "rwr_deg_combined_existing_method_summary.csv", index=False
)


# ------------------------------------------------------------
# 10. Audit, manifest, status and figures
# ------------------------------------------------------------
complete_fold_count = runtime_table[["Repeat", "Fold"]].drop_duplicates().shape[0]
analysis_complete = complete_fold_count == TOTAL_FOLDS
assignment_counts = (
    fold_assignments.groupby(["Repeat", "Sample_ID"]).size().reset_index(name="Count")
)
assignment_ok_for_complete_repeats = True
for repeat in complete_repeats:
    counts = assignment_counts.loc[assignment_counts["Repeat"] == repeat, "Count"]
    if len(counts) != len(X_DEV) or not (counts == 1).all():
        assignment_ok_for_complete_repeats = False
        break

analysis_audit = pd.DataFrame(
    [
        {
            "Analysis": "RWR-DEG network-aware baseline",
            "Development_samples": len(X_DEV),
            "Eligible_genes": X_DEV.shape[1],
            "Locked_internal_test_used": False,
            "Repeated_repeats": REPEATS,
            "Folds_per_repeat": FOLDS,
            "Expected_validation_folds": TOTAL_FOLDS,
            "Completed_validation_folds": complete_fold_count,
            "Complete_repeats": len(complete_repeats),
            "Complete": analysis_complete,
            "Each_sample_validated_once_per_complete_repeat": assignment_ok_for_complete_repeats,
            "Exact_repeated_assignments_reused": True,
            "Assignment_source": str(REPEATED_ASSIGNMENTS_FILE),
            "k": FINAL_K,
            "Main_restart_probability": MAIN_RESTART,
            "Sensitivity_restart_probabilities": "|".join(map(str, RESTART_VALUES)),
            "Seed_definition": "fold-local limma Stat_score normalized to sum 1",
            "Network": "fixed high-confidence STRING eligible-gene graph",
            "STRING_weighting": "combined_score" if USE_STRING_WEIGHTS else "unweighted",
            "Isolate_handling": "self-loop for isolated genes only",
            "Classifier": "LR",
            "Decision_threshold": DEFAULT_THRESHOLD,
            "Limma_scope": "refitted on each fold-training partition only",
            "Restart_values_tuned_on_validation_labels": False,
            "Frozen_primary_panel_changed": False,
            "Main_heldout_results_changed": False,
            "External_validation_results_changed": False,
            "Upstream_input_note": (
                "Uses the main study harmonized development matrix; supervised "
                "limma ranking and RWR panel construction are fold-local."
            ),
        }
    ]
)
analysis_audit.to_csv(OUTPUT_DIR / "rwr_deg_analysis_audit.csv", index=False)

manifest = {
    "analysis": "RWR-DEG network-aware repeated baseline",
    "project_directory": str(PROJECT_DIR),
    "output_directory": str(OUTPUT_DIR),
    "development_shape": list(X_DEV.shape),
    "class_counts": {
        str(key): int(value)
        for key, value in pd.Series(Y_DEV).value_counts().sort_index().items()
    },
    "assignment_source": str(REPEATED_ASSIGNMENTS_FILE),
    "string_edge_source": str(STRING_EDGE_PATH),
    "unique_string_edges": int(len(STRING_EDGES)),
    "use_string_combined_score_weights": USE_STRING_WEIGHTS,
    "repeats": REPEATS,
    "folds": FOLDS,
    "expected_folds": TOTAL_FOLDS,
    "completed_folds": int(complete_fold_count),
    "complete_repeats": complete_repeats,
    "complete": bool(analysis_complete),
    "k": FINAL_K,
    "main_restart_probability": MAIN_RESTART,
    "restart_probabilities": list(RESTART_VALUES),
    "rwr_tolerance": RWR_TOLERANCE,
    "rwr_max_iter": RWR_MAX_ITER,
    "classifier": "LR",
    "decision_threshold": DEFAULT_THRESHOLD,
    "checkpoint_resume_enabled": True,
    "frozen_primary_panel_changed": False,
    "output_files": sorted(path.name for path in OUTPUT_DIR.glob("*.csv")),
}
_save_json_atomic(manifest, OUTPUT_DIR / "rwr_deg_run_manifest.json")

# Figures are regenerated from aggregate output only. No values are entered manually.
figure_png = FIGURE_DIR / "Figure_RWR_DEG_network_baseline.png"
figure_pdf = FIGURE_DIR / "Figure_RWR_DEG_network_baseline.pdf"
try:
    import matplotlib.pyplot as plt

    if analysis_complete and not combined_summary.empty:
        existing_oof = pd.read_csv(existing_oof_file) if existing_oof_file.exists() else pd.DataFrame()
        existing_stability = (
            pd.read_csv(existing_stability_file)
            if existing_stability_file.exists()
            else pd.DataFrame()
        )
        rwr_main_oof = oof_metrics.loc[
            np.isclose(oof_metrics["Restart_probability"], MAIN_RESTART)
        ][["Repeat", "ROC_AUC"]].copy()
        rwr_main_oof["Method"] = "RWR-DEG"
        rwr_main_stability = repeat_stability.loc[
            np.isclose(repeat_stability["Restart_probability"], MAIN_RESTART)
        ][["Repeat", "Mean_Jaccard"]].copy()
        rwr_main_stability["Method"] = "RWR-DEG"

        method_order = ["DEG-only", "mRMR", "LASSO", "NIBFS", "RWR-DEG"]
        auc_frame = pd.concat(
            [existing_oof[["Repeat", "Method", "ROC_AUC"]], rwr_main_oof],
            ignore_index=True,
        )
        stability_frame = pd.concat(
            [
                existing_stability[["Repeat", "Method", "Mean_Jaccard"]],
                rwr_main_stability,
            ],
            ignore_index=True,
        )

        fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9))
        auc_data = [
            auc_frame.loc[auc_frame["Method"] == method, "ROC_AUC"].to_numpy(float)
            for method in method_order
        ]
        axes[0].boxplot(auc_data, tick_labels=method_order, showmeans=True)
        axes[0].set_ylabel("OOF ROC-AUC across repeats")
        axes[0].set_title("Predictive robustness")
        axes[0].tick_params(axis="x", rotation=25)

        stab_data = [
            stability_frame.loc[
                stability_frame["Method"] == method, "Mean_Jaccard"
            ].to_numpy(float)
            for method in method_order
        ]
        axes[1].boxplot(stab_data, tick_labels=method_order, showmeans=True)
        axes[1].set_ylabel("Mean pairwise Jaccard within repeat")
        axes[1].set_title("Panel stability")
        axes[1].tick_params(axis="x", rotation=25)

        fig.suptitle(
            "RWR-DEG network baseline versus existing feature-selection methods "
            "(10 x 5-fold, LR, k=20)"
        )
        fig.tight_layout()
        fig.savefig(figure_png, dpi=600, bbox_inches="tight")
        fig.savefig(figure_pdf, bbox_inches="tight")
        plt.close(fig)

    if not restart_sensitivity.empty:
        sensitivity_png = FIGURE_DIR / "Figure_RWR_DEG_restart_sensitivity.png"
        sensitivity_pdf = FIGURE_DIR / "Figure_RWR_DEG_restart_sensitivity.pdf"
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
        ordered = restart_sensitivity.sort_values("Restart_probability")
        axes[0].plot(
            ordered["Restart_probability"],
            ordered["ROC_AUC_mean"],
            marker="o",
        )
        axes[0].set_xlabel("Restart probability")
        axes[0].set_ylabel("Mean repeated OOF ROC-AUC")
        axes[0].set_title("Predictive sensitivity")
        axes[1].plot(
            ordered["Restart_probability"],
            ordered["Mean_of_repeat_mean_Jaccard"],
            marker="o",
        )
        axes[1].set_xlabel("Restart probability")
        axes[1].set_ylabel("Mean repeated Jaccard")
        axes[1].set_title("Stability sensitivity")
        fig.tight_layout()
        fig.savefig(sensitivity_png, dpi=600, bbox_inches="tight")
        fig.savefig(sensitivity_pdf, bbox_inches="tight")
        plt.close(fig)
except Exception as figure_error:
    warnings.warn(f"RWR-DEG figures were not generated: {figure_error}")

status_path = OUTPUT_DIR / (
    "RWR_DEG_ANALYSIS_COMPLETE.txt"
    if analysis_complete
    else "RWR_DEG_ANALYSIS_INCOMPLETE.txt"
)
status_path.write_text(
    f"Completed folds: {complete_fold_count}/{TOTAL_FOLDS}\n"
    f"Complete repeats: {len(complete_repeats)}/{REPEATS}\n"
    f"New folds in this session: {new_fold_count}\n"
    f"Output directory: {OUTPUT_DIR}\n",
    encoding="utf-8",
)
opposite = OUTPUT_DIR / (
    "RWR_DEG_ANALYSIS_INCOMPLETE.txt"
    if analysis_complete
    else "RWR_DEG_ANALYSIS_COMPLETE.txt"
)
if opposite.exists():
    opposite.unlink()


# ------------------------------------------------------------
# 11. User-facing completion report
# ------------------------------------------------------------
print("\n" + "=" * 82)
if analysis_complete:
    print("RWR-DEG REPEATED 10 x 5-FOLD BASELINE COMPLETED")
else:
    print("RWR-DEG PARTIALLY COMPLETED — SAFE TO RESUME")
print("=" * 82)
print(f"Completed folds : {complete_fold_count}/{TOTAL_FOLDS}")
print(f"Complete repeats: {len(complete_repeats)}/{REPEATS}")
print(f"New this session: {new_fold_count}")
print("Output          :", OUTPUT_DIR)

print("\nFold-level summary:")
try:
    display(fold_summary)
except NameError:
    print(fold_summary.to_string(index=False))

if not oof_summary.empty:
    print("\nRepeated OOF performance summary (complete repeats only):")
    try:
        display(oof_summary)
    except NameError:
        print(oof_summary.to_string(index=False))

if not stability_summary.empty:
    print("\nRepeated stability summary (complete repeats only):")
    try:
        display(stability_summary)
    except NameError:
        print(stability_summary.to_string(index=False))

if not gene_recurrence.empty:
    print("\nMost recurrent main RWR-DEG genes:")
    main_recurrence = gene_recurrence.loc[
        np.isclose(gene_recurrence["Restart_probability"], MAIN_RESTART)
    ].head(30)
    try:
        display(main_recurrence)
    except NameError:
        print(main_recurrence.to_string(index=False))

if analysis_complete and DISPLAY_FINAL_FIGURE and figure_png.exists():
    try:
        from IPython.display import Image, Markdown, display as ipy_display

        ipy_display(Markdown("## RWR-DEG network baseline — final repeated comparison"))
        ipy_display(Image(filename=str(figure_png), width=1200))
    except Exception as display_error:
        warnings.warn(f"Final figure could not be displayed inline: {display_error}")

print("\nGenerated aggregate files:")
for path in sorted(OUTPUT_DIR.glob("*")):
    if path.is_file():
        print(" -", path.name)

if not analysis_complete:
    print(
        "\nContinue with the same output folder and RWR_FORCE_RERUN=False. "
        "Use a fold count that completes the current repeat whenever practical."
    )
