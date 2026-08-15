# ============================================================
# INDEPENDENT EXTERNAL VALIDATION 2 — GSE70947 / GPL13607
# ============================================================
#
# Scientific role
# ---------------
# This script evaluates the already frozen NIBFS top-20 panel and the already
# fitted LR, RF, and LightGBM models in an independent, paired, cross-platform
# breast-tissue microarray cohort. It does NOT perform feature reselection,
# model refitting, hyperparameter tuning, or threshold optimization.
#
# Dataset
# -------
# GSE70947, GPL13607 (Agilent): 148 breast adenocarcinoma samples and
# 148 matched adjacent-normal samples (296 samples; 148 patient pairs).
#
# Run from Colab after placing this file in PROJECT_DIR/src:
#
#   GSE70947_PROJECT_DIR = str(PROJECT_DIR)
#   GSE70947_FORCE_RERUN = False
#   GSE70947_BOOTSTRAP_ITERATIONS = 2000
#   GSE70947_DISPLAY_FIGURE = True
#   %run -i "{PROJECT_DIR}/src/external_validation_GSE70947_V5.py"
#
# Optional local-file overrides (useful if an NCBI download is interrupted):
#
#   GSE70947_SERIES_MATRIX_PATH = "/path/GSE70947_series_matrix.txt.gz"
#   GSE70947_ANNOTATION_PATH = "/path/GPL13607.annot.gz"
#
# Outputs are written only to:
#   PROJECT_DIR/results/EXTERNAL_VALIDATION_GSE70947/
#
# Frozen main-run artifacts are read-only and are never overwritten.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import hashlib
from io import StringIO
import json
import os
import re
import sys
import warnings
import urllib.error

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from scipy.stats import t, ttest_rel, wilcoxon
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# -----------------------------------------------------------------------------
# 0. Configuration
# -----------------------------------------------------------------------------

GSE = "GSE70947"
GPL = "GPL13607"
EXPECTED_SAMPLES = 296
EXPECTED_PAIRS = 148
EXPECTED_PER_CLASS = 148
FINAL_K = 20
MODEL_NAMES = ("LR", "RF", "LightGBM")
DEFAULT_THRESHOLD = 0.5


def _resolve_project_dir() -> Path:
    override = globals().get("GSE70947_PROJECT_DIR")
    if override:
        candidate = Path(str(override)).expanduser().resolve()
        if (candidate / "src").is_dir():
            return candidate
        raise FileNotFoundError(
            f"GSE70947_PROJECT_DIR tidak valid: {candidate}"
        )

    active = globals().get("PROJECT_DIR")
    if active:
        candidate = Path(str(active)).expanduser().resolve()
        if (candidate / "src").is_dir():
            return candidate

    script_file = globals().get("__file__")
    if script_file:
        candidate = Path(str(script_file)).expanduser().resolve().parent.parent
        if (candidate / "src").is_dir():
            return candidate

    start = Path.cwd().resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "src").is_dir() and (candidate / "runs").exists():
            return candidate

    raise FileNotFoundError(
        "Folder proyek tidak dapat ditemukan. Tetapkan "
        "GSE70947_PROJECT_DIR ke folder utama proyek."
    )


PROJECT_DIR = _resolve_project_dir()
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

CONFIG_PATH = PROJECT_DIR / "config.yaml"
CFG: dict[str, Any] = {}
if CONFIG_PATH.exists():
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        CFG = yaml.safe_load(handle) or {}

RANDOM_STATE = int(
    globals().get(
        "GSE70947_RANDOM_STATE",
        CFG.get("project", {}).get("random_state", 42),
    )
)
BOOTSTRAP_ITERATIONS = int(
    globals().get(
        "GSE70947_BOOTSTRAP_ITERATIONS",
        CFG.get("external_validation", {}).get("bootstrap_iterations", 2000),
    )
)
FORCE_RERUN = bool(globals().get("GSE70947_FORCE_RERUN", False))
DISPLAY_FIGURE = bool(globals().get("GSE70947_DISPLAY_FIGURE", True))
REQUIRE_EXPECTED_PAIRING = bool(
    globals().get("GSE70947_REQUIRE_148_PAIRS", True)
)

if BOOTSTRAP_ITERATIONS < 200:
    raise ValueError("GSE70947_BOOTSTRAP_ITERATIONS harus >= 200.")

OUTPUT_DIR = PROJECT_DIR / "results" / "EXTERNAL_VALIDATION_GSE70947"
RAW_DIR = OUTPUT_DIR / "raw"
REFERENCE_DIR = OUTPUT_DIR / "reference"
FIGURE_DIR = OUTPUT_DIR / "figures"
for directory in (OUTPUT_DIR, RAW_DIR, REFERENCE_DIR, FIGURE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

COMPLETION_MARKER = OUTPUT_DIR / "EXTERNAL_VALIDATION_GSE70947_COMPLETE.txt"


# Project helpers are imported only after PROJECT_DIR is on sys.path.
try:
    from src.download_utils import (
        HGNC_COMPLETE_SET_URL,
        download_first_available,
        download_if_missing,
        geo_annotation_url,
        geo_series_urls,
    )
    from src.geo_io import (
        create_sample_metadata,
        parse_series_matrix,
        read_geo_annotation,
    )
    from src.hgnc import HGNCResolver
    from src.preprocessing import (
        conditional_log2_probe_table,
        quantile_normalize_samples,
    )
except Exception as exc:
    raise ImportError(
        "Modul proyek NIBFS tidak dapat diimpor. Pastikan script berada "
        "di PROJECT_DIR/src dan environment notebook sudah terpasang."
    ) from exc


# -----------------------------------------------------------------------------
# 1. Generic helpers and source-of-truth discovery
# -----------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _find_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lookup = {str(column).strip().lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        found = lookup.get(str(candidate).strip().lower())
        if found is not None:
            return found
    return None


def _bh_adjust(pvalues: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(pvalues), dtype=float)
    output = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return output
    p = values[valid]
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    output[valid] = restored
    return output


def _candidate_main_runs() -> list[Path]:
    override = globals().get("GSE70947_MAIN_RUN_DIR")
    if override:
        candidate = Path(str(override)).expanduser().resolve()
        return [candidate]

    candidates: list[Path] = []
    for panel_file in PROJECT_DIR.rglob("final_NIBFS_gene_panel_k20.csv"):
        # Expected path: RUN_DIR/results/main/tables/file.csv
        if len(panel_file.parents) >= 4:
            run_dir = panel_file.parents[3]
            if (run_dir / "results" / "main" / "tables").is_dir():
                candidates.append(run_dir)

    # A legacy alternative name is accepted only if the locked k=20 file is
    # unavailable in that run.
    for panel_file in PROJECT_DIR.rglob("final_top20_NIBFS_gene_panel.csv"):
        if len(panel_file.parents) >= 4:
            run_dir = panel_file.parents[3]
            if (run_dir / "results" / "main" / "tables").is_dir():
                candidates.append(run_dir)

    return sorted(set(candidates), key=lambda path: path.stat().st_mtime, reverse=True)


def _main_run_is_complete(run_dir: Path) -> bool:
    tables = run_dir / "results" / "main" / "tables"
    models = run_dir / "results" / "main" / "models"
    panel_ok = any(
        (tables / name).exists()
        for name in (
            "final_NIBFS_gene_panel_k20.csv",
            "final_top20_NIBFS_gene_panel.csv",
        )
    )
    threshold_ok = (tables / "discovery_OOF_Youden_thresholds.csv").exists()
    model_ok = all(
        (models / f"{name}_full_development_k20.joblib").exists()
        for name in MODEL_NAMES
    )
    return panel_ok and threshold_ok and model_ok


def _resolve_main_run() -> Path:
    candidates = _candidate_main_runs()
    complete = [candidate for candidate in candidates if _main_run_is_complete(candidate)]
    if not complete:
        details = "\n".join(str(candidate) for candidate in candidates[:10])
        raise FileNotFoundError(
            "Tidak ditemukan main run dengan frozen panel, OOF thresholds, "
            "dan tiga frozen model joblib lengkap. Kandidat yang ditemukan:\n"
            + (details or "(tidak ada)")
        )
    return complete[0]


MAIN_RUN_DIR = _resolve_main_run()
MAIN_TABLE_DIR = MAIN_RUN_DIR / "results" / "main" / "tables"
MAIN_MODEL_DIR = MAIN_RUN_DIR / "results" / "main" / "models"


def _resolve_panel_file() -> Path:
    for name in (
        "final_NIBFS_gene_panel_k20.csv",
        "final_top20_NIBFS_gene_panel.csv",
    ):
        candidate = MAIN_TABLE_DIR / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Frozen top-20 panel tidak ditemukan.")


PANEL_FILE = _resolve_panel_file()
THRESHOLD_FILE = MAIN_TABLE_DIR / "discovery_OOF_Youden_thresholds.csv"
MODEL_FILES = {
    name: MAIN_MODEL_DIR / f"{name}_full_development_k20.joblib"
    for name in MODEL_NAMES
}


# -----------------------------------------------------------------------------
# 2. GEO acquisition and paired sample metadata
# -----------------------------------------------------------------------------


def _resolve_or_download_geo_files() -> tuple[Path, Path]:
    series_override = globals().get("GSE70947_SERIES_MATRIX_PATH")
    annotation_override = globals().get("GSE70947_ANNOTATION_PATH")

    if series_override:
        series_path = Path(str(series_override)).expanduser().resolve()
        if not series_path.exists():
            raise FileNotFoundError(series_path)
    else:
        series_path = download_first_available(
            geo_series_urls(GSE, GPL),
            RAW_DIR / f"{GSE}_series_matrix.txt.gz",
            timeout=600,
        )

    if annotation_override:
        annotation_path = Path(str(annotation_override)).expanduser().resolve()
        if not annotation_path.exists():
            raise FileNotFoundError(annotation_path)
    else:
        # Some GEO platforms, including GPL13607, may not expose the optional
        # pre-built *.annot.gz file. First try the conventional annotation URL;
        # on HTTP 404, retrieve the complete official GEO platform record instead.
        # read_geo_annotation() can parse the platform table because it begins
        # with the same tab-delimited ID header used by GEO annotation files.
        conventional_url = geo_annotation_url(GPL)
        conventional_path = RAW_DIR / f"{GPL}.annot.gz"
        try:
            print(f"Trying platform annotation: {conventional_url}", flush=True)
            annotation_path = download_if_missing(
                conventional_url,
                conventional_path,
                timeout=600,
            )
            print(f"Platform annotation available: {annotation_path}", flush=True)
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise

            conventional_path.unlink(missing_ok=True)
            fallback_url = (
                "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
                f"?acc={GPL}&targ=self&view=full&form=text"
            )
            fallback_path = RAW_DIR / f"{GPL}_platform_full.txt"
            print(
                f"{GPL}.annot.gz is not available (HTTP 404). "
                "Using the official full GEO platform table instead.",
                flush=True,
            )
            print(f"Trying platform fallback: {fallback_url}", flush=True)
            annotation_path = download_if_missing(
                fallback_url,
                fallback_path,
                timeout=600,
            )
            print(f"Platform fallback available: {annotation_path}", flush=True)

    return Path(series_path), Path(annotation_path)


def _get_hgnc_resolver() -> tuple[HGNCResolver, Path]:
    active = globals().get("resolver")
    if active is not None and isinstance(active, HGNCResolver):
        # Still identify a provenance file when possible.
        candidates = list(MAIN_RUN_DIR.rglob("hgnc_complete_set.txt"))
        if candidates:
            return active, max(candidates, key=lambda path: path.stat().st_mtime)

    candidates = [
        path for path in MAIN_RUN_DIR.rglob("hgnc_complete_set.txt")
        if path.is_file() and path.stat().st_size > 0
    ]
    if candidates:
        hgnc_path = max(candidates, key=lambda path: path.stat().st_mtime)
    else:
        hgnc_path = download_if_missing(
            HGNC_COMPLETE_SET_URL,
            REFERENCE_DIR / "hgnc_complete_set.txt",
            timeout=600,
        )
    return HGNCResolver.from_complete_set(hgnc_path), Path(hgnc_path)


def _label_and_pair_samples(metadata: pd.DataFrame) -> pd.DataFrame:
    out = metadata.copy()
    title_col = _find_column(out, ["Sample_title"])
    if title_col is None:
        raise KeyError("Sample_title tidak ditemukan pada Series Matrix GSE70947.")

    titles = out[title_col].fillna("").astype(str).str.strip()

    # Official GSE70947 Series Matrix titles use the exact paired form
    # "CM016-normal" / "CM016-tumor" (also CMG identifiers). Parse the
    # phenotype only from the terminal suffix, so labels are not inferred from
    # row order or GSM number. Legacy .N.c2/.T.c2 descriptions are retained as
    # a conservative fallback for compatibility with GEO sample descriptions.
    extracted = titles.str.extract(
        r"(?i)^(?P<Pair_ID>.+?)-(?P<Tissue>normal|tumou?r)$"
    )
    parser_rule = pd.Series("hyphen-normal-tumor", index=out.index, dtype="string")

    missing = extracted["Pair_ID"].isna()
    if missing.any():
        legacy = titles.str.extract(
            r"(?i)^(?P<Pair_ID>.+?)\.(?P<Tissue>[NT])(?:\.c2)?$"
        )
        legacy_ok = missing & legacy["Pair_ID"].notna()
        extracted.loc[legacy_ok, ["Pair_ID", "Tissue"]] = legacy.loc[
            legacy_ok, ["Pair_ID", "Tissue"]
        ]
        parser_rule.loc[legacy_ok] = "legacy-dot-N-T"

    out["Pair_ID"] = extracted["Pair_ID"].astype("string").str.strip()
    tissue = extracted["Tissue"].astype("string").str.lower().str.strip()
    tissue_code = tissue.replace(
        {
            "normal": "N",
            "n": "N",
            "tumor": "T",
            "tumour": "T",
            "t": "T",
        }
    )
    out["Label_binary"] = tissue_code.map({"N": 0, "T": 1})
    out["Tissue_class"] = out["Label_binary"].map(
        {0: "Adjacent normal", 1: "Breast adenocarcinoma"}
    )
    out["Sample_title_parser"] = parser_rule

    unresolved = out["Label_binary"].isna() | out["Pair_ID"].isna()
    if unresolved.any():
        bad = out.loc[unresolved, ["GSM_ID", title_col]].head(20)
        raise ValueError(
            "Label atau Pair_ID tidak dapat ditentukan dari Sample_title.\n"
            + bad.to_string(index=False)
        )

    out["Label_binary"] = out["Label_binary"].astype(int)
    out["Pair_ID"] = out["Pair_ID"].astype(str)

    if out["GSM_ID"].duplicated().any():
        duplicated = out.loc[out["GSM_ID"].duplicated(), "GSM_ID"].tolist()
        raise ValueError(f"GSM duplikat ditemukan: {duplicated[:10]}")

    pair_counts = out.groupby("Pair_ID")["Label_binary"].agg(["size", "nunique", "sum"])
    invalid_pairs = pair_counts[
        (pair_counts["size"] != 2)
        | (pair_counts["nunique"] != 2)
        | (pair_counts["sum"] != 1)
    ]
    if not invalid_pairs.empty:
        raise ValueError(
            "Pasangan GSE70947 tidak lengkap atau tidak berisi satu normal dan "
            "satu kanker. Contoh:\n" + invalid_pairs.head(20).to_string()
        )

    counts = out["Label_binary"].value_counts().to_dict()
    if REQUIRE_EXPECTED_PAIRING:
        observed = {
            "samples": int(len(out)),
            "pairs": int(out["Pair_ID"].nunique()),
            "normal": int(counts.get(0, 0)),
            "cancer": int(counts.get(1, 0)),
        }
        expected = {
            "samples": EXPECTED_SAMPLES,
            "pairs": EXPECTED_PAIRS,
            "normal": EXPECTED_PER_CLASS,
            "cancer": EXPECTED_PER_CLASS,
        }
        if observed != expected:
            raise RuntimeError(
                f"Komposisi GSE70947 tidak sesuai desain resmi. "
                f"Observed={observed}; expected={expected}."
            )

    return out.sort_values(["Pair_ID", "Label_binary"]).reset_index(drop=True)



# -----------------------------------------------------------------------------
# 2b. GPL13607-specific annotation parser
# -----------------------------------------------------------------------------


def _normalise_refseq_token(value: object) -> str | None:
    """Return an unversioned RefSeq RNA accession, or None.

    Only RNA accessions are accepted. This prevents generic GenBank accessions,
    Ensembl transcript IDs, and free-text identifiers from being treated as
    gene identifiers.
    """
    if value is None or pd.isna(value):
        return None
    token = str(value).strip().strip('"').upper()
    match = re.fullmatch(r"((?:NM|NR|XM|XR)_\d+)(?:\.\d+)?", token)
    return match.group(1) if match else None


def _extract_refseq_tokens(value: object) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    text = str(value).upper()
    return {
        match.group(1)
        for match in re.finditer(r"\b((?:NM|NR|XM|XR)_\d+)(?:\.\d+)?\b", text)
    }


def _build_unique_hgnc_refseq_map(hgnc_path: Path) -> dict[str, str]:
    """Build an unambiguous RefSeq-RNA-to-approved-symbol map from HGNC.

    The complete HGNC table is the same frozen reference used by the project.
    Only rows with status=Approved are retained. An accession is accepted only
    when it maps to exactly one approved symbol.
    """
    hgnc = pd.read_csv(hgnc_path, sep="\t", dtype=str, low_memory=False)
    lookup = {
        str(column).strip().lower().replace("_", "").replace(" ", ""): column
        for column in hgnc.columns
    }
    symbol_col = lookup.get("symbol")
    status_col = lookup.get("status")
    if symbol_col is None or status_col is None:
        raise ValueError(
            "HGNC complete set tidak memiliki kolom symbol/status yang diperlukan."
        )

    hgnc = hgnc.loc[
        hgnc[status_col].fillna("").str.strip().str.lower().eq("approved")
    ].copy()

    refseq_columns = [
        column
        for key, column in lookup.items()
        if key in {
            "refseqaccession",
            "refseqaccessions",
            "refseq",
            "refseqid",
            "refseqids",
        }
    ]
    if not refseq_columns:
        raise ValueError(
            "HGNC complete set tidak memiliki kolom RefSeq accession. "
            f"Columns={hgnc.columns.tolist()}"
        )

    candidates: dict[str, set[str]] = {}
    for _, row in hgnc[[symbol_col] + refseq_columns].iterrows():
        symbol = str(row[symbol_col]).strip()
        if not symbol or symbol.lower() in {"nan", "none"}:
            continue
        for column in refseq_columns:
            for accession in _extract_refseq_tokens(row[column]):
                candidates.setdefault(accession, set()).add(symbol)

    unique = {
        accession: next(iter(symbols))
        for accession, symbols in candidates.items()
        if len(symbols) == 1
    }
    if not unique:
        raise RuntimeError("HGNC RefSeq map kosong setelah pemeriksaan unambiguity.")
    return unique


def _read_gpl13607_full_table(path: Path) -> pd.DataFrame:
    lines: list[str] = []
    started = False
    if str(path).lower().endswith(".gz"):
        import gzip
        handle_context = gzip.open(
            path, "rt", encoding="utf-8", errors="replace"
        )
    else:
        handle_context = open(
            path, "r", encoding="utf-8", errors="replace"
        )

    with handle_context as handle:
        for line in handle:
            if line.startswith("ID\t"):
                started = True
            if started:
                lines.append(line)

    if not lines:
        raise ValueError(f"Annotation table tidak ditemukan di {path}")

    annot = pd.read_csv(
        StringIO("".join(lines)),
        sep="\t",
        dtype=str,
        low_memory=False,
    )
    annot.columns = [str(column).strip() for column in annot.columns]
    return annot


def _download_optional_old_gpl13607_annotation() -> Path | None:
    """Download GEO's official supplementary annotation when available.

    This is not a third-party mapping. It is the supplementary annotation file
    linked from the GPL13607 GEO record. It is used only as corroborating or
    fallback evidence and is harmonised through the same HGNC resolver.
    """
    url = (
        "https://www.ncbi.nlm.nih.gov/geo/download/"
        "?acc=GPL13607&file=GPL13607_old_annotations.txt.gz&format=file"
    )
    target = RAW_DIR / "GPL13607_old_annotations.txt.gz"
    try:
        return Path(download_if_missing(url, target, timeout=600))
    except Exception as exc:
        warnings.warn(
            "Official GPL13607 supplementary annotation could not be loaded; "
            f"continuing with GeneName + RefSeq evidence from the full platform table: {exc}"
        )
        return None


def _read_gpl13607_annotation(
    path: Path,
    resolver_object: HGNCResolver,
    hgnc_path: Path,
) -> pd.DataFrame:
    """Map GPL13607 feature IDs to current HGNC symbols without substitution.

    Evidence hierarchy
    ------------------
    1. GeneName resolved through the project's HGNC approved/alias resolver.
    2. RefSeq RNA accessions in GB_ACC/accessions resolved through the same
       frozen HGNC complete-set reference.
    3. GEO's official GPL13607 supplementary annotation, when available.

    A feature is retained only when all available mapped evidence agrees on one
    approved symbol. This specifically recovers old/ambiguous labels such as
    CDC2 through the approved CDK1 RefSeq accession, while preserving the rule
    that no missing frozen gene may be replaced by a different gene.
    """
    annot = _read_gpl13607_full_table(path)
    column_lookup = {
        str(column).strip().lower().replace("_", "").replace(" ", ""): column
        for column in annot.columns
    }
    id_col = column_lookup.get("id")
    gene_col = None
    for candidate in ("genesymbol", "symbol", "genename", "gene"):
        if candidate in column_lookup:
            gene_col = column_lookup[candidate]
            break
    gb_col = column_lookup.get("gbacc")
    accessions_col = column_lookup.get("accessions")
    probe_col = column_lookup.get("probename")
    description_col = column_lookup.get("description")

    if id_col is None or gene_col is None:
        raise ValueError(
            "GPL13607 annotation columns tidak dikenali. "
            f"Columns={annot.columns.tolist()}"
        )

    keep_columns = [id_col, gene_col]
    for optional in (gb_col, accessions_col, probe_col, description_col):
        if optional is not None and optional not in keep_columns:
            keep_columns.append(optional)
    raw = annot[keep_columns].copy()
    rename_map = {id_col: "ID_REF", gene_col: "Raw_Gene"}
    if gb_col is not None:
        rename_map[gb_col] = "GB_ACC"
    if accessions_col is not None:
        rename_map[accessions_col] = "Accessions"
    if probe_col is not None:
        rename_map[probe_col] = "ProbeName"
    if description_col is not None:
        rename_map[description_col] = "Description"
    raw = raw.rename(columns=rename_map)
    for column in ("GB_ACC", "Accessions", "ProbeName", "Description"):
        if column not in raw.columns:
            raw[column] = ""

    raw["ID_REF"] = raw["ID_REF"].astype(str).str.strip()
    raw["Raw_Gene"] = raw["Raw_Gene"].fillna("").astype(str).str.strip()
    raw["Gene_from_symbol"] = raw["Raw_Gene"].map(
        resolver_object.resolve_annotation_cell
    )

    refseq_to_gene = _build_unique_hgnc_refseq_map(hgnc_path)

    def map_accession_evidence(row: pd.Series) -> str | None:
        tokens = set()
        tokens.update(_extract_refseq_tokens(row.get("GB_ACC")))
        tokens.update(_extract_refseq_tokens(row.get("Accessions")))
        # Description is used only to recover explicit RefSeq accessions, not
        # free-text gene names.
        tokens.update(_extract_refseq_tokens(row.get("Description")))
        genes = {refseq_to_gene[token] for token in tokens if token in refseq_to_gene}
        return next(iter(genes)) if len(genes) == 1 else None

    raw["Gene_from_refseq"] = raw.apply(map_accession_evidence, axis=1)

    def reconcile(row: pd.Series) -> tuple[str | None, str, bool]:
        symbol_gene = row["Gene_from_symbol"]
        refseq_gene = row["Gene_from_refseq"]
        evidence = {gene for gene in (symbol_gene, refseq_gene) if pd.notna(gene)}
        if len(evidence) == 1:
            gene = next(iter(evidence))
            if pd.notna(symbol_gene) and pd.notna(refseq_gene):
                source = "GeneName+RefSeq_agree"
            elif pd.notna(symbol_gene):
                source = "GeneName_HGNC"
            else:
                source = "RefSeq_HGNC"
            return gene, source, False
        if len(evidence) > 1:
            return None, "conflicting_full_table_evidence", True
        return None, "unresolved_full_table", False

    reconciled = raw.apply(reconcile, axis=1, result_type="expand")
    reconciled.columns = ["Gene_full", "Mapping_source_full", "Evidence_conflict"]
    raw = pd.concat([raw, reconciled], axis=1)

    # Resolve duplicated feature IDs conservatively within the full table.
    full_candidates = raw.dropna(subset=["Gene_full"])[["ID_REF", "Gene_full"]].copy()
    full_candidates = full_candidates.rename(columns={"Gene_full": "Gene"})
    full_counts = full_candidates.groupby("ID_REF")["Gene"].nunique()
    full_valid_ids = full_counts[full_counts == 1].index
    full_map = (
        full_candidates.loc[full_candidates["ID_REF"].isin(full_valid_ids)]
        .drop_duplicates("ID_REF")
        .assign(Evidence_source="GPL13607_full_GeneName_or_RefSeq")
    )

    old_annotation_path = _download_optional_old_gpl13607_annotation()
    old_map = pd.DataFrame(columns=["ID_REF", "Gene", "Evidence_source"])
    if old_annotation_path is not None:
        try:
            old_parsed = read_geo_annotation(old_annotation_path, resolver_object)
            old_parsed = old_parsed[["ID_REF", "Gene"]].copy()
            old_parsed["ID_REF"] = old_parsed["ID_REF"].astype(str).str.strip()
            old_counts = old_parsed.groupby("ID_REF")["Gene"].nunique()
            old_valid_ids = old_counts[old_counts == 1].index
            old_map = (
                old_parsed.loc[old_parsed["ID_REF"].isin(old_valid_ids)]
                .drop_duplicates("ID_REF")
                .assign(Evidence_source="GEO_official_old_annotation_HGNC")
            )
        except Exception as exc:
            warnings.warn(
                "Official GPL13607 supplementary annotation was downloaded "
                f"but could not be parsed; continuing without it: {exc}"
            )
            old_annotation_path = None

    combined = pd.concat([full_map, old_map], ignore_index=True)
    combined_counts = combined.groupby("ID_REF")["Gene"].nunique()
    valid_ids = combined_counts[combined_counts == 1].index
    out = (
        combined.loc[combined["ID_REF"].isin(valid_ids), ["ID_REF", "Gene"]]
        .drop_duplicates("ID_REF")
        .reset_index(drop=True)
    )

    if out.empty:
        raise RuntimeError(
            "Tidak ada simbol gen GPL13607 yang berhasil dipetakan melalui "
            "GeneName/RefSeq dan HGNC."
        )

    # Preserve a complete machine-readable audit of the annotation decision.
    raw.to_csv(
        OUTPUT_DIR / "external_GSE70947_annotation_mapping_detail.csv.gz",
        index=False,
        compression="gzip",
    )
    combined.to_csv(
        OUTPUT_DIR / "external_GSE70947_annotation_evidence_combined.csv.gz",
        index=False,
        compression="gzip",
    )

    trace_pattern = re.compile(
        r"CDK1|CDC2|NM_001786|NM_033379|NM_001170406",
        flags=re.IGNORECASE,
    )
    trace_mask = raw[
        ["Raw_Gene", "GB_ACC", "Accessions", "ProbeName", "Description"]
    ].fillna("").astype(str).apply(
        lambda column: column.str.contains(trace_pattern, regex=True)
    ).any(axis=1)
    trace = raw.loc[trace_mask].copy()
    mapped_cdk1_ids = set(out.loc[out["Gene"].eq("CDK1"), "ID_REF"])
    if mapped_cdk1_ids:
        trace = pd.concat(
            [trace, raw.loc[raw["ID_REF"].isin(mapped_cdk1_ids)]],
            ignore_index=True,
        ).drop_duplicates()
    trace.to_csv(
        OUTPUT_DIR / "external_GSE70947_CDK1_annotation_trace.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "Platform": GPL,
                "Primary_annotation_file": str(path),
                "Supplementary_annotation_file": (
                    str(old_annotation_path) if old_annotation_path is not None else ""
                ),
                "ID_column": id_col,
                "Gene_column": gene_col,
                "GB_ACC_column": gb_col or "",
                "Accessions_column": accessions_col or "",
                "Platform_rows": int(len(annot)),
                "Full_table_mapped_features": int(len(full_map)),
                "Supplementary_mapped_features": int(len(old_map)),
                "Final_unambiguous_feature_mappings": int(len(out)),
                "Approved_gene_symbols": int(out["Gene"].nunique()),
                "CDK1_feature_count": int((out["Gene"] == "CDK1").sum()),
                "Parser": (
                    "GPL13607 GeneName + RefSeq accession + project HGNC; "
                    "official GEO supplementary annotation when available"
                ),
            }
        ]
    ).to_csv(
        OUTPUT_DIR / "external_GSE70947_annotation_parser_audit.csv",
        index=False,
    )

    print("GPL13607 annotation parser V5:", flush=True)
    print(f"  ID column                 : {id_col}", flush=True)
    print(f"  Gene column               : {gene_col}", flush=True)
    print(f"  GB_ACC column             : {gb_col}", flush=True)
    print(f"  Platform rows             : {len(annot):,}", flush=True)
    print(f"  Full-table mappings       : {len(full_map):,}", flush=True)
    print(f"  Supplementary mappings    : {len(old_map):,}", flush=True)
    print(f"  Final unambiguous probes  : {len(out):,}", flush=True)
    print(f"  Approved genes            : {out['Gene'].nunique():,}", flush=True)
    print(f"  CDK1 mapped probes        : {(out['Gene'] == 'CDK1').sum():,}", flush=True)

    return out


# -----------------------------------------------------------------------------
# 3. Label-free external preprocessing
# -----------------------------------------------------------------------------


def _coverage_then_variance_probe_mapping(
    expression: pd.DataFrame,
    probe_map: pd.DataFrame,
    sample_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Collapse probes to genes without using phenotype labels.

    The GSE70947 processed Series Matrix contains blank values for some features
    that were flagged by the original submitters. Therefore, probe selection is
    deterministic and label-free: maximum sample coverage first, then highest
    variance, then lexical ID_REF. Remaining sporadic missing values are filled
    with the external-cohort gene median before independent quantile
    normalization. All decisions and counts are exported for audit.
    """
    expr = expression[["ID_REF"] + sample_ids].copy()
    expr["ID_REF"] = expr["ID_REF"].astype(str).str.strip()
    expr[sample_ids] = expr[sample_ids].apply(pd.to_numeric, errors="coerce")

    mapped = expr.merge(probe_map, on="ID_REF", how="inner", validate="many_to_one")
    if mapped.empty:
        raise RuntimeError("Tidak ada probe GPL13607 yang berhasil dipetakan ke HGNC.")

    mapped["Non_missing_count"] = mapped[sample_ids].notna().sum(axis=1)
    mapped["Missing_count"] = len(sample_ids) - mapped["Non_missing_count"]
    mapped["Probe_variance"] = mapped[sample_ids].var(axis=1, ddof=1, skipna=True)
    mapped["Probe_variance"] = mapped["Probe_variance"].fillna(-np.inf)

    selected = (
        mapped.sort_values(
            ["Gene", "Non_missing_count", "Probe_variance", "ID_REF"],
            ascending=[True, False, False, True],
            kind="mergesort",
        )
        .drop_duplicates("Gene", keep="first")
        .copy()
    )

    selected = selected[selected["Non_missing_count"] > 0].copy()
    if selected.empty:
        raise RuntimeError("Semua probe terpilih kosong.")

    mapping_audit = selected[
        [
            "Gene",
            "ID_REF",
            "Non_missing_count",
            "Missing_count",
            "Probe_variance",
        ]
    ].sort_values("Gene").reset_index(drop=True)

    X = selected[["Gene"] + sample_ids].set_index("Gene").T
    X.index.name = "GSM_ID"
    X = X.apply(pd.to_numeric, errors="coerce")

    missing_before = int(X.isna().sum().sum())
    genes_with_missing = int((X.isna().sum(axis=0) > 0).sum())

    # External-cohort median imputation is phenotype-blind and only addresses
    # submitter-flagged blank processed values.
    medians = X.median(axis=0, skipna=True)
    all_missing_genes = medians[medians.isna()].index.tolist()
    if all_missing_genes:
        X = X.drop(columns=all_missing_genes)
        mapping_audit = mapping_audit[
            ~mapping_audit["Gene"].isin(all_missing_genes)
        ].copy()
        medians = X.median(axis=0, skipna=True)

    X = X.fillna(medians)
    missing_after = int(X.isna().sum().sum())
    if missing_after:
        raise RuntimeError(
            f"Masih ada {missing_after} nilai kosong setelah median imputation."
        )

    X_qn = quantile_normalize_samples(X)
    if not np.isfinite(X_qn.to_numpy(dtype=float)).all():
        raise RuntimeError("Nilai non-finite ditemukan setelah quantile normalization.")

    audit = {
        "probes_in_series_matrix": int(len(expr)),
        "mapped_probe_rows": int(len(mapped)),
        "genes_after_probe_collapse": int(X.shape[1]),
        "genes_removed_all_missing": int(len(all_missing_genes)),
        "genes_with_any_missing_before_imputation": genes_with_missing,
        "missing_values_before_imputation": missing_before,
        "missing_values_after_imputation": missing_after,
        "probe_selection_rule": (
            "maximum non-missing sample coverage, then highest variance, "
            "then lexical ID_REF"
        ),
        "imputation_rule": "external-cohort median per gene; labels unused",
        "normalization_rule": (
            "external-only quantile normalization after gene-level collapse; "
            "labels unused"
        ),
    }
    return X_qn, mapping_audit, audit


# -----------------------------------------------------------------------------
# 4. Frozen artifacts, prediction, and metrics
# -----------------------------------------------------------------------------


def _load_frozen_panel() -> tuple[pd.DataFrame, list[str]]:
    panel = pd.read_csv(PANEL_FILE)
    gene_col = _find_column(panel, ["Gene", "gene", "Symbol"])
    if gene_col is None:
        raise KeyError(f"Kolom Gene tidak ditemukan pada {PANEL_FILE}")
    panel = panel.rename(columns={gene_col: "Gene"}).copy()
    panel["Gene"] = panel["Gene"].astype(str).str.strip()

    rank_col = _find_column(panel, ["Rank_NIBFS", "Selection_rank", "Rank"])
    if rank_col is not None:
        panel = panel.sort_values(rank_col, kind="mergesort")

    panel = panel.drop_duplicates("Gene").head(FINAL_K).reset_index(drop=True)
    if len(panel) != FINAL_K:
        raise RuntimeError(f"Frozen panel harus berisi {FINAL_K} gen; ditemukan {len(panel)}.")
    genes = panel["Gene"].tolist()
    return panel, genes


def _load_thresholds() -> pd.DataFrame:
    thresholds = pd.read_csv(THRESHOLD_FILE)
    method_col = _find_column(thresholds, ["Feature_selection_method", "Method"])
    classifier_col = _find_column(thresholds, ["Classifier", "Model"])
    k_col = _find_column(thresholds, ["k", "K"])
    threshold_col = _find_column(thresholds, ["Threshold"])
    rule_col = _find_column(thresholds, ["Rule", "Threshold_source"])

    required = [method_col, classifier_col, k_col, threshold_col]
    if any(column is None for column in required):
        raise KeyError(
            "Struktur discovery_OOF_Youden_thresholds.csv tidak lengkap: "
            f"{thresholds.columns.tolist()}"
        )

    keep = thresholds[
        thresholds[method_col].astype(str).str.upper().eq("NIBFS")
        & pd.to_numeric(thresholds[k_col], errors="coerce").eq(FINAL_K)
    ].copy()
    if keep.empty:
        raise RuntimeError("Threshold NIBFS k=20 tidak ditemukan.")

    out = pd.DataFrame(
        {
            "Classifier": keep[classifier_col].astype(str),
            "Threshold": pd.to_numeric(keep[threshold_col], errors="raise"),
            "Rule": (
                keep[rule_col].astype(str)
                if rule_col is not None
                else "Youden index from discovery OOF predictions"
            ),
        }
    )
    out = out.drop_duplicates("Classifier")
    missing = sorted(set(MODEL_NAMES) - set(out["Classifier"]))
    if missing:
        raise RuntimeError(f"OOF threshold tidak tersedia untuk: {missing}")
    return out.sort_values("Classifier").reset_index(drop=True)


def _load_models(frozen_genes: list[str]) -> tuple[dict[str, Any], dict[str, list[str]]]:
    models: dict[str, Any] = {}
    model_feature_order: dict[str, list[str]] = {}

    for name, path in MODEL_FILES.items():
        if not path.exists():
            raise FileNotFoundError(path)
        model = joblib.load(path)
        feature_names = getattr(model, "feature_names_in_", None)
        if feature_names is None:
            feature_names = getattr(model, "feature_name_", None)
        if feature_names is None and hasattr(model, "booster_"):
            try:
                feature_names = model.booster_.feature_name()
            except Exception:
                feature_names = None

        if feature_names is None:
            n_features = getattr(model, "n_features_in_", None)
            if n_features is None or int(n_features) != FINAL_K:
                raise RuntimeError(
                    f"Jumlah fitur frozen model {name} tidak dapat diverifikasi."
                )
            warnings.warn(
                f"Frozen model {name} tidak menyimpan nama fitur; "
                "urutan frozen panel dari file utama digunakan. Model tidak "
                "di-fit ulang.",
                RuntimeWarning,
            )
            feature_order = list(frozen_genes)
        else:
            feature_order = [str(value) for value in feature_names]

        if len(feature_order) != FINAL_K or set(feature_order) != set(frozen_genes):
            raise RuntimeError(
                f"Fitur model {name} tidak identik dengan frozen top-20. "
                f"Model={feature_order}; frozen={frozen_genes}"
            )
        models[name] = model
        model_feature_order[name] = feature_order

    return models, model_feature_order


def _classification_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    predicted = (p >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "Threshold": float(threshold),
        "ROC_AUC": float(roc_auc_score(y, p)),
        "PR_AUC": float(average_precision_score(y, p)),
        "Accuracy": float(accuracy_score(y, predicted)),
        "Balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "Sensitivity": float(recall_score(y, predicted, zero_division=0)),
        "Specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "Precision": float(precision_score(y, predicted, zero_division=0)),
        "F1": float(f1_score(y, predicted, zero_division=0)),
        "MCC": float(matthews_corrcoef(y, predicted)),
        "Brier_score": float(brier_score_loss(y, p)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def _pair_cluster_bootstrap(
    frame: pd.DataFrame,
    threshold: float,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    # Build a fixed (n_pairs x 2) integer-index matrix once. This preserves the
    # matched-pair cluster while avoiding thousands of expensive DataFrame
    # concatenations during bootstrap resampling.
    reset = frame.reset_index(drop=True)
    pair_groups = [
        group.index.to_numpy(dtype=int)
        for _, group in reset.groupby("Pair_ID", sort=True)
    ]
    if not pair_groups or any(len(indices) != 2 for indices in pair_groups):
        raise RuntimeError("Pair-cluster bootstrap requires exactly two samples per pair.")
    pair_index_matrix = np.vstack(pair_groups)
    y_all = reset["True_Label"].to_numpy(dtype=int)
    p_all = reset["Probability"].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    metric_names = [
        "ROC_AUC",
        "PR_AUC",
        "Accuracy",
        "Balanced_accuracy",
        "Sensitivity",
        "Specificity",
        "Precision",
        "F1",
        "MCC",
        "Brier_score",
    ]
    collected = {name: [] for name in metric_names}
    n_pairs = pair_index_matrix.shape[0]

    for _ in range(iterations):
        sampled_pair_rows = rng.integers(0, n_pairs, size=n_pairs)
        sampled_sample_indices = pair_index_matrix[sampled_pair_rows].reshape(-1)
        values = _classification_metrics(
            y_all[sampled_sample_indices],
            p_all[sampled_sample_indices],
            threshold,
        )
        for name in metric_names:
            collected[name].append(values[name])

    rows = []
    for name in metric_names:
        values = np.asarray(collected[name], dtype=float)
        rows.append(
            {
                "Metric": name,
                "Bootstrap_iterations": int(iterations),
                "CI_method": "pair-cluster percentile bootstrap",
                "CI_low": float(np.nanquantile(values, 0.025)),
                "CI_high": float(np.nanquantile(values, 0.975)),
                "Bootstrap_mean": float(np.nanmean(values)),
                "Bootstrap_SD": float(np.nanstd(values, ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def _predict(
    X_panel: pd.DataFrame,
    metadata: pd.DataFrame,
    models: dict[str, Any],
    feature_orders: dict[str, list[str]],
) -> pd.DataFrame:
    meta = metadata.set_index("GSM_ID").loc[X_panel.index]
    rows = []
    for classifier in MODEL_NAMES:
        order = feature_orders[classifier]
        probability = models[classifier].predict_proba(X_panel[order])[:, 1]
        for sample_id, value in zip(X_panel.index, probability):
            rows.append(
                {
                    "Dataset": "Independent external GSE70947",
                    "Sample_ID": str(sample_id),
                    "Pair_ID": str(meta.loc[sample_id, "Pair_ID"]),
                    "Tissue_class": str(meta.loc[sample_id, "Tissue_class"]),
                    "True_Label": int(meta.loc[sample_id, "Label_binary"]),
                    "Feature_selection_method": "NIBFS",
                    "Classifier": classifier,
                    "k": FINAL_K,
                    "Probability": float(value),
                }
            )
    return pd.DataFrame(rows)


def _evaluate_predictions(
    predictions: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    threshold_lookup = thresholds.set_index("Classifier")
    metric_rows = []
    ci_parts = []

    for classifier, group in predictions.groupby("Classifier", sort=False):
        rules = [
            ("Default", DEFAULT_THRESHOLD, "Default 0.5"),
            (
                "OOF-transferred",
                float(threshold_lookup.loc[classifier, "Threshold"]),
                str(threshold_lookup.loc[classifier, "Rule"]),
            ),
        ]
        for rule_name, threshold, threshold_source in rules:
            metrics = _classification_metrics(
                group["True_Label"].to_numpy(int),
                group["Probability"].to_numpy(float),
                threshold,
            )
            metric_rows.append(
                {
                    "Dataset": "Independent external GSE70947",
                    "Feature_selection_method": "NIBFS",
                    "Classifier": classifier,
                    "k": FINAL_K,
                    "Evaluation_rule": rule_name,
                    "Threshold_source": threshold_source,
                    "Samples": int(len(group)),
                    "Pairs": int(group["Pair_ID"].nunique()),
                    **metrics,
                }
            )

            seed = RANDOM_STATE + 1000 * MODEL_NAMES.index(classifier) + (
                0 if rule_name == "Default" else 1
            )
            ci = _pair_cluster_bootstrap(
                group,
                threshold,
                BOOTSTRAP_ITERATIONS,
                seed,
            )
            ci.insert(0, "Evaluation_rule", rule_name)
            ci.insert(0, "Classifier", classifier)
            ci_parts.append(ci)

    metrics_df = pd.DataFrame(metric_rows)
    ci_df = pd.concat(ci_parts, ignore_index=True)
    return metrics_df, ci_df


# -----------------------------------------------------------------------------
# 5. Paired expression-direction consistency
# -----------------------------------------------------------------------------


def _direction_consistency(
    X_panel: pd.DataFrame,
    metadata: pd.DataFrame,
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logfc_col = _find_column(panel, ["logFC", "log2FC", "Training_logFC"])
    if logfc_col is None:
        raise KeyError(
            "Frozen panel tidak memiliki kolom logFC untuk direction consistency."
        )
    discovery_logfc = panel.set_index("Gene")[logfc_col].astype(float)
    rank_col = _find_column(panel, ["Rank_NIBFS", "Selection_rank", "Rank"])
    rank_lookup = (
        panel.set_index("Gene")[rank_col].astype(float)
        if rank_col is not None
        else pd.Series(np.arange(1, len(panel) + 1), index=panel["Gene"])
    )

    meta = metadata.set_index("GSM_ID").loc[X_panel.index]
    rows = []
    for gene in panel["Gene"]:
        values = pd.DataFrame(
            {
                "Pair_ID": meta["Pair_ID"].astype(str),
                "Label_binary": meta["Label_binary"].astype(int),
                "Expression": X_panel[gene].astype(float),
            },
            index=X_panel.index,
        )
        paired = values.pivot(index="Pair_ID", columns="Label_binary", values="Expression")
        if set(paired.columns) != {0, 1} or paired.isna().any().any():
            raise RuntimeError(f"Pasangan ekspresi tidak lengkap untuk gen {gene}.")
        delta = paired[1] - paired[0]
        n = len(delta)
        mean_delta = float(delta.mean())
        median_delta = float(delta.median())
        sd_delta = float(delta.std(ddof=1))
        sem = sd_delta / np.sqrt(n) if n > 1 else np.nan
        critical = float(t.ppf(0.975, df=n - 1)) if n > 1 else np.nan
        ci_low = mean_delta - critical * sem if n > 1 else np.nan
        ci_high = mean_delta + critical * sem if n > 1 else np.nan

        t_result = ttest_rel(paired[1], paired[0], nan_policy="raise")
        try:
            w_result = wilcoxon(paired[1], paired[0], zero_method="wilcox")
            wilcoxon_p = float(w_result.pvalue)
        except ValueError:
            wilcoxon_p = 1.0

        training_value = float(discovery_logfc.loc[gene])
        rows.append(
            {
                "Gene": gene,
                "Rank_NIBFS": float(rank_lookup.loc[gene]),
                "Pairs": int(n),
                "Training_logFC": training_value,
                "External_paired_mean_difference": mean_delta,
                "External_paired_median_difference": median_delta,
                "External_paired_SD": sd_delta,
                "External_paired_95CI_low": float(ci_low),
                "External_paired_95CI_high": float(ci_high),
                "Paired_t_P_value": float(t_result.pvalue),
                "Wilcoxon_P_value": wilcoxon_p,
                "Training_direction": "Up" if training_value > 0 else "Down",
                "External_direction": "Up" if mean_delta > 0 else "Down",
                "Direction_consistent": bool(
                    np.sign(training_value) == np.sign(mean_delta)
                ),
            }
        )

    direction = pd.DataFrame(rows).sort_values("Rank_NIBFS").reset_index(drop=True)
    direction["Paired_t_BH_FDR"] = _bh_adjust(direction["Paired_t_P_value"])
    direction["Wilcoxon_BH_FDR"] = _bh_adjust(direction["Wilcoxon_P_value"])

    summary = pd.DataFrame(
        [
            {
                "Available_genes": int(len(direction)),
                "Direction_consistent": int(direction["Direction_consistent"].sum()),
                "Direction_discordant": int((~direction["Direction_consistent"]).sum()),
                "Consistency_fraction": float(direction["Direction_consistent"].mean()),
                "Genes_paired_t_FDR_lt_0.05": int(
                    (direction["Paired_t_BH_FDR"] < 0.05).sum()
                ),
                "Genes_Wilcoxon_FDR_lt_0.05": int(
                    (direction["Wilcoxon_BH_FDR"] < 0.05).sum()
                ),
            }
        ]
    )
    return direction, summary


# -----------------------------------------------------------------------------
# 6. Publication figure
# -----------------------------------------------------------------------------


def _make_figure(
    predictions: pd.DataFrame,
    direction: pd.DataFrame,
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))

    # A. ROC curves
    ax = axes[0]
    for classifier in MODEL_NAMES:
        group = predictions[predictions["Classifier"] == classifier]
        fpr, tpr_values, _ = roc_curve(group["True_Label"], group["Probability"])
        auc = roc_auc_score(group["True_Label"], group["Probability"])
        ax.plot(fpr, tpr_values, linewidth=2, label=f"{classifier} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title("A. Frozen-model ROC")
    ax.legend(loc="lower right", fontsize=8)

    # B. Precision-recall curves
    ax = axes[1]
    for classifier in MODEL_NAMES:
        group = predictions[predictions["Classifier"] == classifier]
        precision, recall, _ = precision_recall_curve(
            group["True_Label"], group["Probability"]
        )
        ap = average_precision_score(group["True_Label"], group["Probability"])
        ax.plot(recall, precision, linewidth=2, label=f"{classifier} (AP={ap:.3f})")
    prevalence = float(predictions.drop_duplicates("Sample_ID")["True_Label"].mean())
    ax.axhline(prevalence, linestyle="--", linewidth=1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("B. Frozen-model precision–recall")
    ax.legend(loc="lower left", fontsize=8)

    # C. Paired expression differences for frozen genes
    ax = axes[2]
    plot_data = direction.sort_values(
        "External_paired_mean_difference", ascending=True
    ).reset_index(drop=True)
    y_positions = np.arange(len(plot_data))
    x = plot_data["External_paired_mean_difference"].to_numpy(float)
    lower = x - plot_data["External_paired_95CI_low"].to_numpy(float)
    upper = plot_data["External_paired_95CI_high"].to_numpy(float) - x
    ax.errorbar(
        x,
        y_positions,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        capsize=2,
        linewidth=1,
    )
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_data["Gene"], fontsize=8)
    ax.set_xlabel("Cancer − adjacent-normal expression")
    consistent = int(direction["Direction_consistent"].sum())
    ax.set_title(f"C. Paired direction consistency ({consistent}/{len(direction)})")

    fig.suptitle(
        "Independent cross-platform validation in GSE70947 "
        "(148 matched breast-tissue pairs)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    pdf_path = FIGURE_DIR / "Figure_External_Validation_GSE70947.pdf"
    png_path = FIGURE_DIR / "Figure_External_Validation_GSE70947.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def _display_png(path: Path) -> None:
    if not DISPLAY_FIGURE:
        return
    try:
        from IPython.display import Image, Markdown, display

        display(Markdown("## Independent external validation 2 — GSE70947"))
        display(Image(filename=str(path), width=1500))
    except Exception as exc:
        warnings.warn(f"Gambar tersimpan tetapi tidak dapat ditampilkan: {exc}")


# -----------------------------------------------------------------------------
# 7. Main execution
# -----------------------------------------------------------------------------


def main() -> None:
    expected_outputs = [
        OUTPUT_DIR / "external_GSE70947_metrics_default_and_transferred.csv",
        OUTPUT_DIR / "external_GSE70947_pair_bootstrap_95CI.csv",
        OUTPUT_DIR / "external_GSE70947_predictions.csv",
        OUTPUT_DIR / "external_GSE70947_direction_consistency.csv",
        FIGURE_DIR / "Figure_External_Validation_GSE70947.png",
        COMPLETION_MARKER,
    ]
    if not FORCE_RERUN and all(path.exists() for path in expected_outputs):
        print("GSE70947 external validation already complete — existing outputs retained.")
        print("Output:", OUTPUT_DIR)
        _display_png(FIGURE_DIR / "Figure_External_Validation_GSE70947.png")
        return

    print("=" * 88)
    print("INDEPENDENT EXTERNAL VALIDATION 2 — GSE70947 / GPL13607")
    print("=" * 88)
    print("PROJECT_DIR       :", PROJECT_DIR)
    print("MAIN_RUN_DIR      :", MAIN_RUN_DIR)
    print("PANEL_FILE        :", PANEL_FILE)
    print("THRESHOLD_FILE    :", THRESHOLD_FILE)
    print("OUTPUT_DIR        :", OUTPUT_DIR)
    print("Bootstrap pairs   :", BOOTSTRAP_ITERATIONS)

    panel, frozen_genes = _load_frozen_panel()
    thresholds = _load_thresholds()
    models, feature_orders = _load_models(frozen_genes)

    series_path, annotation_path = _resolve_or_download_geo_files()
    resolver_object, hgnc_path = _get_hgnc_resolver()

    print("Series matrix     :", series_path)
    print("Platform annot.   :", annotation_path)
    print("HGNC reference    :", hgnc_path)

    expression, metadata_dictionary = parse_series_matrix(series_path)
    metadata = _label_and_pair_samples(
        create_sample_metadata(metadata_dictionary)
    )

    sample_ids = [sample for sample in metadata["GSM_ID"] if sample in expression.columns]
    if len(sample_ids) != len(metadata):
        missing_samples = sorted(set(metadata["GSM_ID"]) - set(sample_ids))
        raise RuntimeError(
            "Sample metadata tidak seluruhnya tersedia di expression matrix: "
            + ", ".join(missing_samples[:20])
        )
    expression = expression[["ID_REF"] + sample_ids].copy()

    expression, log2_applied = conditional_log2_probe_table(expression, threshold=100.0)
    if log2_applied:
        raise RuntimeError(
            "GSE70947 Series Matrix seharusnya sudah berupa log2 quantile-normalized "
            "signal, tetapi rule mendeteksi skala mentah. Pemeriksaan manual diperlukan."
        )

    probe_map = _read_gpl13607_annotation(
        annotation_path, resolver_object, hgnc_path
    )
    X_all, selected_probe_mapping, preprocessing_audit = (
        _coverage_then_variance_probe_mapping(
            expression,
            probe_map,
            sample_ids,
        )
    )

    metadata = metadata.set_index("GSM_ID").loc[X_all.index].reset_index()

    availability = pd.DataFrame(
        {
            "Gene": frozen_genes,
            "Available_in_GSE70947": [gene in X_all.columns for gene in frozen_genes],
        }
    )
    missing_genes = availability.loc[
        ~availability["Available_in_GSE70947"], "Gene"
    ].tolist()
    if missing_genes:
        raise RuntimeError(
            "Frozen top-20 tidak lengkap pada GPL13607: " + ", ".join(missing_genes)
        )

    X_panel = X_all[frozen_genes].copy()
    if X_panel.shape != (EXPECTED_SAMPLES, FINAL_K):
        raise RuntimeError(
            f"Dimensi panel eksternal tidak sesuai: {X_panel.shape}; "
            f"expected {(EXPECTED_SAMPLES, FINAL_K)}."
        )

    predictions = _predict(X_panel, metadata, models, feature_orders)
    metrics, bootstrap_ci = _evaluate_predictions(predictions, thresholds)
    direction, direction_summary = _direction_consistency(X_panel, metadata, panel)
    pdf_path, png_path = _make_figure(predictions, direction)

    # Save primary outputs.
    metadata.to_csv(OUTPUT_DIR / "external_GSE70947_sample_manifest.csv", index=False)
    selected_probe_mapping.to_csv(
        OUTPUT_DIR / "external_GSE70947_selected_probe_mapping.csv", index=False
    )
    availability.to_csv(
        OUTPUT_DIR / "external_GSE70947_frozen_panel_coverage.csv", index=False
    )
    X_panel.to_csv(
        OUTPUT_DIR / "external_GSE70947_frozen_panel_expression.csv.gz",
        compression="gzip",
        index=True,
        index_label="GSM_ID",
    )
    thresholds.to_csv(
        OUTPUT_DIR / "external_GSE70947_oof_transferred_thresholds.csv", index=False
    )
    predictions.to_csv(OUTPUT_DIR / "external_GSE70947_predictions.csv", index=False)
    metrics.to_csv(
        OUTPUT_DIR / "external_GSE70947_metrics_default_and_transferred.csv",
        index=False,
    )
    bootstrap_ci.to_csv(
        OUTPUT_DIR / "external_GSE70947_pair_bootstrap_95CI.csv", index=False
    )
    direction.to_csv(
        OUTPUT_DIR / "external_GSE70947_direction_consistency.csv", index=False
    )
    direction_summary.to_csv(
        OUTPUT_DIR / "external_GSE70947_direction_consistency_summary.csv",
        index=False,
    )

    preprocessing_table = pd.DataFrame(
        [
            {
                "Dataset": GSE,
                "Platform": GPL,
                "Samples": int(len(metadata)),
                "Pairs": int(metadata["Pair_ID"].nunique()),
                "Normal": int((metadata["Label_binary"] == 0).sum()),
                "Cancer": int((metadata["Label_binary"] == 1).sum()),
                "Series_matrix_log2_applied_by_script": bool(log2_applied),
                **preprocessing_audit,
            }
        ]
    )
    preprocessing_table.to_csv(
        OUTPUT_DIR / "external_GSE70947_preprocessing_audit.csv", index=False
    )

    no_leakage_audit = pd.DataFrame(
        [
            {"Check": "Frozen panel imported from main run", "Passed": True},
            {"Check": "Frozen model files loaded; no model fit called", "Passed": True},
            {"Check": "OOF thresholds imported from discovery", "Passed": True},
            {"Check": "No external feature selection", "Passed": True},
            {"Check": "No external hyperparameter tuning", "Passed": True},
            {"Check": "Probe selection did not use labels", "Passed": True},
            {"Check": "Imputation did not use labels", "Passed": True},
            {"Check": "Normalization did not use labels", "Passed": True},
            {"Check": "Complete frozen top-20 coverage", "Passed": not missing_genes},
            {"Check": "148 complete matched pairs", "Passed": int(metadata["Pair_ID"].nunique()) == EXPECTED_PAIRS},
        ]
    )
    no_leakage_audit.to_csv(
        OUTPUT_DIR / "external_GSE70947_no_leakage_audit.csv", index=False
    )
    if not no_leakage_audit["Passed"].all():
        raise RuntimeError("No-leakage audit failed.")

    source_files = {
        "series_matrix": series_path,
        "platform_annotation": annotation_path,
        "hgnc_reference": hgnc_path,
        "frozen_panel": PANEL_FILE,
        "oof_thresholds": THRESHOLD_FILE,
        **{f"frozen_model_{name}": path for name, path in MODEL_FILES.items()},
    }
    provenance_rows = []
    for role, path in source_files.items():
        provenance_rows.append(
            {
                "Role": role,
                "Path": str(path),
                "Size_bytes": int(path.stat().st_size),
                "SHA256": _sha256(path),
            }
        )
    pd.DataFrame(provenance_rows).to_csv(
        OUTPUT_DIR / "external_GSE70947_source_manifest.csv", index=False
    )

    run_record = {
        "analysis": "Independent external validation 2",
        "script_version": "V5",
        "annotation_strategy": "GPL13607 GeneName + RefSeq + HGNC; official GEO supplementary annotation when available",
        "dataset": GSE,
        "platform": GPL,
        "completed_utc": _now_utc(),
        "project_dir": str(PROJECT_DIR),
        "main_run_dir": str(MAIN_RUN_DIR),
        "output_dir": str(OUTPUT_DIR),
        "frozen_k": FINAL_K,
        "models": list(MODEL_NAMES),
        "sample_count": int(len(metadata)),
        "pair_count": int(metadata["Pair_ID"].nunique()),
        "normal_count": int((metadata["Label_binary"] == 0).sum()),
        "cancer_count": int((metadata["Label_binary"] == 1).sum()),
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "frozen_panel_complete": bool(availability["Available_in_GSE70947"].all()),
        "labels_used_in_preprocessing": False,
        "feature_reselection": False,
        "model_refitting": False,
        "threshold_refitting": False,
        "figure_pdf": str(pdf_path),
        "figure_png": str(png_path),
    }
    with (OUTPUT_DIR / "external_GSE70947_RUN_RECORD.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(run_record, handle, indent=2)

    completion_text = (
        "EXTERNAL VALIDATION GSE70947 COMPLETE\n"
        f"Completed UTC: {run_record['completed_utc']}\n"
        f"Samples: {run_record['sample_count']}\n"
        f"Pairs: {run_record['pair_count']}\n"
        f"Frozen genes: {FINAL_K}/{FINAL_K}\n"
        "Feature reselection: no\n"
        "Model refitting: no\n"
        "Threshold refitting: no\n"
    )
    COMPLETION_MARKER.write_text(completion_text, encoding="utf-8")

    print("\nINTEGRITY CHECK PASSED")
    print("Samples              :", len(metadata))
    print("Matched pairs        :", metadata["Pair_ID"].nunique())
    print("Normal / cancer      :", (metadata["Label_binary"] == 0).sum(), "/", (metadata["Label_binary"] == 1).sum())
    print("Frozen-panel coverage:", int(availability["Available_in_GSE70947"].sum()), "/", FINAL_K)
    print("Frozen models loaded :", ", ".join(MODEL_NAMES))
    print("Feature reselection  : NO")
    print("Model refitting      : NO")
    print("Threshold refitting  : NO")
    print("\nMetrics:")
    print(
        metrics[
            [
                "Classifier",
                "Evaluation_rule",
                "Threshold",
                "ROC_AUC",
                "PR_AUC",
                "Accuracy",
                "F1",
                "MCC",
                "Brier_score",
            ]
        ].to_string(index=False)
    )
    print("\nDirection consistency:")
    print(direction_summary.to_string(index=False))
    print("\nSELESAI. Output:", OUTPUT_DIR)

    _display_png(png_path)


main()
