
"""Independent TCGA-BRCA RNA-seq validation for the frozen NIBFS top-20 panel.

Scientific design
-----------------
External cohort:
    TCGA-BRCA paired Primary Tumor and Solid Tissue Normal RNA-seq samples
    retrieved from the NCI Genomic Data Commons (GDC), workflow STAR - Counts.

Independence:
    No TCGA labels are used for feature selection, hyperparameter tuning,
    preprocessing fitting, or classifier fitting. External labels are used only
    after predictions are locked, for evaluation and paired biological
    replication.

Frozen components:
    * exact NIBFS top-20 genes and their discovery logFC directions;
    * 608-sample development partition;
    * LR, RF, and LightGBM algorithm families and fixed hyperparameters;
    * 0.5 probability threshold;
    * signed-panel score definition.

Cross-technology bridge:
    Microarray and RNA-seq values are each converted to within-sample percentile
    ranks across the shared gene universe. This monotone, label-free
    representation avoids direct comparison of microarray intensities and
    RNA-seq abundance units. The signed-panel score is fully frozen and requires
    no classifier fitting. Transfer classifiers are fitted only on the 608
    microarray development samples using the frozen top-20 panel and the same
    rank representation, then applied once to TCGA-BRCA.

This analysis validates panel replication and cross-technology transferability.
It must not be described as application of an unchanged microarray-scale model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import md5, sha256
from io import StringIO
from pathlib import Path
from typing import Any, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib
import json
import math
import os
import platform
import re
import shutil
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from scipy.stats import binomtest, wilcoxon
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


EXPECTED_TOP20 = [
    "CDK1", "EGFR", "CCNB1", "BUB1B", "FN1",
    "CDC20", "EZH2", "STAT1", "TOP2A", "CAV1",
    "RRM2", "GNAI1", "KIT", "PPARG", "CCNA2",
    "UBE2C", "FGF2", "CCNB2", "MAD2L1", "FOXO1",
]

EXPECTED_LOGFC = {
    "CDK1": 2.590235,
    "EGFR": -2.506577,
    "CCNB1": 2.356355,
    "BUB1B": 2.146463,
    "FN1": 1.817686,
    "CDC20": 2.094132,
    "EZH2": 1.865817,
    "STAT1": 1.854808,
    "TOP2A": 3.243555,
    "CAV1": -2.404419,
    "RRM2": 3.457499,
    "GNAI1": -1.947952,
    "KIT": -2.367023,
    "PPARG": -2.069141,
    "CCNA2": 1.531200,
    "UBE2C": 2.208105,
    "FGF2": -2.003222,
    "CCNB2": 1.826779,
    "MAD2L1": 1.964061,
    "FOXO1": -1.510347,
}

GDC_API_BASE = "https://api.gdc.cancer.gov"


@dataclass(frozen=True)
class RNASeqValidationConfig:
    project_id: str = "TCGA-BRCA"
    tumor_sample_type: str = "Primary Tumor"
    normal_sample_type: str = "Solid Tissue Normal"
    workflow_type: str = "STAR - Counts"
    abundance_column: str = "tpm_unstranded"
    expected_development_samples: int = 608
    expected_panel_size: int = 20
    minimum_pairs: int = 50
    bootstrap_replicates: int = 2000
    random_state: int = 42
    download_workers: int = 8
    parse_workers: int = 4
    request_timeout_seconds: int = 240
    request_retries: int = 5
    require_complete_panel: bool = True
    resume: bool = True
    force_rerun: bool = False
    smoke_test_pairs: int | None = None

    def validate(self) -> None:
        if self.project_id != "TCGA-BRCA":
            raise ValueError("This package is locked to TCGA-BRCA.")
        if self.expected_development_samples != 608:
            raise ValueError("This package is locked to 608 development samples.")
        if self.expected_panel_size != 20:
            raise ValueError("This package is locked to the frozen top-20 panel.")
        if self.bootstrap_replicates < 200:
            raise ValueError("Use at least 200 bootstrap replicates.")
        if self.minimum_pairs < 10:
            raise ValueError("minimum_pairs is implausibly small.")
        if self.download_workers < 1 or self.parse_workers < 1:
            raise ValueError("Worker counts must be positive.")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def _find_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    lookup = {
        re.sub(r"[^a-z0-9]+", "", str(column).casefold()): str(column)
        for column in frame.columns
    }
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]+", "", candidate.casefold())
        if key in lookup:
            return lookup[key]
    return None


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = md5()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _resolve_project_dir(namespace: Mapping[str, Any]) -> Path:
    for name in ("RNASEQ_PROJECT_DIR", "PROJECT_DIR", "PACKAGE_DIR"):
        value = namespace.get(name)
        if value:
            path = Path(str(value)).expanduser().resolve()
            if (path / "src" / "strict_engine.py").is_file():
                return path

    for root in (
        Path("/content/drive/MyDrive"),
    ):
        if not root.exists():
            continue
        matches = list(root.rglob("NIBFS_CBC_TOTAL_RERUN_STRICT_v10.4*/src/strict_engine.py"))
        if matches:
            return matches[0].parent.parent.resolve()

    raise FileNotFoundError(
        "Strict NIBFS project was not found. Set RNASEQ_PROJECT_DIR to the "
        "project root containing src/strict_engine.py."
    )


def _resolve_output_dir(namespace: Mapping[str, Any], project_dir: Path) -> Path:
    override = namespace.get("RNASEQ_OUTPUT_DIR")
    if override:
        output = Path(str(override)).expanduser().resolve()
    else:
        output = (
            project_dir
            / "results"
            / "TCGA_BRCA_RNASEQ_EXTERNAL_VALIDATION"
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def _resolve_local_cache_dir(namespace: Mapping[str, Any]) -> Path:
    override = namespace.get("RNASEQ_LOCAL_CACHE_DIR")
    cache = (
        Path(str(override)).expanduser().resolve()
        if override
        else Path("/content/TCGA_BRCA_RNASEQ_PAIRED_CACHE")
    )
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _load_cfg(namespace: Mapping[str, Any], project_dir: Path) -> dict:
    cfg = namespace.get("cfg")
    if isinstance(cfg, dict):
        return cfg

    candidates = [
        project_dir / "config.yaml",
        project_dir / "configs" / "config_total_rerun.yaml",
        project_dir / "configs" / "config.yaml",
    ]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return {}

    import yaml
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return loaded if isinstance(loaded, dict) else {}


def _resolve_store(
    namespace: Mapping[str, Any],
    project_dir: Path,
) -> Any:
    """Use an active store or reconstruct it from the saved raw-probe cache."""
    for name in ("store", "raw_store", "discovery_store", "RAW_STORE"):
        candidate = namespace.get(name)
        if candidate is not None and (
            hasattr(candidate, "metadata")
            or hasattr(candidate, "sample_metadata")
        ):
            return candidate

    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))

    try:
        from src.raw_discovery import RawDiscoveryStore
    except Exception as exc:
        raise RuntimeError(
            "Could not import RawDiscoveryStore from the saved strict project."
        ) from exc

    cache_dir = project_dir / "data" / "cache" / "raw_probe"
    required = [
        cache_dir / "discovery_raw_metadata.csv",
        cache_dir / "GPL570_probe_to_HGNC_mapping.csv",
        cache_dir / "raw_probe_cache_manifest.csv",
        cache_dir / "fixed_structural_gene_universe.csv",
        cache_dir / "structural_gene_presence_by_cohort.csv",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The saved raw-probe cache is incomplete. Missing:\n- "
            + "\n- ".join(map(str, missing))
        )

    store = RawDiscoveryStore.load(project_dir)

    missing_matrices = [
        store.matrix_path(gse)
        for gse in store.cohorts
        if not store.matrix_path(gse).is_file()
    ]
    if missing_matrices:
        raise FileNotFoundError(
            "Cached cohort matrices are incomplete. Missing:\n- "
            + "\n- ".join(map(str, missing_matrices[:20]))
        )

    return store


def _normalize_fold_assignments(frame: pd.DataFrame) -> pd.DataFrame:
    sample_col = _find_column(frame, ("GSM_ID", "Sample_ID", "Sample", "sample_id"))
    fold_col = _find_column(frame, ("Fold", "Fold_ID", "Validation_fold", "CV_fold"))
    set_col = _find_column(frame, ("Set", "Subset", "Role", "Split"))

    if sample_col is None or fold_col is None:
        raise KeyError("Fold table requires sample and fold columns.")

    out = frame.copy()
    if set_col is not None:
        set_text = out[set_col].astype(str).map(_normalize_text)
        development = set_text.str.contains(
            r"development|model development|train|validation",
            regex=True,
        )
        if development.any():
            out = out.loc[development].copy()

    out = out[[sample_col, fold_col]].copy()
    out.columns = ["GSM_ID", "Fold"]
    out["GSM_ID"] = out["GSM_ID"].astype(str).str.strip()
    out = out[out["GSM_ID"].ne("")].drop_duplicates("GSM_ID")

    numeric = pd.to_numeric(out["Fold"], errors="coerce")
    if numeric.notna().all():
        out["Fold"] = numeric.astype(int)
    else:
        extracted = out["Fold"].astype(str).str.extract(r"(\d+)", expand=False)
        if extracted.isna().any():
            raise ValueError("Fold identifiers cannot be parsed.")
        out["Fold"] = extracted.astype(int)

    unique = sorted(out["Fold"].unique())
    remap = {old: i + 1 for i, old in enumerate(unique)}
    out["Fold"] = out["Fold"].map(remap).astype(int)
    return out.sort_values(["Fold", "GSM_ID"]).reset_index(drop=True)


def _resolve_development_ids(
    namespace: Mapping[str, Any],
    project_dir: Path,
    expected_n: int,
) -> tuple[pd.DataFrame, str]:
    for name in (
        "fold_assignments",
        "primary_fold_assignments",
        "cv_fold_assignments",
        "development_fold_assignments",
    ):
        value = namespace.get(name)
        if isinstance(value, pd.DataFrame) and not value.empty:
            try:
                normalized = _normalize_fold_assignments(value)
            except Exception:
                continue
            if (
                normalized["GSM_ID"].nunique() == expected_n
                and normalized["Fold"].nunique() == 5
            ):
                return normalized, f"active notebook object `{name}`"

    candidates: list[Path] = []
    for pattern in (
        "fold_assignments.csv",
        "*primary*fold*assignments*.csv",
        "*development*fold*assignments*.csv",
    ):
        candidates.extend(path for path in project_dir.rglob(pattern) if path.is_file())

    ranked = []
    for path in candidates:
        lower = str(path).casefold()
        if any(token in lower for token in ("repeated", "loco", "external", "permutation")):
            continue
        try:
            table = _normalize_fold_assignments(pd.read_csv(path))
        except Exception:
            continue
        score = int(table["GSM_ID"].nunique() == expected_n) * 10
        score += int(table["Fold"].nunique() == 5) * 10
        score += int("main" in lower or "primary" in lower)
        ranked.append((score, path.stat().st_mtime, path, table))

    if not ranked:
        raise FileNotFoundError("The original 608-sample five-fold table was not found.")

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _, path, table = ranked[0]
    if table["GSM_ID"].nunique() != expected_n or table["Fold"].nunique() != 5:
        raise RuntimeError(f"Selected development fold table is invalid: {path}")
    return table, str(path)


def _resolve_sample_labels(store: Any, development_ids: Sequence[str]) -> pd.Series:
    """Read labels from the saved store without rerunning the main notebook."""
    sample_metadata = getattr(store, "sample_metadata", None)

    if callable(sample_metadata):
        metadata = pd.DataFrame(
            sample_metadata(list(map(str, development_ids)))
        ).copy()
    elif hasattr(store, "metadata"):
        metadata = pd.DataFrame(store.metadata).copy()
    elif sample_metadata is not None:
        metadata = pd.DataFrame(sample_metadata).copy()
    else:
        raise AttributeError(
            "The loaded raw discovery store exposes neither metadata nor "
            "sample_metadata."
        )

    sample_col = _find_column(
        metadata,
        ("GSM_ID", "sample_id", "Sample_ID", "sample", "geo_accession"),
    )
    label_col = _find_column(
        metadata,
        ("Label", "Class", "Outcome", "Diagnosis", "y"),
    )
    if sample_col is None:
        if metadata.index.is_unique:
            metadata = metadata.reset_index().rename(columns={"index": "GSM_ID"})
            sample_col = "GSM_ID"
        else:
            raise KeyError("Sample ID column was not found in saved metadata.")
    if label_col is None:
        raise KeyError("Label column was not found in saved metadata.")

    metadata[sample_col] = metadata[sample_col].astype(str).str.strip()
    metadata = metadata.drop_duplicates(sample_col).set_index(sample_col)

    def encode(value: Any) -> int:
        text = _normalize_text(value)
        if text in {"1", "cancer", "tumor", "tumour", "breast cancer", "case"}:
            return 1
        if text in {"0", "normal", "control", "healthy", "adjacent normal"}:
            return 0
        raise ValueError(f"Unrecognized discovery label: {value!r}")

    missing = sorted(set(map(str, development_ids)) - set(metadata.index.astype(str)))
    if missing:
        raise KeyError("Development labels missing for: " + ", ".join(missing[:10]))

    labels = metadata.loc[list(map(str, development_ids)), label_col].map(encode)
    labels.name = "Label"
    return labels.astype(int)


def _resolve_panel(
    namespace: Mapping[str, Any],
    project_dir: Path,
) -> tuple[pd.DataFrame, str]:
    candidate_frames = []
    for name in (
        "final_nibfs_panel_k20",
        "final_panel_k20",
        "frozen_top20",
        "final_panel",
    ):
        value = namespace.get(name)
        if isinstance(value, pd.DataFrame) and not value.empty:
            candidate_frames.append((value.copy(), f"active notebook object `{name}`"))

    patterns = (
        "final_NIBFS_gene_panel_k20.csv",
        "final_feature_panels_full_development_all_k.csv",
        "*final*NIBFS*panel*k20*.csv",
        "*final*feature*panels*all*k*.csv",
    )
    for pattern in patterns:
        for path in project_dir.rglob(pattern):
            if path.is_file():
                try:
                    candidate_frames.append((pd.read_csv(path), str(path)))
                except Exception:
                    pass

    for frame, source in candidate_frames:
        gene_col = _find_column(frame, ("Gene", "gene_symbol", "Symbol"))
        if gene_col is None:
            continue

        method_col = _find_column(frame, ("Method", "Feature_selection"))
        k_col = _find_column(frame, ("k", "Panel_size"))
        rank_col = _find_column(frame, ("Rank_NIBFS", "Selection_rank", "Rank"))
        logfc_col = _find_column(frame, ("logFC", "log2FC", "Log2_Fold_Change"))

        subset = frame.copy()
        if method_col is not None:
            subset = subset[
                subset[method_col].astype(str).map(_normalize_text).eq("nibfs")
            ].copy()
        if k_col is not None:
            numeric_k = pd.to_numeric(subset[k_col], errors="coerce")
            if (numeric_k == 20).any():
                subset = subset[numeric_k == 20].copy()
        if rank_col is not None:
            subset = subset.sort_values(rank_col)

        subset[gene_col] = subset[gene_col].astype(str).str.strip().str.upper()
        subset = subset.drop_duplicates(gene_col)
        genes = subset[gene_col].tolist()[:20]
        if len(genes) != 20:
            continue

        if genes != EXPECTED_TOP20:
            raise RuntimeError(
                "The detected frozen top-20 differs from the locked manuscript panel.\n"
                f"Detected: {genes}\nExpected: {EXPECTED_TOP20}\nSource: {source}"
            )

        out = pd.DataFrame(
            {
                "Rank_NIBFS": np.arange(1, 21),
                "Gene": genes,
            }
        )
        if logfc_col is not None:
            values = (
                subset.set_index(gene_col)[logfc_col]
                .reindex(genes)
                .pipe(pd.to_numeric, errors="coerce")
            )
            out["Discovery_logFC"] = values.to_numpy()
        else:
            out["Discovery_logFC"] = [EXPECTED_LOGFC[gene] for gene in genes]

        if out["Discovery_logFC"].isna().any():
            out["Discovery_logFC"] = [
                EXPECTED_LOGFC[gene] for gene in out["Gene"]
            ]
        out["Discovery_direction"] = np.where(
            out["Discovery_logFC"] > 0,
            "Up_in_cancer",
            "Down_in_cancer",
        )
        return out, source

    raise FileNotFoundError(
        "The exact frozen NIBFS top-20 table was not found in the strict project."
    )


def _build_development_gene_matrix(
    *,
    project_dir: Path,
    store: Any,
    cfg: dict,
    development_ids: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
    strict_engine = importlib.import_module("src.strict_engine")

    threshold = float(
        cfg.get("preprocessing", {}).get("log2_threshold", 100.0)
    )
    mapper = strict_engine.FoldGeneMapper(
        store,
        log2_threshold=threshold,
        selection_scope="per_cohort",
    )
    mapped = mapper.fit_transform(
        list(map(str, development_ids)),
        list(map(str, development_ids)),
    )
    matrix = mapped.X_train_raw.copy()
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str).str.upper()
    matrix = matrix.loc[list(map(str, development_ids))]
    matrix = matrix.loc[:, ~matrix.columns.duplicated()].copy()

    return matrix, mapped.selected_probes.copy(), mapped.log2_audit.copy()


def _post_json(
    endpoint: str,
    payload: dict,
    *,
    timeout: int,
    retries: int,
) -> dict:
    url = f"{GDC_API_BASE}/{endpoint.lstrip('/')}"
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"GDC API request failed after {retries} attempts: {url}") from last_error


def _query_tcga_brca_samples(config: RNASeqValidationConfig) -> pd.DataFrame:
    payload = {
        "filters": {
            "op": "in",
            "content": {
                "field": "project.project_id",
                "value": [config.project_id],
            },
        },
        "format": "JSON",
        "fields": ",".join(
            [
                "case_id",
                "submitter_id",
                "samples.sample_id",
                "samples.submitter_id",
                "samples.sample_type",
                "samples.tissue_type",
            ]
        ),
        "size": "2000",
    }
    response = _post_json(
        "cases",
        payload,
        timeout=config.request_timeout_seconds,
        retries=config.request_retries,
    )
    rows = []
    for case in response.get("data", {}).get("hits", []):
        for sample in case.get("samples", []) or []:
            rows.append(
                {
                    "case_id": str(case.get("case_id", "")),
                    "case_submitter_id": str(case.get("submitter_id", "")),
                    "sample_id": str(sample.get("sample_id", "")),
                    "sample_submitter_id": str(sample.get("submitter_id", "")),
                    "sample_type": str(sample.get("sample_type", "")),
                    "tissue_type": str(sample.get("tissue_type", "")),
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        raise RuntimeError("GDC cases query returned no TCGA-BRCA samples.")

    table = table[
        table["sample_type"].isin(
            [config.tumor_sample_type, config.normal_sample_type]
        )
    ].copy()
    table = table[
        table["case_id"].ne("") & table["sample_id"].ne("")
    ].drop_duplicates(["case_id", "sample_id"])

    # Deterministic one sample per case and sample type.
    table = table.sort_values(
        ["case_submitter_id", "sample_type", "sample_submitter_id", "sample_id"]
    )
    selected = (
        table.groupby(["case_id", "sample_type"], as_index=False)
        .first()
    )
    counts = selected.groupby("case_id")["sample_type"].nunique()
    paired_case_ids = counts[counts == 2].index
    selected = selected[selected["case_id"].isin(paired_case_ids)].copy()
    selected["Label"] = np.where(
        selected["sample_type"].eq(config.tumor_sample_type),
        1,
        0,
    )
    selected["Pair_ID"] = selected["case_submitter_id"]
    return selected.sort_values(["Pair_ID", "Label"]).reset_index(drop=True)


def _query_star_count_files(
    selected_samples: pd.DataFrame,
    config: RNASeqValidationConfig,
) -> pd.DataFrame:
    sample_ids = selected_samples["sample_id"].astype(str).tolist()
    payload = {
        "filters": {
            "op": "and",
            "content": [
                {
                    "op": "in",
                    "content": {
                        "field": "cases.samples.sample_id",
                        "value": sample_ids,
                    },
                },
                {
                    "op": "in",
                    "content": {
                        "field": "files.data_category",
                        "value": ["Transcriptome Profiling"],
                    },
                },
                {
                    "op": "in",
                    "content": {
                        "field": "files.data_type",
                        "value": ["Gene Expression Quantification"],
                    },
                },
                {
                    "op": "in",
                    "content": {
                        "field": "analysis.workflow_type",
                        "value": [config.workflow_type],
                    },
                },
                {
                    "op": "in",
                    "content": {
                        "field": "files.access",
                        "value": ["open"],
                    },
                },
                {
                    "op": "in",
                    "content": {
                        "field": "files.is_latest",
                        "value": [True],
                    },
                },
            ],
        },
        "format": "JSON",
        "fields": ",".join(
            [
                "file_id",
                "file_name",
                "md5sum",
                "file_size",
                "created_datetime",
                "updated_datetime",
                "analysis.workflow_type",
                "cases.case_id",
                "cases.submitter_id",
                "cases.samples.sample_id",
                "cases.samples.submitter_id",
                "cases.samples.sample_type",
            ]
        ),
        "size": "5000",
    }
    response = _post_json(
        "files",
        payload,
        timeout=config.request_timeout_seconds,
        retries=config.request_retries,
    )

    selected_ids = set(sample_ids)
    rows = []
    for hit in response.get("data", {}).get("hits", []):
        linked_sample_ids = []
        for case in hit.get("cases", []) or []:
            for sample in case.get("samples", []) or []:
                sample_id = str(sample.get("sample_id", ""))
                if sample_id in selected_ids:
                    linked_sample_ids.append(sample_id)
        linked_sample_ids = sorted(set(linked_sample_ids))
        for sample_id in linked_sample_ids:
            rows.append(
                {
                    "sample_id": sample_id,
                    "file_id": str(hit.get("file_id", "")),
                    "file_name": str(hit.get("file_name", "")),
                    "md5sum": str(hit.get("md5sum", "")),
                    "file_size": hit.get("file_size"),
                    "created_datetime": hit.get("created_datetime"),
                    "updated_datetime": hit.get("updated_datetime"),
                    "workflow_type": (
                        hit.get("analysis", {}) or {}
                    ).get("workflow_type"),
                }
            )

    files = pd.DataFrame(rows)
    if files.empty:
        raise RuntimeError("No open STAR-Counts files were linked to selected samples.")

    files = files.sort_values(
        ["sample_id", "updated_datetime", "file_id"],
        ascending=[True, False, True],
    ).drop_duplicates("sample_id")

    merged = selected_samples.merge(files, on="sample_id", how="left")
    complete_case = (
        merged.groupby("Pair_ID")["file_id"]
        .apply(lambda series: series.notna().sum() == 2)
    )
    complete_ids = complete_case[complete_case].index
    merged = merged[merged["Pair_ID"].isin(complete_ids)].copy()
    merged = merged[merged["file_id"].notna()].copy()

    if config.smoke_test_pairs is not None:
        keep_pairs = sorted(merged["Pair_ID"].unique())[: int(config.smoke_test_pairs)]
        merged = merged[merged["Pair_ID"].isin(keep_pairs)].copy()

    n_pairs = merged["Pair_ID"].nunique()
    required = (
        min(config.minimum_pairs, int(config.smoke_test_pairs))
        if config.smoke_test_pairs is not None
        else config.minimum_pairs
    )
    if n_pairs < required:
        raise RuntimeError(
            f"Only {n_pairs} complete paired cases were found; required at least {required}."
        )

    counts = merged.groupby("Pair_ID")["Label"].agg(["count", "nunique"])
    if not ((counts["count"] == 2) & (counts["nunique"] == 2)).all():
        raise RuntimeError("Pair selection did not produce one tumor and one normal per case.")

    return merged.sort_values(["Pair_ID", "Label"]).reset_index(drop=True)


def _download_one_file(
    row: Mapping[str, Any],
    destination_dir: Path,
    config: RNASeqValidationConfig,
) -> dict:
    file_id = str(row["file_id"])
    file_name = str(row["file_name"]) or f"{file_id}.tsv"
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", file_name)
    destination = destination_dir / f"{file_id}__{safe_name}"
    expected_md5 = str(row.get("md5sum", "") or "").lower()

    if destination.is_file():
        if not expected_md5 or _md5_file(destination) == expected_md5:
            return {
                "file_id": file_id,
                "local_path": str(destination),
                "download_status": "cached",
                "bytes": destination.stat().st_size,
                "md5_verified": bool(expected_md5),
            }
        destination.unlink()

    url = f"{GDC_API_BASE}/data/{file_id}"
    last_error: Exception | None = None
    for attempt in range(config.request_retries):
        partial = destination.with_suffix(destination.suffix + ".partial")
        try:
            with requests.get(
                url,
                stream=True,
                timeout=config.request_timeout_seconds,
            ) as response:
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            partial.replace(destination)
            verified = not expected_md5 or _md5_file(destination) == expected_md5
            if not verified:
                destination.unlink(missing_ok=True)
                raise IOError(f"MD5 mismatch for {file_id}")
            return {
                "file_id": file_id,
                "local_path": str(destination),
                "download_status": "downloaded",
                "bytes": destination.stat().st_size,
                "md5_verified": bool(expected_md5),
            }
        except Exception as exc:
            last_error = exc
            partial.unlink(missing_ok=True)
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"Failed to download GDC file {file_id}") from last_error


def _download_files(
    manifest: pd.DataFrame,
    destination_dir: Path,
    config: RNASeqValidationConfig,
) -> pd.DataFrame:
    destination_dir.mkdir(parents=True, exist_ok=True)
    records = manifest.to_dict("records")
    results = []
    with ThreadPoolExecutor(max_workers=config.download_workers) as executor:
        future_map = {
            executor.submit(_download_one_file, row, destination_dir, config): row
            for row in records
        }
        total = len(future_map)
        completed = 0
        for future in as_completed(future_map):
            completed += 1
            result = future.result()
            results.append(result)
            if completed == 1 or completed % 20 == 0 or completed == total:
                print(f"Downloaded/cached {completed}/{total} STAR-Counts files", flush=True)
    result_frame = pd.DataFrame(results)
    return manifest.merge(result_frame, on="file_id", how="left")


def _parse_star_counts(path: Path, abundance_column: str) -> pd.Series:
    frame = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        low_memory=False,
    )
    gene_name_col = _find_column(frame, ("gene_name", "gene_symbol", "symbol"))
    gene_id_col = _find_column(frame, ("gene_id", "ensembl_gene_id"))
    gene_type_col = _find_column(frame, ("gene_type", "gene_biotype"))
    value_col = _find_column(
        frame,
        (
            abundance_column,
            "tpm_unstranded",
            "fpkm_uq_unstranded",
            "fpkm_unstranded",
            "unstranded",
        ),
    )
    if gene_name_col is None or value_col is None:
        raise KeyError(
            f"Required STAR-Counts columns missing in {path.name}: {list(frame.columns)}"
        )

    if gene_id_col is not None:
        frame = frame[
            frame[gene_id_col].astype(str).str.startswith("ENSG")
        ].copy()
    if gene_type_col is not None:
        protein = frame[gene_type_col].astype(str).eq("protein_coding")
        if protein.any():
            frame = frame[protein].copy()

    frame["Gene"] = frame[gene_name_col].astype(str).str.strip().str.upper()
    frame["Value"] = pd.to_numeric(frame[value_col], errors="coerce")
    frame = frame[
        frame["Gene"].ne("")
        & frame["Gene"].ne("NAN")
        & frame["Value"].notna()
    ].copy()
    series = frame.groupby("Gene", sort=False)["Value"].mean()
    return series


def _build_external_expression(
    downloaded_manifest: pd.DataFrame,
    config: RNASeqValidationConfig,
) -> pd.DataFrame:
    rows = downloaded_manifest.to_dict("records")
    parsed: dict[str, pd.Series] = {}
    with ThreadPoolExecutor(max_workers=config.parse_workers) as executor:
        future_map = {
            executor.submit(
                _parse_star_counts,
                Path(str(row["local_path"])),
                config.abundance_column,
            ): row
            for row in rows
        }
        total = len(future_map)
        completed = 0
        for future in as_completed(future_map):
            completed += 1
            row = future_map[future]
            parsed[str(row["sample_id"])] = future.result()
            if completed == 1 or completed % 20 == 0 or completed == total:
                print(f"Parsed {completed}/{total} STAR-Counts files", flush=True)

    matrix = pd.DataFrame.from_dict(parsed, orient="index").sort_index()
    matrix.index.name = "sample_id"
    matrix = matrix.fillna(0.0)
    matrix = np.log2(matrix + 1.0)
    matrix.columns = matrix.columns.astype(str).str.upper()
    return matrix


def _within_sample_percentile_rank(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.rank(axis=1, method="average", pct=True)
    if ranked.isna().any().any():
        raise RuntimeError("Percentile-rank transformation produced missing values.")
    return ranked.astype(float)


def _create_models(random_state: int) -> dict[str, Any]:
    lr = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=5000,
                    random_state=random_state,
                ),
            ),
        ]
    )
    rf = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError(
            "lightgbm is required. Install it before running this validation."
        ) from exc

    lgbm = LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=0.0,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
        is_unbalance=True,
    )
    return {"LR": lr, "RF": rf, "LightGBM": lgbm}


def _signed_panel_score(
    ranked_top20: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.Series:
    signs = panel.set_index("Gene")["Discovery_logFC"].map(np.sign)
    score = ranked_top20.mul(signs, axis=1).mean(axis=1)
    score.name = "FrozenSignedPanelScore"
    return score


def _safe_metric(metric_name: str, y_true: np.ndarray, score: np.ndarray, threshold: float) -> float:
    pred = (score >= threshold).astype(int)
    if metric_name == "ROC_AUC":
        return float(roc_auc_score(y_true, score))
    if metric_name == "PR_AUC":
        return float(average_precision_score(y_true, score))
    if metric_name == "Accuracy":
        return float(accuracy_score(y_true, pred))
    if metric_name == "Balanced_Accuracy":
        return float(balanced_accuracy_score(y_true, pred))
    if metric_name == "F1":
        return float(f1_score(y_true, pred, zero_division=0))
    if metric_name == "MCC":
        return float(matthews_corrcoef(y_true, pred))
    if metric_name == "Sensitivity":
        return float(recall_score(y_true, pred, pos_label=1, zero_division=0))
    if metric_name == "Specificity":
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        return float(tn / (tn + fp)) if (tn + fp) else float("nan")
    if metric_name == "Precision":
        return float(precision_score(y_true, pred, zero_division=0))
    if metric_name == "Brier":
        clipped = np.clip(score, 0.0, 1.0)
        return float(brier_score_loss(y_true, clipped))
    raise KeyError(metric_name)


def _pair_bootstrap_metrics(
    predictions: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_names = [
        "ROC_AUC",
        "PR_AUC",
        "Accuracy",
        "Balanced_Accuracy",
        "F1",
        "MCC",
        "Sensitivity",
        "Specificity",
        "Precision",
    ]
    pair_ids = predictions["Pair_ID"].drop_duplicates().tolist()
    pair_to_indices = {
        pair: predictions.index[predictions["Pair_ID"].eq(pair)].to_numpy()
        for pair in pair_ids
    }
    rng = np.random.default_rng(random_state)

    summary_rows = []
    bootstrap_rows = []

    for model, block in predictions.groupby("Model", sort=False):
        threshold = float(block["Threshold"].iloc[0])
        y = block["Label"].to_numpy(dtype=int)
        scores = block["Score"].to_numpy(dtype=float)

        point = {
            metric: _safe_metric(metric, y, scores, threshold)
            for metric in metric_names
        }

        model_bootstrap = {metric: [] for metric in metric_names}
        for replicate in range(bootstrap_replicates):
            sampled_pairs = rng.choice(pair_ids, size=len(pair_ids), replace=True)
            sampled_indices = np.concatenate(
                [pair_to_indices[pair] for pair in sampled_pairs]
            )
            sample = predictions.loc[sampled_indices]
            sample = sample[sample["Model"].eq(model)]
            y_b = sample["Label"].to_numpy(dtype=int)
            score_b = sample["Score"].to_numpy(dtype=float)
            if np.unique(y_b).size < 2:
                continue
            for metric in metric_names:
                value = _safe_metric(metric, y_b, score_b, threshold)
                model_bootstrap[metric].append(value)
                bootstrap_rows.append(
                    {
                        "Model": model,
                        "Replicate": replicate + 1,
                        "Metric": metric,
                        "Value": value,
                    }
                )

        for metric in metric_names:
            values = np.asarray(model_bootstrap[metric], dtype=float)
            summary_rows.append(
                {
                    "Model": model,
                    "Metric": metric,
                    "Estimate": point[metric],
                    "CI95_low": float(np.nanquantile(values, 0.025)),
                    "CI95_high": float(np.nanquantile(values, 0.975)),
                    "Bootstrap_unit": "TCGA participant pair",
                    "Bootstrap_replicates_completed": int(np.isfinite(values).sum()),
                    "Threshold": threshold,
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(bootstrap_rows)


def _paired_gene_replication(
    external_log2: pd.DataFrame,
    manifest: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    metadata = manifest.set_index("sample_id")
    rows = []
    p_values = []

    for row in panel.itertuples(index=False):
        gene = row.Gene
        differences = []
        tumor_values = []
        normal_values = []
        for pair_id, pair_block in manifest.groupby("Pair_ID"):
            tumor_id = pair_block.loc[pair_block["Label"].eq(1), "sample_id"].iloc[0]
            normal_id = pair_block.loc[pair_block["Label"].eq(0), "sample_id"].iloc[0]
            tumor = float(external_log2.loc[tumor_id, gene])
            normal = float(external_log2.loc[normal_id, gene])
            differences.append(tumor - normal)
            tumor_values.append(tumor)
            normal_values.append(normal)

        differences_array = np.asarray(differences, dtype=float)
        try:
            test = wilcoxon(differences_array, zero_method="wilcox", alternative="two-sided")
            p_value = float(test.pvalue)
        except ValueError:
            p_value = 1.0

        external_effect = float(np.median(differences_array))
        concordant = bool(
            np.sign(external_effect) == np.sign(float(row.Discovery_logFC))
        )
        rows.append(
            {
                "Rank_NIBFS": int(row.Rank_NIBFS),
                "Gene": gene,
                "Discovery_logFC": float(row.Discovery_logFC),
                "Discovery_direction": row.Discovery_direction,
                "TCGA_median_paired_log2TPM_difference": external_effect,
                "TCGA_mean_tumor_log2TPM": float(np.mean(tumor_values)),
                "TCGA_mean_normal_log2TPM": float(np.mean(normal_values)),
                "Direction_concordant": concordant,
                "Paired_Wilcoxon_p": p_value,
                "Pairs": len(differences),
            }
        )
        p_values.append(p_value)

    table = pd.DataFrame(rows)
    table["Paired_Wilcoxon_FDR"] = multipletests(
        p_values,
        method="fdr_bh",
    )[1]
    table["Significant_FDR_0.05"] = table["Paired_Wilcoxon_FDR"] <= 0.05

    concordant_count = int(table["Direction_concordant"].sum())
    binomial = binomtest(
        concordant_count,
        n=len(table),
        p=0.5,
        alternative="greater",
    )
    table.attrs["direction_concordance_count"] = concordant_count
    table.attrs["direction_concordance_fraction"] = concordant_count / len(table)
    table.attrs["direction_concordance_binomial_p"] = float(binomial.pvalue)
    return table


def _plot_roc(predictions: pd.DataFrame, output_dir: Path) -> list[Path]:
    figure, axis = plt.subplots(figsize=(7.5, 6.5))
    for model, block in predictions.groupby("Model", sort=False):
        fpr, tpr, _ = roc_curve(block["Label"], block["Score"])
        auc = roc_auc_score(block["Label"], block["Score"])
        axis.plot(fpr, tpr, label=f"{model} (AUC={auc:.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    axis.set_xlabel("False-positive rate")
    axis.set_ylabel("True-positive rate")
    axis.set_title("TCGA-BRCA paired RNA-seq external validation")
    axis.legend(loc="lower right")
    figure.tight_layout()
    paths = [
        output_dir / "Figure_TCGA_BRCA_RNAseq_ROC.png",
        output_dir / "Figure_TCGA_BRCA_RNAseq_ROC.pdf",
    ]
    figure.savefig(paths[0], dpi=300, bbox_inches="tight")
    figure.savefig(paths[1], bbox_inches="tight")
    plt.close(figure)
    return paths


def _plot_direction_replication(table: pd.DataFrame, output_dir: Path) -> list[Path]:
    ordered = table.sort_values("Rank_NIBFS", ascending=False)
    figure, axis = plt.subplots(figsize=(8.5, 7.5))
    axis.barh(
        ordered["Gene"],
        ordered["TCGA_median_paired_log2TPM_difference"],
    )
    axis.axvline(0.0, linewidth=1)
    axis.set_xlabel("Median paired tumor − normal log2(TPM+1)")
    axis.set_ylabel("Frozen NIBFS gene")
    axis.set_title("Directional replication in TCGA-BRCA RNA-seq")
    figure.tight_layout()
    paths = [
        output_dir / "Figure_TCGA_BRCA_gene_direction_replication.png",
        output_dir / "Figure_TCGA_BRCA_gene_direction_replication.pdf",
    ]
    figure.savefig(paths[0], dpi=300, bbox_inches="tight")
    figure.savefig(paths[1], bbox_inches="tight")
    plt.close(figure)
    return paths


def _write_manuscript_text(
    *,
    output_dir: Path,
    manifest: pd.DataFrame,
    metrics: pd.DataFrame,
    replication: pd.DataFrame,
    shared_genes: int,
) -> Path:
    n_pairs = manifest["Pair_ID"].nunique()
    n_samples = len(manifest)
    concordant = int(replication["Direction_concordant"].sum())
    significant = int(replication["Significant_FDR_0.05"].sum())

    auc_rows = metrics[metrics["Metric"].eq("ROC_AUC")].copy()
    auc_phrases = [
        (
            f"{row.Model}: {row.Estimate:.4f} "
            f"(pair-bootstrap 95% CI {row.CI95_low:.4f}–{row.CI95_high:.4f})"
        )
        for row in auc_rows.itertuples(index=False)
    ]

    text = f"""SUPPLEMENTARY METHODS — READY TO ADAPT

Independent cross-technology validation used paired TCGA-BRCA RNA-seq samples
from the NCI Genomic Data Commons. One Primary Tumor and one Solid Tissue Normal
STAR-Counts sample were selected deterministically per participant, resulting in
{n_pairs} participant pairs ({n_samples} samples). The exact frozen NIBFS top-20
panel and discovery directions were retained without external feature selection.
Microarray development expression and TCGA-BRCA log2(TPM+1) expression were
converted separately to within-sample percentile ranks across {shared_genes}
shared genes. This label-free transformation was fixed before examining external
outcomes. A signed panel score was calculated directly from the frozen discovery
directions. Logistic regression, random forest, and LightGBM transfer models were
fitted only on the original 608-sample microarray development set using the
frozen top-20 rank features and were applied once to TCGA-BRCA. External labels
were used only for evaluation. Confidence intervals were estimated by 2,000
bootstrap resamples at the participant-pair level.

SUPPLEMENTARY RESULTS — READY TO ADAPT

In paired TCGA-BRCA RNA-seq validation, ROC-AUC values were {'; '.join(auc_phrases)}.
The discovery direction was replicated for {concordant}/20 panel genes, and
{significant}/20 genes showed paired tumor-normal differences at Benjamini-
Hochberg FDR < 0.05. These results support cross-technology replication of the
frozen panel. Because transfer classifiers were refitted on a label-free
within-sample rank representation using development data only, this analysis
validates panel transferability rather than application of an unchanged
microarray-scale fitted model.
"""
    path = output_dir / "manuscript_text_TCGA_BRCA_RNAseq_validation.txt"
    path.write_text(text, encoding="utf-8")
    return path


def _write_manifest(output_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "output_file_manifest.csv":
            continue
        rows.append(
            {
                "Relative_path": str(path.relative_to(output_dir)),
                "Bytes": path.stat().st_size,
                "SHA256": _sha256_file(path),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "output_file_manifest.csv", index=False)
    return table


def run_tcga_brca_rnaseq_validation(
    namespace: Mapping[str, Any],
) -> dict[str, Any]:
    config = RNASeqValidationConfig(
        bootstrap_replicates=int(
            namespace.get("RNASEQ_BOOTSTRAP_REPLICATES", 2000)
        ),
        random_state=int(namespace.get("RNASEQ_RANDOM_STATE", 42)),
        download_workers=int(namespace.get("RNASEQ_DOWNLOAD_WORKERS", 8)),
        parse_workers=int(namespace.get("RNASEQ_PARSE_WORKERS", 4)),
        resume=bool(namespace.get("RNASEQ_RESUME", True)),
        force_rerun=bool(namespace.get("RNASEQ_FORCE_RERUN", False)),
        smoke_test_pairs=namespace.get("RNASEQ_SMOKE_TEST_PAIRS"),
    )
    config.validate()

    started = time.time()
    project_dir = _resolve_project_dir(namespace)
    output_dir = _resolve_output_dir(namespace, project_dir)
    cache_dir = _resolve_local_cache_dir(namespace)
    store = _resolve_store(namespace, project_dir)
    cfg = _load_cfg(namespace, project_dir)

    marker = output_dir / "TCGA_BRCA_RNASEQ_VALIDATION.marker"
    if config.force_rerun and output_dir.exists():
        if not marker.exists():
            raise RuntimeError(f"Refusing to delete unmarked directory: {output_dir}")
        shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        "Dedicated TCGA-BRCA paired RNA-seq validation output.\n",
        encoding="utf-8",
    )

    fold_assignments, fold_source = _resolve_development_ids(
        namespace,
        project_dir,
        config.expected_development_samples,
    )
    development_ids = fold_assignments["GSM_ID"].astype(str).tolist()
    labels = _resolve_sample_labels(store, development_ids)
    panel, panel_source = _resolve_panel(namespace, project_dir)

    panel.to_csv(output_dir / "frozen_NIBFS_top20_used.csv", index=False)
    fold_assignments.to_csv(
        output_dir / "development_fold_assignments_used.csv",
        index=False,
    )

    print("Building the 608-sample development gene matrix...", flush=True)
    X_development, selected_probes, log2_audit = _build_development_gene_matrix(
        project_dir=project_dir,
        store=store,
        cfg=cfg,
        development_ids=development_ids,
    )
    selected_probes.to_csv(
        output_dir / "development_representative_probe_mapping.csv",
        index=False,
    )
    log2_audit.to_csv(
        output_dir / "development_log2_audit.csv",
        index=False,
    )

    print("Querying GDC for paired TCGA-BRCA samples...", flush=True)
    selected_samples = _query_tcga_brca_samples(config)
    selected_samples.to_csv(
        output_dir / "TCGA_BRCA_candidate_paired_samples.csv",
        index=False,
    )

    print("Querying GDC for open STAR-Counts files...", flush=True)
    file_manifest = _query_star_count_files(selected_samples, config)
    file_manifest.to_csv(
        output_dir / "TCGA_BRCA_selected_STAR_Counts_manifest.csv",
        index=False,
    )

    processed_matrix_path = output_dir / "TCGA_BRCA_paired_log2_TPM_full.csv.gz"
    downloaded_manifest_path = output_dir / "TCGA_BRCA_download_audit.csv"

    if (
        config.resume
        and processed_matrix_path.is_file()
        and downloaded_manifest_path.is_file()
    ):
        print("Using cached processed TCGA-BRCA expression matrix.", flush=True)
        X_external = pd.read_csv(
            processed_matrix_path,
            index_col=0,
            compression="gzip",
        )
        downloaded_manifest = pd.read_csv(downloaded_manifest_path)
    else:
        raw_dir = cache_dir / "star_counts"
        downloaded_manifest = _download_files(file_manifest, raw_dir, config)
        downloaded_manifest.to_csv(downloaded_manifest_path, index=False)
        X_external = _build_external_expression(downloaded_manifest, config)
        X_external.to_csv(
            processed_matrix_path,
            compression="gzip",
        )

    # Align metadata and expression.
    manifest = file_manifest[
        file_manifest["sample_id"].isin(X_external.index.astype(str))
    ].copy()
    manifest = manifest.sort_values(["Pair_ID", "Label"]).reset_index(drop=True)
    X_external.index = X_external.index.astype(str)
    X_external.columns = X_external.columns.astype(str).str.upper()
    X_external = X_external.loc[manifest["sample_id"].astype(str)]

    shared = sorted(set(X_development.columns) & set(X_external.columns))
    missing_panel = sorted(set(panel["Gene"]) - set(shared))
    if missing_panel and config.require_complete_panel:
        raise RuntimeError(
            "RNA-seq validation does not have complete frozen-panel coverage: "
            + ", ".join(missing_panel)
        )
    if len(shared) < 5000:
        raise RuntimeError(
            f"Only {len(shared)} genes are shared between platforms; expected thousands."
        )

    coverage = panel.copy()
    coverage["Present_in_development"] = coverage["Gene"].isin(X_development.columns)
    coverage["Present_in_TCGA_BRCA"] = coverage["Gene"].isin(X_external.columns)
    coverage["Used"] = coverage["Gene"].isin(shared)
    coverage.to_csv(
        output_dir / "TCGA_BRCA_frozen_panel_gene_coverage.csv",
        index=False,
    )

    X_dev_rank = _within_sample_percentile_rank(
        X_development.loc[:, shared]
    )
    X_ext_rank = _within_sample_percentile_rank(
        X_external.loc[:, shared]
    )
    top20 = panel["Gene"].tolist()
    X_dev_top20 = X_dev_rank.loc[:, top20].copy()
    X_ext_top20 = X_ext_rank.loc[:, top20].copy()

    X_dev_top20.to_csv(
        output_dir / "development_rank_features_frozen_top20.csv.gz",
        compression="gzip",
    )
    X_ext_top20.to_csv(
        output_dir / "TCGA_BRCA_rank_features_frozen_top20.csv.gz",
        compression="gzip",
    )

    prediction_rows = []

    # Fully frozen signed-panel score.
    dev_signed = _signed_panel_score(X_dev_top20, panel)
    ext_signed = _signed_panel_score(X_ext_top20, panel)
    for sample_id, score in ext_signed.items():
        row = manifest.loc[manifest["sample_id"].eq(sample_id)].iloc[0]
        prediction_rows.append(
            {
                "sample_id": sample_id,
                "Pair_ID": row["Pair_ID"],
                "case_id": row["case_id"],
                "case_submitter_id": row["case_submitter_id"],
                "sample_type": row["sample_type"],
                "Label": int(row["Label"]),
                "Model": "FrozenSignedPanelScore",
                "Score": float(score),
                "Threshold": 0.0,
                "Representation": "within-sample percentile rank",
                "External_labels_used_for_fitting": False,
            }
        )

    # Development-only transfer classifiers.
    models = _create_models(config.random_state)
    for model_name, model in models.items():
        fitted = clone(model).fit(X_dev_top20, labels.loc[X_dev_top20.index])
        probabilities = fitted.predict_proba(X_ext_top20)[:, 1]
        for sample_id, probability in zip(X_ext_top20.index, probabilities):
            row = manifest.loc[manifest["sample_id"].eq(sample_id)].iloc[0]
            prediction_rows.append(
                {
                    "sample_id": sample_id,
                    "Pair_ID": row["Pair_ID"],
                    "case_id": row["case_id"],
                    "case_submitter_id": row["case_submitter_id"],
                    "sample_type": row["sample_type"],
                    "Label": int(row["Label"]),
                    "Model": model_name,
                    "Score": float(probability),
                    "Threshold": 0.5,
                    "Representation": "within-sample percentile rank",
                    "External_labels_used_for_fitting": False,
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(
        output_dir / "TCGA_BRCA_RNAseq_predictions.csv",
        index=False,
    )

    metrics, bootstrap = _pair_bootstrap_metrics(
        predictions,
        bootstrap_replicates=config.bootstrap_replicates,
        random_state=config.random_state,
    )
    metrics.to_csv(
        output_dir / "TCGA_BRCA_RNAseq_performance_with_pair_bootstrap_CI.csv",
        index=False,
    )
    bootstrap.to_csv(
        output_dir / "TCGA_BRCA_RNAseq_pair_bootstrap_distribution.csv.gz",
        index=False,
        compression="gzip",
    )

    replication = _paired_gene_replication(
        X_external.loc[:, top20],
        manifest,
        panel,
    )
    replication.to_csv(
        output_dir / "TCGA_BRCA_gene_direction_replication.csv",
        index=False,
    )

    concordance_summary = pd.DataFrame(
        [
            {
                "Panel_size": len(replication),
                "Direction_concordant_genes": int(
                    replication["Direction_concordant"].sum()
                ),
                "Direction_concordance_fraction": float(
                    replication["Direction_concordant"].mean()
                ),
                "Direction_concordance_one_sided_binomial_p": float(
                    replication.attrs["direction_concordance_binomial_p"]
                ),
                "Genes_significant_paired_Wilcoxon_FDR_0.05": int(
                    replication["Significant_FDR_0.05"].sum()
                ),
                "TCGA_participant_pairs": int(manifest["Pair_ID"].nunique()),
            }
        ]
    )
    concordance_summary.to_csv(
        output_dir / "TCGA_BRCA_gene_direction_replication_summary.csv",
        index=False,
    )

    # Compact supplementary tables.
    supplementary_metrics = metrics.pivot(
        index="Model",
        columns="Metric",
        values=["Estimate", "CI95_low", "CI95_high"],
    )
    supplementary_metrics.columns = [
        f"{stat}_{metric}" for stat, metric in supplementary_metrics.columns
    ]
    supplementary_metrics = supplementary_metrics.reset_index()
    supplementary_metrics.insert(
        1,
        "External_cohort",
        "TCGA-BRCA paired RNA-seq",
    )
    supplementary_metrics.insert(
        2,
        "Pairs",
        int(manifest["Pair_ID"].nunique()),
    )
    supplementary_metrics.to_csv(
        output_dir / "Supplementary_Table_SXX_TCGA_BRCA_RNAseq_validation.csv",
        index=False,
    )

    supplementary_replication = replication[
        [
            "Rank_NIBFS",
            "Gene",
            "Discovery_logFC",
            "Discovery_direction",
            "TCGA_median_paired_log2TPM_difference",
            "Direction_concordant",
            "Paired_Wilcoxon_p",
            "Paired_Wilcoxon_FDR",
            "Significant_FDR_0.05",
            "Pairs",
        ]
    ].copy()
    supplementary_replication.to_csv(
        output_dir / "Supplementary_Table_SXY_TCGA_BRCA_gene_replication.csv",
        index=False,
    )

    figure_paths = []
    figure_paths.extend(_plot_roc(predictions, output_dir))
    figure_paths.extend(_plot_direction_replication(replication, output_dir))

    manuscript_path = _write_manuscript_text(
        output_dir=output_dir,
        manifest=manifest,
        metrics=metrics,
        replication=replication,
        shared_genes=len(shared),
    )

    summary = {
        "analysis": "TCGA_BRCA_paired_RNAseq_external_validation",
        "started_utc": _utc_now(),
        "completed_utc": _utc_now(),
        "project_dir": str(project_dir),
        "output_dir": str(output_dir),
        "fold_assignment_source": fold_source,
        "panel_source": panel_source,
        "development_samples": int(len(development_ids)),
        "development_cancer": int(labels.sum()),
        "development_normal": int((1 - labels).sum()),
        "external_pairs": int(manifest["Pair_ID"].nunique()),
        "external_samples": int(len(manifest)),
        "shared_gene_universe": int(len(shared)),
        "frozen_panel_size": int(len(top20)),
        "complete_panel_coverage": not bool(missing_panel),
        "direction_concordant_genes": int(replication["Direction_concordant"].sum()),
        "direction_concordance_fraction": float(
            replication["Direction_concordant"].mean()
        ),
        "direction_concordance_binomial_p": float(
            replication.attrs["direction_concordance_binomial_p"]
        ),
        "significant_genes_FDR_0.05": int(
            replication["Significant_FDR_0.05"].sum()
        ),
        "bootstrap_replicates": config.bootstrap_replicates,
        "bootstrap_unit": "participant pair",
        "cross_platform_representation": "within-sample percentile rank across shared genes",
        "external_labels_used_for_fitting": False,
        "scientific_scope": (
            "Frozen-panel biological replication and development-only "
            "rank-transfer classification; not unchanged microarray-scale model transfer."
        ),
        "runtime_minutes": (time.time() - started) / 60.0,
        "config": asdict(config),
        "python_version": sys.version,
        "platform": platform.platform(),
    }
    summary_path = output_dir / "TCGA_BRCA_RNAseq_validation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    _write_manifest(output_dir)
    zip_path = shutil.make_archive(
        str(output_dir),
        "zip",
        root_dir=output_dir.parent,
        base_dir=output_dir.name,
    )

    print("\n" + "=" * 78)
    print("TCGA-BRCA RNA-SEQ EXTERNAL VALIDATION COMPLETE")
    print("=" * 78)
    print("Pairs:", summary["external_pairs"])
    print("Shared genes:", summary["shared_gene_universe"])
    print("Complete top-20 coverage:", summary["complete_panel_coverage"])
    print(
        "Direction concordance:",
        f"{summary['direction_concordant_genes']}/20",
    )
    print("Results:", output_dir)
    print("Backup ZIP:", zip_path)

    return {
        "output_dir": output_dir,
        "backup_zip": Path(zip_path),
        "summary": summary,
        "metrics": metrics,
        "replication": replication,
        "predictions": predictions,
        "manuscript_text": manuscript_path,
    }


def main(namespace: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if namespace is None:
        namespace = globals()
    return run_tcga_brca_rnaseq_validation(namespace)
