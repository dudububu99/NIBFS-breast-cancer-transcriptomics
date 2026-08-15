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


def copy_file(source: str | Path, destination: str | Path) -> None:
    source = Path(source)
    destination = Path(destination)
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)

def copy_figure_family(source_png: str | Path, alias_pngs: Iterable[str | Path]) -> None:
    """Copy a PNG and its PDF companion to compatibility filenames."""
    source_png = Path(source_png)
    for alias in alias_pngs:
        alias = Path(alias)
        copy_file(source_png, alias)
        source_pdf = source_png.with_suffix(".pdf")
        copy_file(source_pdf, alias.with_suffix(".pdf"))

def dataframe_checklist(directory: str | Path, filenames: Iterable[str], category: str) -> pd.DataFrame:
    directory = Path(directory)
    rows = []
    for name in filenames:
        path = directory / name
        row = {
            "Category": category,
            "File": name,
            "Exists": path.exists(),
            "Size_bytes": path.stat().st_size if path.exists() else 0,
            "Rows": np.nan,
            "Columns": np.nan,
        }
        if path.exists() and path.suffix.lower() in {".csv", ".gz"}:
            try:
                frame = pd.read_csv(path)
                row["Rows"] = len(frame)
                row["Columns"] = len(frame.columns)
            except Exception:
                pass
        rows.append(row)
    return pd.DataFrame(rows)

def export_expression_matrix(
    X: pd.DataFrame,
    output_csv: str | Path,
    *,
    save_gzip: bool = True,
    float_format: str = "%.7g",
) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    X.to_csv(output_csv, index=True, index_label="GSM_ID", float_format=float_format)
    if save_gzip:
        X.to_csv(output_csv.with_suffix(output_csv.suffix + ".gz"), index=True, index_label="GSM_ID", compression="gzip", float_format=float_format)

def clustered_heatmap_orders(X: pd.DataFrame, genes: list[str]) -> tuple[list[str], list[str], pd.DataFrame]:
    genes = [g for g in genes if g in X.columns]
    Z = (X[genes] - X[genes].mean()) / X[genes].std(ddof=1).replace(0, 1)
    if len(Z) > 1:
        sample_order = leaves_list(linkage(Z.to_numpy(), method="average", metric="correlation"))
    else:
        sample_order = np.arange(len(Z))
    if len(genes) > 1:
        gene_order = leaves_list(linkage(Z.T.to_numpy(), method="average", metric="correlation"))
    else:
        gene_order = np.arange(len(genes))
    sample_ids = Z.index.to_numpy()[sample_order].astype(str).tolist()
    ordered_genes = np.array(genes)[gene_order].astype(str).tolist()
    return sample_ids, ordered_genes, Z.loc[sample_ids, ordered_genes]

def choose_compact_heatmap_samples(
    metadata: pd.DataFrame,
    *,
    max_per_cohort_class: int = 6,
    random_state: int = 42,
) -> tuple[list[str], pd.DataFrame]:
    """Deterministic balanced subset for a compact publication heatmap."""
    rng = np.random.default_rng(random_state)
    rows = []
    chosen: list[str] = []
    for (geo, label), group in metadata.groupby(["GEO_ID", "Label"], sort=True):
        ids = group["GSM_ID"].astype(str).to_numpy()
        n = min(max_per_cohort_class, len(ids))
        if n:
            take = np.sort(rng.choice(ids, size=n, replace=False)).tolist()
            chosen.extend(take)
        rows.append({"GEO_ID": geo, "Label": label, "Available": len(ids), "Selected": n})
    return chosen, pd.DataFrame(rows)

def build_kan_bridge(
    cv_predictions: pd.DataFrame,
    heldout_predictions: pd.DataFrame,
    training_metadata: pd.DataFrame,
    heldout_metadata: pd.DataFrame,
    *,
    k: int,
    external_predictions: pd.DataFrame | None = None,
    external_metadata: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Create explicit RF/LightGBM meta-feature matrices for the later KAN-LM-QR study.

    OOF probabilities come only from the corresponding validation fold. Held-out and
    external probabilities come from base learners refitted on the complete model-development set.
    """

    def _pivot(pred: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        sub = pred[
            (pred["Feature_selection_method"] == "NIBFS")
            & (pred["k"] == k)
            & (pred["Classifier"].isin(["RF", "LightGBM"]))
        ].copy()
        if sub.empty:
            return pd.DataFrame()
        index_cols = ["Sample_ID", "True_Label"]
        if "Fold" in sub.columns:
            index_cols.append("Fold")
        wide = (
            sub.pivot_table(index=index_cols, columns="Classifier", values="Probability", aggfunc="first")
            .reset_index()
            .rename(columns={"RF": "p_RF", "LightGBM": "p_LightGBM"})
        )
        wide.insert(0, "Dataset", dataset_name)
        return wide

    train = _pivot(cv_predictions, "Model-development OOF")
    if not train.empty:
        meta = training_metadata[["GSM_ID", "GEO_ID", "Label"]].copy().rename(columns={"GSM_ID": "Sample_ID"})
        meta["Sample_ID"] = meta["Sample_ID"].astype(str)
        train = train.merge(meta, on="Sample_ID", how="left")
        train = train[["Dataset", "Sample_ID", "GEO_ID", "Label", "True_Label", "Fold", "p_RF", "p_LightGBM"]]
        if train["Sample_ID"].duplicated().any():
            raise ValueError("KAN OOF bridge contains duplicate model-development samples")
        if train[["p_RF", "p_LightGBM"]].isna().any().any():
            raise ValueError("KAN OOF bridge has missing RF or LightGBM probabilities")

    held = _pivot(heldout_predictions, "Post-harmonization held-out")
    if not held.empty:
        meta = heldout_metadata[["GSM_ID", "GEO_ID", "Label"]].copy().rename(columns={"GSM_ID": "Sample_ID"})
        meta["Sample_ID"] = meta["Sample_ID"].astype(str)
        held = held.merge(meta, on="Sample_ID", how="left")
        held = held[["Dataset", "Sample_ID", "GEO_ID", "Label", "True_Label", "p_RF", "p_LightGBM"]]

    external = pd.DataFrame()
    if external_predictions is not None and external_metadata is not None:
        external = _pivot(external_predictions, "Independent external GSE15852")
        if not external.empty:
            meta = external_metadata.copy()
            if "Label" not in meta.columns:
                if "Label_binary" not in meta.columns:
                    raise KeyError(
                        "External metadata must contain Label or Label_binary. "
                        f"Available columns: {meta.columns.tolist()}"
                    )
                meta["Label"] = meta["Label_binary"].map(
                    {0: "Normal", 1: "Tumor"}
                )
            meta = meta[["GSM_ID", "Label"]].copy().rename(
                columns={"GSM_ID": "Sample_ID"}
            )
            meta["Sample_ID"] = meta["Sample_ID"].astype(str)
            external = external.merge(meta, on="Sample_ID", how="left")
            external = external[["Dataset", "Sample_ID", "Label", "True_Label", "p_RF", "p_LightGBM"]]

    return {"train_oof": train, "heldout": held, "external": external}

def write_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def create_legacy_figure_aliases(figures_dir: str | Path) -> pd.DataFrame:
    figures_dir = Path(figures_dir)
    rows = []
    for canonical, aliases in LEGACY_FIGURE_ALIASES.items():
        source = figures_dir / canonical
        copy_figure_family(source, [figures_dir / alias for alias in aliases])
        for alias in aliases:
            rows.append({"Canonical": canonical, "Compatibility_alias": alias, "Created": (figures_dir / alias).exists()})
    return pd.DataFrame(rows)
