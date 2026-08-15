from __future__ import annotations
from io import StringIO
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

from .hgnc import HGNCResolver

def open_geo_file(path: str | Path):
    path = str(path)
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") if path.endswith(".gz") else open(path, "r", encoding="utf-8", errors="replace")

def parse_series_matrix(path: str | Path) -> Tuple[pd.DataFrame, Dict[str, List[List[str]]]]:
    meta_rows, expr_lines, inside = [], [], False
    with open_geo_file(path) as handle:
        for line in handle:
            clean = line.rstrip("\n")
            if clean == "!series_matrix_table_begin":
                inside = True
                continue
            if clean == "!series_matrix_table_end":
                break
            if inside:
                expr_lines.append(clean)
            elif clean.startswith("!"):
                parts = clean.split("\t")
                meta_rows.append((parts[0].lstrip("!"), [v.strip('"') for v in parts[1:]]))
    if not expr_lines:
        raise ValueError(f"Expression table not found in {path}")
    expression = pd.read_csv(StringIO("\n".join(expr_lines)), sep="\t", low_memory=False)
    metadata: Dict[str, List[List[str]]] = {}
    for key, values in meta_rows:
        metadata.setdefault(key, []).append(values)
    return expression, metadata

def create_sample_metadata(metadata_dict: Dict[str, List[List[str]]]) -> pd.DataFrame:
    ids = metadata_dict.get("Sample_geo_accession", [None])[0]
    if ids is None:
        raise ValueError("Sample_geo_accession missing")
    out = pd.DataFrame({"GSM_ID": ids})
    for field in ["Sample_title", "Sample_source_name_ch1", "Sample_organism_ch1", "Sample_description"]:
        if field in metadata_dict and len(metadata_dict[field][0]) == len(ids):
            out[field] = metadata_dict[field][0]
    for i, values in enumerate(metadata_dict.get("Sample_characteristics_ch1", []), start=1):
        if len(values) == len(ids):
            out[f"Sample_characteristics_ch1_{i}"] = values
    return out

def label_discovery_sample(geo_id: str, row: pd.Series) -> str:
    title = str(row.get("Sample_title", "")).lower()
    source = str(row.get("Sample_source_name_ch1", "")).lower()
    all_text = " | ".join(str(v).lower() for v in row.fillna("").values)
    if geo_id == "GSE61304":
        if "normal breast epithelium" in title: return "Normal"
        if "breast tumor epithelium" in title: return "Cancer"
    elif geo_id == "GSE42568":
        if "breast tissue, normal" in source: return "Normal"
        if "breast tissue, cancer" in source: return "Cancer"
    elif geo_id == "GSE29044":
        if re.search(r"\btumou?r tissue\b", title): return "Cancer"
        if re.search(r"\bnormal tissue\b", title): return "Normal"
    elif geo_id == "GSE26910":
        if "prostate" in all_text: return "Exclude"
        if re.search(r"normal\s*/\s*tumou?r\s*:\s*normal", all_text) or "breast normal" in all_text: return "Normal"
        if re.search(r"normal\s*/\s*tumou?r\s*:\s*tumou?r", all_text) or "breast tumor" in all_text or "breast tumour" in all_text: return "Cancer"
    elif geo_id == "GSE3744":
        if "breast tumor" in all_text: return "Cancer"
        if "normal breast" in all_text or "breast normal" in all_text: return "Normal"
    elif geo_id == "GSE29431":
        if title.startswith("normal-"): return "Normal"
        if title.startswith("tumor-") or "breast cancer tissue" in source or "disease state: breast cancer" in all_text or "disease state: cancer" in all_text: return "Cancer"
    elif geo_id == "GSE31138":
        if title.startswith("normal_"): return "Normal"
        if title.startswith("cancer_"): return "Cancer"
    elif geo_id == "GSE71053":
        if "disease status: normal" in all_text: return "Normal"
        if "disease status: tumor" in all_text: return "Cancer"
    elif geo_id == "GSE10780":
        if "(normal)" in all_text or "unremarkable breast ducts" in all_text: return "Normal"
        if any(x in all_text for x in ["invasive ductal carcinoma", "ductal carcinoma in situ", "carcinoma", "(tumor)"]): return "Cancer"
    elif geo_id in {"GSE30010", "GSE111662"}:
        return "Normal"
    return "Exclude"

def label_external_samples(meta: pd.DataFrame, geo_id: str) -> pd.DataFrame:
    text_cols = [c for c in meta.columns if c.startswith("Sample_")]
    text = meta[text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    out = meta.copy()
    out["Label_binary"] = np.nan
    if geo_id == "GSE15852":
        out.loc[text.str.contains("normal", regex=False), "Label_binary"] = 0
        out.loc[text.str.contains("cancer|tumor|tumour", regex=True), "Label_binary"] = 1
    else:
        raise ValueError(f"No external labeling rule for {geo_id}")
    out = out.dropna(subset=["Label_binary"]).copy()
    out["Label_binary"] = out["Label_binary"].astype(int)
    return out

def read_geo_annotation(path: str | Path, resolver: HGNCResolver) -> pd.DataFrame:
    lines, started = [], False
    with open_geo_file(path) as handle:
        for line in handle:
            if line.startswith("ID\t"):
                started = True
            if started:
                lines.append(line)
    if not lines:
        raise ValueError(f"Annotation table not found in {path}")
    annot = pd.read_csv(StringIO("".join(lines)), sep="\t", dtype=str, low_memory=False)
    candidates = ["Gene symbol", "Gene Symbol", "GENE_SYMBOL", "Symbol"]
    gene_col = next((c for c in candidates if c in annot.columns), None)
    if gene_col is None:
        raise ValueError(f"Gene-symbol column not found: {annot.columns.tolist()}")
    out = annot[["ID", gene_col]].rename(columns={"ID": "ID_REF", gene_col: "Raw_Gene"}).copy()
    out["ID_REF"] = out["ID_REF"].astype(str).str.strip()
    out["Gene"] = out["Raw_Gene"].map(resolver.resolve_annotation_cell)
    out = out.dropna(subset=["Gene"])[["ID_REF", "Gene"]]
    valid = out.groupby("ID_REF")["Gene"].nunique()
    valid_ids = valid[valid == 1].index
    return out[out["ID_REF"].isin(valid_ids)].drop_duplicates("ID_REF").reset_index(drop=True)
