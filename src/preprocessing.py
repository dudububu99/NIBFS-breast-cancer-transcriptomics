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


def conditional_log2_probe_table(expression: pd.DataFrame, threshold: float = 100.0) -> tuple[pd.DataFrame, bool]:
    out = expression.copy()
    sample_cols = [c for c in out.columns if c != "ID_REF"]
    out[sample_cols] = out[sample_cols].apply(pd.to_numeric, errors="coerce")
    maximum = float(np.nanmax(out[sample_cols].to_numpy(dtype=float)))
    transformed = maximum > threshold
    if transformed:
        minimum = float(np.nanmin(out[sample_cols].to_numpy(dtype=float)))
        shift = 1.0 - minimum if minimum < 0 else 1.0
        out[sample_cols] = np.log2(out[sample_cols] + shift)
    return out, transformed

def highest_variance_probe_per_gene(expression: pd.DataFrame, probe_map: pd.DataFrame) -> pd.DataFrame:
    expr = expression.copy()
    expr["ID_REF"] = expr["ID_REF"].astype(str).str.strip()
    sample_cols = [c for c in expr.columns if c != "ID_REF"]
    expr[sample_cols] = expr[sample_cols].apply(pd.to_numeric, errors="coerce")
    mapped = expr.merge(probe_map, on="ID_REF", how="inner", validate="many_to_one")
    mapped["Probe_Variance"] = mapped[sample_cols].var(axis=1, ddof=1)
    selected = (mapped.sort_values(["Gene", "Probe_Variance", "ID_REF"], ascending=[True, False, True])
                .drop_duplicates("Gene"))
    out = selected[["Gene"] + sample_cols].set_index("Gene").T
    out.index.name = "GSM_ID"
    return out

def quantile_normalize_samples(X: pd.DataFrame) -> pd.DataFrame:
    X = X.astype(float)
    matrix = X.T.to_numpy()
    order = np.argsort(matrix, axis=0, kind="mergesort")
    sorted_matrix = np.sort(matrix, axis=0)
    rank_means = np.nanmean(sorted_matrix, axis=1)
    normalized = np.empty_like(matrix, dtype=float)
    for col in range(matrix.shape[1]):
        mapped = rank_means.copy()
        vals = sorted_matrix[:, col]
        start = 0
        while start < len(vals):
            end = start + 1
            while end < len(vals) and vals[end] == vals[start]:
                end += 1
            mapped[start:end] = np.mean(rank_means[start:end])
            start = end
        normalized[order[:, col], col] = mapped
    return pd.DataFrame(normalized.T, index=X.index, columns=X.columns)

def combat_harmonize(X: pd.DataFrame, batch: Iterable[str], labels: Iterable[int], preserve_class: bool = True) -> tuple[pd.DataFrame, dict]:
    try:
        from neuroCombat import neuroCombat
    except Exception as exc:
        raise ImportError("Install neuroCombat==0.2.12") from exc
    covars = pd.DataFrame({"batch": list(batch), "class": list(labels)}, index=X.index)
    result = neuroCombat(
        dat=X.T,
        covars=covars.reset_index(drop=True),
        batch_col="batch",
        categorical_cols=["class"] if preserve_class else None,
        eb=True,
        parametric=True,
    )
    corrected = pd.DataFrame(result["data"].T, index=X.index, columns=X.columns)
    return corrected, result["estimates"]

def filter_bottom_variance(X: pd.DataFrame, fraction: float = 0.10) -> tuple[pd.DataFrame, float]:
    variances = X.var(axis=0, ddof=1)
    cutoff = float(variances.quantile(fraction))
    keep = variances[variances > cutoff].sort_index().index
    return X.loc[:, keep].copy(), cutoff

def binary_labels(labels: Iterable[str]) -> np.ndarray:
    values = pd.Series(list(labels)).map({"Normal": 0, "Cancer": 1})
    if values.isna().any():
        raise ValueError("Unmapped labels found")
    return values.astype(int).to_numpy()

def zscore_by_training(X_train: pd.DataFrame, X_eval: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=1).replace(0, 1).fillna(1)
    return (X_train - mean) / std, (X_eval[X_train.columns] - mean) / std
