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


def run_enrichment(genes: Iterable[str], background: Iterable[str], sources: list[str]) -> pd.DataFrame:
    """Run g:Profiler while retaining the intersecting gene symbols.

    gprofiler-official omits the ``intersections`` column by default because
    ``no_evidences=True``. The paper tables require those genes, therefore the
    request explicitly sets ``no_evidences=False``.
    """
    from gprofiler import GProfiler

    gp = GProfiler(return_dataframe=True)
    return gp.profile(
        organism="hsapiens",
        query=list(genes),
        background=list(background),
        sources=sources,
        user_threshold=0.05,
        significance_threshold_method="fdr",
        no_evidences=False,
    )

def _format_intersections(value) -> str:
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return ";".join(map(str, value))
    if pd.isna(value):
        return ""
    text = str(value)
    # CSV reloads may turn a Python list into its string representation.
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                return ";".join(map(str, parsed))
        except Exception:
            pass
    return text

def standardize_enrichment(df: pd.DataFrame, final_k: int) -> pd.DataFrame:
    columns = [
        "Database",
        "Term",
        "Adjusted_p_value",
        "Gene_Count",
        "Associated_Genes",
        "Gene_Ratio",
        "minus_log10_adjusted_p",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    required = {"source", "name", "p_value", "intersection_size"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            "g:Profiler output is missing required columns: " + ", ".join(missing)
        )

    if "intersections" in df.columns:
        associated = df["intersections"].map(_format_intersections)
    elif "intersection" in df.columns:
        associated = df["intersection"].map(_format_intersections)
    else:
        # Keep the table usable if an API response omits evidence fields.
        associated = pd.Series([""] * len(df), index=df.index, dtype="object")

    out = pd.DataFrame(
        {
            "Database": df["source"],
            "Term": df["name"],
            "Adjusted_p_value": pd.to_numeric(df["p_value"], errors="coerce"),
            "Gene_Count": pd.to_numeric(df["intersection_size"], errors="coerce"),
            "Associated_Genes": associated,
        }
    )
    out["Gene_Ratio"] = out["Gene_Count"] / max(int(final_k), 1)
    out["minus_log10_adjusted_p"] = -np.log10(
        out["Adjusted_p_value"].clip(lower=1e-300)
    )
    return out.sort_values(["Database", "Adjusted_p_value"]).reset_index(drop=True)
