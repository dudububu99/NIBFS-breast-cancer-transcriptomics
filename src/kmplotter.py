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


REQUIRED_ANY = {
    "Gene": ["Gene", "gene"],
    "Hazard_ratio": ["Hazard ratio", "Hazard_ratio", "HR", "hr"],
    "P_value": ["Log-rank p-value", "P_value", "p-value", "p_value", "p"],
}

def load_kmplotter_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()
    rename = {}
    for target, options in REQUIRED_ANY.items():
        col = next((c for c in options if c in df.columns), None)
        if col is None:
            raise ValueError(f"KM Plotter CSV missing {target}; columns={df.columns.tolist()}")
        rename[col] = target
    df = df.rename(columns=rename)
    if {"CI_low", "CI_high"}.issubset(df.columns):
        df["CI_low"] = pd.to_numeric(df["CI_low"], errors="coerce")
        df["CI_high"] = pd.to_numeric(df["CI_high"], errors="coerce")
    else:
        ci_col = next((c for c in ["95% confidence interval", "CI", "95% CI"] if c in df.columns), None)
        if ci_col is None:
            raise ValueError("KM Plotter CSV must contain CI_low and CI_high, or a combined CI column")
        parsed = df[ci_col].astype(str).str.extract(r"([0-9.]+)\s*[-–,]\s*([0-9.]+)")
        df["CI_low"] = pd.to_numeric(parsed[0], errors="coerce")
        df["CI_high"] = pd.to_numeric(parsed[1], errors="coerce")
    df["Hazard_ratio"] = pd.to_numeric(df["Hazard_ratio"], errors="coerce")
    df["P_value"] = pd.to_numeric(df["P_value"].astype(str).str.replace("<", "", regex=False), errors="coerce")
    return df
