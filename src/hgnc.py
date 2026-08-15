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


def _split(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [x.strip() for x in str(value).split("|") if x.strip()]

class HGNCResolver:
    def __init__(self, approved: dict[str, str], unique_aliases: dict[str, str]) -> None:
        self.approved = approved
        self.unique_aliases = unique_aliases

    @classmethod
    def from_complete_set(cls, path: str | Path) -> "HGNCResolver":
        df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
        df = df[df["status"].fillna("").str.lower().eq("approved")].copy()
        df["symbol"] = df["symbol"].astype(str).str.strip()
        approved = {s.upper(): s for s in df["symbol"] if s}
        candidates: dict[str, set[str]] = {}
        for row in df.itertuples(index=False):
            official = str(getattr(row, "symbol")).strip()
            for col in ("prev_symbol", "alias_symbol"):
                for alias in _split(getattr(row, col, None)):
                    candidates.setdefault(alias.upper(), set()).add(official)
        unique = {a: next(iter(v)) for a, v in candidates.items() if len(v) == 1 and a not in approved}
        return cls(approved=approved, unique_aliases=unique)

    def resolve(self, value: object) -> Optional[str]:
        if value is None or pd.isna(value):
            return None
        key = str(value).strip().strip('"').upper()
        if not key or key in {"---", "NA", "N/A", "NAN", "NONE", "NULL"}:
            return None
        return self.approved.get(key) or self.unique_aliases.get(key)

    def resolve_annotation_cell(self, value: object) -> Optional[str]:
        if value is None or pd.isna(value):
            return None
        tokens = re.split(r"\s*///\s*|\s*//\s*|\s*;\s*", str(value))
        genes = {self.resolve(token) for token in tokens}
        genes.discard(None)
        return next(iter(genes)) if len(genes) == 1 else None
