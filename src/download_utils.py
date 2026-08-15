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


HGNC_COMPLETE_SET_URL = "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"

def geo_series_url(gse: str) -> str:
    number = int(gse.upper().replace("GSE", ""))
    family = f"GSE{number // 1000}nnn"
    return f"https://ftp.ncbi.nlm.nih.gov/geo/series/{family}/{gse.upper()}/matrix/{gse.upper()}_series_matrix.txt.gz"

def geo_annotation_url(gpl: str) -> str:
    number = int(gpl.upper().replace("GPL", ""))
    # NCBI GEO stores platform accessions below GPL1000 in the GPLnnn bucket
    # (for example GPL96 and GPL570), not in a non-existent GPL0nnn bucket.
    family = "GPLnnn" if number < 1000 else f"GPL{number // 1000}nnn"
    return f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/{family}/{gpl.upper()}/annot/{gpl.upper()}.annot.gz"

def string_reference_urls(species: int = 9606, version: str = "12.0") -> dict[str, str]:
    base = "https://stringdb-downloads.org/download"
    return {
        "protein_info": f"{base}/protein.info.v{version}/{species}.protein.info.v{version}.txt.gz",
        "protein_links": f"{base}/protein.links.v{version}/{species}.protein.links.v{version}.txt.gz",
    }

def download_if_missing(url: str, destination: str | Path, timeout: int = 180) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    tmp = destination.with_suffix(destination.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "NIBFS-reproducibility/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response, tmp.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        tmp.replace(destination)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return destination

def geo_series_urls(gse: str, gpl: str | None = None) -> list[str]:
    """Return generic and platform-specific GEO Series Matrix candidates."""
    gse = gse.upper()
    generic = geo_series_url(gse)
    urls = [generic]

    if gpl:
        folder = generic.rsplit("/", 1)[0]
        urls.append(
            f"{folder}/{gse}-{gpl.upper()}_series_matrix.txt.gz"
        )

    return urls

def download_first_available(
    urls: list[str],
    destination: str | Path,
    timeout: int = 180
) -> Path:
    """Try alternative URLs and use the first one that exists."""
    errors = []

    for url in urls:
        try:
            print(f"Trying download: {url}", flush=True)
            result = download_if_missing(url, destination, timeout=timeout)
            print(f"Downloaded/available: {result}", flush=True)
            return result
        except urllib.error.HTTPError as exc:
            errors.append(f"{url} -> HTTP {exc.code}")
            if exc.code != 404:
                raise

    raise RuntimeError(
        "No GEO Series Matrix candidate was available:\n"
        + "\n".join(errors)
    )
