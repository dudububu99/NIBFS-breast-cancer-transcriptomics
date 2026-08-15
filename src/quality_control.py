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


def robust_z(values: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    med = float(x.median())
    mad = float((x-med).abs().median())
    if not np.isfinite(mad) or mad == 0:
        sd = float(x.std(ddof=1))
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.zeros(len(x)), index=x.index)
        return (x-med)/sd
    return 0.67448975*(x-med)/mad

def expression_distribution_summary(X: pd.DataFrame, metadata: pd.DataFrame, stage: str) -> pd.DataFrame:
    a = X.to_numpy(dtype=float)
    out = metadata[["GSM_ID","GEO_ID","Label"]].copy().reset_index(drop=True)
    out["Stage"] = stage
    out["Minimum"] = np.nanmin(a, axis=1)
    out["Q1"] = np.nanquantile(a, .25, axis=1)
    out["Median"] = np.nanmedian(a, axis=1)
    out["Mean"] = np.nanmean(a, axis=1)
    out["Q3"] = np.nanquantile(a, .75, axis=1)
    out["Maximum"] = np.nanmax(a, axis=1)
    out["IQR"] = out["Q3"]-out["Q1"]
    out["SD"] = np.nanstd(a, axis=1, ddof=1)
    out["Missing_count"] = np.isnan(a).sum(axis=1)
    out["Missing_percent"] = out["Missing_count"] / X.shape[1] * 100
    return out

def pca_table(X: pd.DataFrame, metadata: pd.DataFrame, stage: str, seed: int = 42) -> tuple[pd.DataFrame,pd.DataFrame]:
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(X.to_numpy(dtype=float))
    out = metadata[["GSM_ID","GEO_ID","Label"]].copy().reset_index(drop=True)
    out["PC1"] = coords[:,0]; out["PC2"] = coords[:,1]; out["Stage"] = stage
    var = pd.DataFrame({"Stage":[stage,stage],"Component":["PC1","PC2"],"Explained_variance_ratio":pca.explained_variance_ratio_[:2]})
    return out,var

def sample_correlation_audit(X: pd.DataFrame, metadata: pd.DataFrame, pca_after: pd.DataFrame, z_threshold: float=3.5) -> tuple[pd.DataFrame,pd.DataFrame]:
    corr = np.corrcoef(X.to_numpy(dtype=float))
    np.fill_diagonal(corr, np.nan)
    mean_corr = np.nanmean(corr,axis=1)
    dist = np.sqrt((pca_after.PC1-pca_after.PC1.median())**2 + (pca_after.PC2-pca_after.PC2.median())**2)
    dist_s = expression_distribution_summary(X,metadata,"After")
    out = metadata[["GSM_ID","GEO_ID","Label"]].copy().reset_index(drop=True)
    out["Mean_sample_correlation"] = mean_corr
    out["PCA_distance"] = dist.to_numpy()
    out["Sample_median"] = dist_s.Median.to_numpy()
    out["Sample_IQR"] = dist_s.IQR.to_numpy()
    out["Median_robust_z"] = robust_z(out.Sample_median)
    out["IQR_robust_z"] = robust_z(out.Sample_IQR)
    out["PCA_distance_robust_z"] = robust_z(out.PCA_distance)
    out["Mean_correlation_robust_z"] = robust_z(out.Mean_sample_correlation)
    out["Distribution_flag"] = (out.Median_robust_z.abs()>=z_threshold)|(out.IQR_robust_z.abs()>=z_threshold)
    out["PCA_flag"] = out.PCA_distance_robust_z>=z_threshold
    out["Low_correlation_flag"] = out.Mean_correlation_robust_z<=-z_threshold
    out["Audit_flag"] = (out.Distribution_flag|out.PCA_flag)&out.Low_correlation_flag
    corr_df = pd.DataFrame(corr,index=metadata.GSM_ID.astype(str),columns=metadata.GSM_ID.astype(str))
    return out,corr_df

def missing_value_summary(stage_matrices: dict[str,pd.DataFrame]) -> pd.DataFrame:
    rows=[]
    for stage,X in stage_matrices.items():
        a=X.to_numpy(dtype=float); n=int(np.isnan(a).sum())
        rows.append({"Stage":stage,"Samples":X.shape[0],"Genes":X.shape[1],"Missing_values":n,"Missing_percent":n/a.size*100 if a.size else 0.0})
    return pd.DataFrame(rows)

def preprocessing_stage_summary(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)
