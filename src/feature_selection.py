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


def run_limma_rpy2(X: pd.DataFrame, y: Iterable[int]) -> pd.DataFrame:
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.conversion import localconverter
    except Exception as exc:
        raise ImportError("rpy2 and R/limma are required") from exc
    ro.r("suppressPackageStartupMessages(library(limma))")
    genes = X.columns.astype(str).tolist()
    expr = X.to_numpy(dtype=float).T
    y_arr = np.asarray(y, dtype=int)
    with localconverter(ro.default_converter + numpy2ri.converter):
        ro.globalenv["expr_matrix"] = expr
        ro.globalenv["group_vector"] = y_arr
    ro.globalenv["gene_names"] = ro.StrVector(genes)
    ro.r('''
      rownames(expr_matrix) <- gene_names
      group_factor <- factor(group_vector, levels=c(0,1), labels=c("Normal","Cancer"))
      design <- model.matrix(~ group_factor)
      fit <- lmFit(expr_matrix, design)
      fit <- eBayes(fit)
      limma_result <- topTable(fit, coef=2, number=Inf, adjust.method="BH", sort.by="none")
      limma_result$Gene <- rownames(limma_result)
    ''')
    cols = list(ro.r("colnames(limma_result)"))
    result = pd.DataFrame({c: list(ro.r(f"limma_result${c}")) for c in cols})
    result = result.rename(columns={"adj.P.Val": "FDR", "P.Value": "P_value"})
    result["Gene"] = result["Gene"].astype(str)
    result["Stat_score"] = result["logFC"].abs() * (-np.log10(result["FDR"].clip(lower=1e-300)))
    result["Rank_stat"] = result["Stat_score"].rank(method="average", ascending=False)
    result = result.sort_values(["Rank_stat", "Gene"], ascending=[True, True]).reset_index(drop=True)
    return result

def prepare_ppi_rank_table(ppi_degree: pd.DataFrame, gene_universe: Iterable[str]) -> pd.DataFrame:
    genes = pd.Index(pd.Series(list(gene_universe)).astype(str).drop_duplicates())
    cols=["Gene","Degree"] + (["Normalized_degree"] if "Normalized_degree" in ppi_degree.columns else [])
    ppi=ppi_degree[cols].copy();ppi["Degree"]=pd.to_numeric(ppi["Degree"],errors="coerce").fillna(0.0)
    if "Normalized_degree" not in ppi: ppi["Normalized_degree"]=ppi["Degree"]
    ppi["Normalized_degree"]=pd.to_numeric(ppi["Normalized_degree"],errors="coerce").fillna(0.0)
    ppi=ppi.groupby("Gene",as_index=False).agg({"Degree":"max","Normalized_degree":"max"})
    out=pd.DataFrame({"Gene":genes}).merge(ppi,on="Gene",how="left");out[["Degree","Normalized_degree"]]=out[["Degree","Normalized_degree"]].fillna(0.0)
    out["Rank_topo"]=out["Normalized_degree"].rank(method="average",ascending=False)
    return out.sort_values(["Rank_topo", "Gene"]).reset_index(drop=True)

def nibfs_rank(limma: pd.DataFrame, ppi_rank: pd.DataFrame, gene_universe: Iterable[str]) -> pd.DataFrame:
    p = len(list(gene_universe))
    out = limma.merge(ppi_rank[["Gene", "Degree", "Normalized_degree", "Rank_topo"]], on="Gene", how="left")
    out["Degree"] = out["Degree"].fillna(0.0)
    out["Normalized_degree"] = out["Normalized_degree"].fillna(0.0)
    out["Rank_topo"] = out["Rank_topo"].fillna(float(p))
    out["Borda_stat"] = p - out["Rank_stat"] + 1
    out["Borda_topo"] = p - out["Rank_topo"] + 1
    out["Borda_score"] = out["Borda_stat"] + out["Borda_topo"]
    out = out.sort_values(["Borda_score", "Rank_stat", "Gene"], ascending=[False, True, True]).reset_index(drop=True)
    out["Rank_NIBFS"] = np.arange(1, len(out) + 1)
    return out

def select_top_k(table: pd.DataFrame, k: int, rank_col: str) -> list[str]:
    return table.sort_values([rank_col, "Gene"]).head(k)["Gene"].astype(str).tolist()

def select_mrmr_features(X: pd.DataFrame, y: Iterable[int], k: int, candidate_size: int = 500, random_state: int = 42) -> pd.DataFrame:
    y_arr = np.asarray(y, dtype=int)
    relevance = pd.Series(mutual_info_classif(X, y_arr, random_state=random_state), index=X.columns)
    candidates = relevance.sort_values(ascending=False).head(min(candidate_size, X.shape[1])).index.tolist()
    corr = X[candidates].corr().abs().fillna(0.0)
    selected: list[str] = []
    rows = []
    while len(selected) < min(k, len(candidates)):
        best_gene, best_tuple = None, None
        for gene in candidates:
            if gene in selected:
                continue
            red = float(corr.loc[gene, selected].mean()) if selected else 0.0
            score = float(relevance.loc[gene] - red)
            candidate = (score, float(relevance.loc[gene]), -red)
            if best_tuple is None or candidate > best_tuple:
                best_tuple, best_gene = candidate, gene
        selected.append(best_gene)
        rows.append({"Gene": best_gene, "Selection_Order": len(selected), "mRMR_score": best_tuple[0], "Relevance": best_tuple[1], "Redundancy": -best_tuple[2]})
    return pd.DataFrame(rows)

def lasso_gene_ranking(X: pd.DataFrame, y: Iterable[int], C: float = 1.0, random_state: int = 42, solver: str = "saga", max_iter: int = 10000) -> pd.DataFrame:
    scaler=StandardScaler();Xs=scaler.fit_transform(X)
    model=LogisticRegression(C=C,penalty="l1",solver=solver,class_weight="balanced",max_iter=max_iter,n_jobs=-1 if solver=="saga" else None,random_state=random_state)
    model.fit(Xs,np.asarray(y,dtype=int))
    out=pd.DataFrame({"Gene":X.columns.astype(str),"Coefficient":model.coef_.ravel()});out["Abs_Coefficient"]=out.Coefficient.abs();out=out.sort_values(["Abs_Coefficient","Gene"],ascending=[False,True]).reset_index(drop=True);out["Rank_LASSO"]=np.arange(1,len(out)+1);out["C"]=float(C);return out

def pairwise_jaccard_summary(panels: Dict[int, List[str]], method: str, k: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows = []
    for i, j in combinations(sorted(panels), 2):
        a, b = set(panels[i]), set(panels[j])
        rows.append({"Method": method, "k": k, "Fold_1": i, "Fold_2": j, "Intersection": len(a & b), "Union": len(a | b), "Jaccard": len(a & b) / len(a | b)})
    pairwise = pd.DataFrame(rows)
    counts = Counter(g for genes in panels.values() for g in genes)
    freq = pd.DataFrame(counts.items(), columns=["Gene", "Fold_Frequency"]).sort_values(["Fold_Frequency", "Gene"], ascending=[False, True]).reset_index(drop=True)
    summary = {"Method": method, "k": k, "Mean_Jaccard": pairwise["Jaccard"].mean(), "SD_Jaccard": pairwise["Jaccard"].std(ddof=1), "Minimum": pairwise["Jaccard"].min(), "Maximum": pairwise["Jaccard"].max(), "Genes_in_all_5_folds": int((freq["Fold_Frequency"] == 5).sum()), "Genes_in_at_least_4_folds": int((freq["Fold_Frequency"] >= 4).sum())}
    return pairwise, freq, summary
