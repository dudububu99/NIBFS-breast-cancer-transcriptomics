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


def bh_adjust(pvalues):
    p=np.asarray(pvalues,dtype=float); n=len(p); order=np.argsort(p); ranked=p[order]
    adj=np.minimum.accumulate((ranked*n/np.arange(1,n+1))[::-1])[::-1]
    out=np.empty(n); out[order]=np.clip(adj,0,1); return out

def stability_statistical_tests(pairwise: pd.DataFrame, k: int=20, proposed: str="NIBFS") -> tuple[pd.DataFrame,pd.DataFrame]:
    df=pairwise[pairwise.k==k].copy(); key=["Fold_1","Fold_2"]
    wide=df.pivot_table(index=key,columns="Method",values="Jaccard",aggfunc="first").dropna()
    methods=list(wide.columns)
    if len(methods)>=3 and len(wide)>0:
        stat,p=friedmanchisquare(*[wide[m].to_numpy() for m in methods])
    else: stat,p=np.nan,np.nan
    global_df=pd.DataFrame([{"k":k,"Test":"Friedman","Statistic":stat,"P_value":p,"N_matched_pairs":len(wide),"Methods":";".join(methods)}])
    rows=[]
    for comp in methods:
        if comp==proposed: continue
        a=wide[proposed].to_numpy(); b=wide[comp].to_numpy(); diff=a-b
        try: w,pv=wilcoxon(a,b,alternative="greater",zero_method="wilcox")
        except ValueError: w,pv=np.nan,1.0
        rows.append({"k":k,"Proposed":proposed,"Comparator":comp,"Alternative":"greater","Statistic":w,"P_value":pv,"Mean_difference":float(np.mean(diff)),"Median_difference":float(np.median(diff)),"N_matched_pairs":len(diff)})
    post=pd.DataFrame(rows)
    if not post.empty: post["BH_adjusted_p"]=bh_adjust(post.P_value)
    return global_df,post

def cv_auc_statistical_tests(cv_metrics: pd.DataFrame, k: int=20, proposed: str="NIBFS") -> pd.DataFrame:
    df=cv_metrics[cv_metrics.k==k].copy(); rows=[]
    for classifier in sorted(df.Classifier.unique()):
        sub=df[df.Classifier==classifier]
        wide=sub.pivot_table(index="Fold",columns="Feature_selection_method",values="ROC_AUC",aggfunc="first").dropna()
        for comp in wide.columns:
            if comp==proposed: continue
            a=wide[proposed].to_numpy(); b=wide[comp].to_numpy(); diff=a-b
            try: w,pv=wilcoxon(a,b,alternative="two-sided",zero_method="wilcox")
            except ValueError: w,pv=np.nan,1.0
            rows.append({"Classifier":classifier,"Proposed":proposed,"Comparator":comp,"Metric":"ROC_AUC","Alternative":"two-sided","Statistic":w,"P_value":pv,"Mean_difference":float(np.mean(diff)),"Median_difference":float(np.median(diff)),"N_folds":len(diff)})
    out=pd.DataFrame(rows)
    if not out.empty: out["BH_adjusted_p"]=bh_adjust(out.P_value)
    return out

def derive_youden_thresholds(oof_predictions: pd.DataFrame, method: str="NIBFS", k: int=20) -> pd.DataFrame:
    df=oof_predictions[(oof_predictions.Feature_selection_method==method)&(oof_predictions.k==k)].copy(); rows=[]
    for clf,g in df.groupby("Classifier"):
        fpr,tpr,thresholds=roc_curve(g.True_Label,g.Probability)
        finite=np.isfinite(thresholds); fpr=fpr[finite];tpr=tpr[finite];thresholds=thresholds[finite]
        score=tpr-fpr; best=np.flatnonzero(score==score.max())
        idx=int(best[np.argmin(np.abs(thresholds[best]-0.5))])
        rows.append({"Feature_selection_method":method,"Classifier":clf,"k":k,"Threshold":float(thresholds[idx]),"Youden_J":float(score[idx]),"OOF_sensitivity":float(tpr[idx]),"OOF_specificity":float(1-fpr[idx]),"Rule":"Youden index from discovery OOF predictions"})
    return pd.DataFrame(rows)
