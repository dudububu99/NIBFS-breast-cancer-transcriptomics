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


def create_models(cfg: dict) -> dict:
    seed=int(cfg["project"]["random_state"]); mc=cfg["models"]
    models={
      "LR":Pipeline([("scaler",StandardScaler()),("classifier",LogisticRegression(penalty="l2",solver="lbfgs",max_iter=5000,class_weight="balanced",random_state=seed))]),
      "RF":RandomForestClassifier(n_estimators=int(mc["rf_n_estimators"]),max_features=mc["rf_max_features"],class_weight="balanced",random_state=seed,n_jobs=-1),
      "LightGBM":LGBMClassifier(n_estimators=int(mc["lgbm_n_estimators"]),learning_rate=float(mc["lgbm_learning_rate"]),num_leaves=int(mc["lgbm_num_leaves"]),subsample=float(mc["lgbm_subsample"]),colsample_bytree=float(mc["lgbm_colsample_bytree"]),objective="binary",class_weight="balanced",random_state=seed,n_jobs=-1,verbosity=-1),
    }
    return {k:v for k,v in models.items() if k in mc["methods"]}

def classification_metrics(y_true: Iterable[int], prob: Iterable[float], threshold: float=.5) -> dict:
    y=np.asarray(y_true,dtype=int);p=np.asarray(prob,dtype=float);pred=(p>=threshold).astype(int);tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"Threshold":float(threshold),"ROC_AUC":roc_auc_score(y,p),"Accuracy":accuracy_score(y,pred),"Balanced_accuracy":balanced_accuracy_score(y,pred),"Sensitivity":recall_score(y,pred,zero_division=0),"Specificity":tn/(tn+fp) if tn+fp else np.nan,"Precision":precision_score(y,pred,zero_division=0),"F1":f1_score(y,pred,zero_division=0),"MCC":matthews_corrcoef(y,pred),"Brier_score":brier_score_loss(y,p),"TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp)}

def fit_predict_panels(X_train:pd.DataFrame,y_train:Iterable[int],X_eval:pd.DataFrame,y_eval:Iterable[int],panels:Dict[str,List[str]],models:dict,dataset:str):
    predictions=[];fitted={};y_eval=np.asarray(y_eval,dtype=int)
    for panel_name,genes in panels.items():
      missing=(set(genes)-set(X_train.columns))|(set(genes)-set(X_eval.columns))
      if missing: raise ValueError(f"Missing genes for {panel_name}: {sorted(missing)}")
      for model_name,estimator in models.items():
        model=clone(estimator);model.fit(X_train[genes],np.asarray(y_train,dtype=int));prob=model.predict_proba(X_eval[genes])[:,1]
        predictions.extend({"Dataset":dataset,"Feature_selection_method":panel_name,"Classifier":model_name,"k":len(genes),"Sample_ID":str(sid),"True_Label":int(label),"Probability":float(pr)} for sid,label,pr in zip(X_eval.index,y_eval,prob));fitted[(panel_name,model_name)]=model
    return pd.DataFrame(predictions),fitted

def metrics_from_predictions(predictions:pd.DataFrame,thresholds:pd.DataFrame|None=None,default_threshold:float=.5,threshold_source:str="Default 0.5") -> pd.DataFrame:
    rows=[]
    for keys,g in predictions.groupby(["Dataset","Feature_selection_method","Classifier","k"],sort=True):
      threshold=default_threshold;source=threshold_source
      if thresholds is not None:
        m=thresholds[(thresholds.Feature_selection_method==keys[1])&(thresholds.Classifier==keys[2])&(thresholds.k==keys[3])]
        if not m.empty: threshold=float(m.iloc[0].Threshold);source=str(m.iloc[0].Rule)
      rows.append({"Dataset":keys[0],"Feature_selection_method":keys[1],"Classifier":keys[2],"k":keys[3],"Threshold_source":source,**classification_metrics(g.True_Label,g.Probability,threshold)})
    return pd.DataFrame(rows)

def evaluate_panels(X_train,y_train,X_eval,y_eval,panels,models,dataset,threshold=.5):
    pred,fitted=fit_predict_panels(X_train,y_train,X_eval,y_eval,panels,models,dataset);metrics=metrics_from_predictions(pred,default_threshold=threshold);return metrics,pred,fitted

def bootstrap_auc_ci(y_true,prob,iterations=2000,random_state=42,alpha=.05):
    y=np.asarray(y_true,dtype=int);p=np.asarray(prob,dtype=float);pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0);rng=np.random.default_rng(random_state);values=[]
    for _ in range(iterations):
      idx=np.concatenate([rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)]);values.append(roc_auc_score(y[idx],p[idx]))
    return tuple(np.quantile(values,[alpha/2,1-alpha/2]).astype(float))

def add_auc_confidence_intervals(metrics,predictions,iterations,seed):
    out=metrics.copy();lows=[];highs=[]
    for row in out.itertuples(index=False):
      group=predictions[(predictions.Dataset==row.Dataset)&(predictions.Feature_selection_method==row.Feature_selection_method)&(predictions.Classifier==row.Classifier)&(predictions.k==row.k)];lo,hi=bootstrap_auc_ci(group.True_Label,group.Probability,iterations,seed);lows.append(lo);highs.append(hi)
    out["AUC_CI_low"]=lows;out["AUC_CI_high"]=highs;return out

def summarize_cv(metrics):
    value=[c for c in ["ROC_AUC","Accuracy","Balanced_accuracy","Sensitivity","Specificity","Precision","F1","MCC","Brier_score"] if c in metrics]
    agg=metrics.groupby(["k","Feature_selection_method","Classifier"])[value].agg(["mean","std"]).reset_index();agg.columns=["_".join([str(x) for x in col if x]).rstrip("_") if isinstance(col,tuple) else col for col in agg.columns];return agg

def calibration_table(predictions,n_bins=10):
    rows=[]
    for keys,g in predictions.groupby(["Dataset","Feature_selection_method","Classifier","k"]):
      frac,mean=calibration_curve(g.True_Label,g.Probability,n_bins=n_bins,strategy="quantile");rows.extend({"Dataset":keys[0],"Feature_selection_method":keys[1],"Classifier":keys[2],"k":keys[3],"Mean_predicted_probability":float(m),"Observed_fraction_positive":float(f)} for m,f in zip(mean,frac))
    return pd.DataFrame(rows)
