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

from .hgnc import HGNCResolver

def read_string_protein_to_gene(path: str|Path,resolver:HGNCResolver)->pd.DataFrame:
    info=pd.read_csv(path,sep="\t",compression="gzip",dtype=str,low_memory=False);protein_col="#string_protein_id" if "#string_protein_id" in info.columns else "string_protein_id"
    if "preferred_name" not in info.columns: raise ValueError(f"preferred_name missing: {info.columns.tolist()}")
    out=info[[protein_col,"preferred_name"]].rename(columns={protein_col:"STRING_ID","preferred_name":"Raw_Name"}).copy();out["Gene"]=out.Raw_Name.map(resolver.resolve);return out.dropna(subset=["Gene"])[["STRING_ID","Gene"]].drop_duplicates().reset_index(drop=True)

def build_gene_edges_and_degree(links_path,protein_to_gene,eligible_genes,required_score=700,chunk_size=1_000_000):
    eligible=set(map(str,eligible_genes));mapping=protein_to_gene[protein_to_gene.Gene.isin(eligible)].copy();mapped=set(mapping.Gene);id_to_gene=dict(zip(mapping.STRING_ID,mapping.Gene));records=[]
    for chunk in pd.read_csv(links_path,sep=r"\s+",compression="gzip",chunksize=chunk_size):
      chunk=chunk[chunk.combined_score>=required_score].copy();chunk["Gene1"]=chunk.protein1.map(id_to_gene);chunk["Gene2"]=chunk.protein2.map(id_to_gene);chunk=chunk.dropna(subset=["Gene1","Gene2"]);chunk=chunk[chunk.Gene1!=chunk.Gene2]
      if chunk.empty: continue
      g1=chunk[["Gene1","Gene2"]].min(axis=1);g2=chunk[["Gene1","Gene2"]].max(axis=1);records.append(pd.DataFrame({"Gene1":g1,"Gene2":g2,"combined_score":chunk.combined_score.astype(int).to_numpy()}))
    edges=pd.concat(records,ignore_index=True).groupby(["Gene1","Gene2"],as_index=False).combined_score.max() if records else pd.DataFrame(columns=["Gene1","Gene2","combined_score"])
    neighbors={g:set() for g in eligible}
    for r in edges.itertuples(index=False): neighbors[r.Gene1].add(r.Gene2);neighbors[r.Gene2].add(r.Gene1)
    denom=max(len(mapped)-1,1);genes=sorted(eligible);degree=pd.DataFrame({"Gene":genes,"Mapped_to_STRING":[g in mapped for g in genes],"Degree":[len(neighbors[g]) for g in genes]});degree["Normalized_degree"]=degree.Degree/denom;degree["STRING_mapped_gene_count"]=len(mapped);return edges,degree

def final_panel_subnetwork(edges,genes):
    panel=set(map(str,genes));return edges[(edges.Gene1.isin(panel))&(edges.Gene2.isin(panel))].copy().reset_index(drop=True)
