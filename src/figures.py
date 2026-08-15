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
from matplotlib.lines import Line2D
import networkx as nx

from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
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


def _save(fig,path,dpi=600):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(path,dpi=dpi,bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"),bbox_inches="tight");plt.close(fig)

def plot_pca_before_after(before, after, variance, output):
    """
    Plot PCA before and after harmonization.

    Visual encoding
    ---------------
    Color  : GEO accession
    Marker : sample class
    Axes   : PC1 and PC2 with explained-variance percentages

    Expected long-format variance columns:
    Stage, Component, Explained_variance_ratio.
    """
    required_pca_columns = {"PC1", "PC2", "GEO_ID", "Label", "Stage"}
    required_variance_columns = {
        "Stage",
        "Component",
        "Explained_variance_ratio",
    }

    for table_name, table in [
        ("before", before),
        ("after", after),
    ]:
        missing = required_pca_columns.difference(table.columns)
        if missing:
            raise ValueError(
                f"{table_name} PCA table is missing columns: "
                f"{sorted(missing)}"
            )

    missing_variance = required_variance_columns.difference(
        variance.columns
    )
    if missing_variance:
        raise ValueError(
            "PCA variance table is missing columns: "
            f"{sorted(missing_variance)}"
        )

    before = before.copy()
    after = after.copy()
    variance = variance.copy()

    before["GEO_ID"] = before["GEO_ID"].astype(str)
    after["GEO_ID"] = after["GEO_ID"].astype(str)
    before["Label"] = before["Label"].astype(str)
    after["Label"] = after["Label"].astype(str)

    all_cohorts = sorted(
        set(before["GEO_ID"]).union(after["GEO_ID"])
    )
    all_labels = sorted(
        set(before["Label"]).union(after["Label"])
    )

    color_map = {
        cohort: plt.get_cmap("tab20")(index % 20)
        for index, cohort in enumerate(all_cohorts)
    }

    preferred_markers = {
        "Cancer": "o",
        "Tumor": "o",
        "Normal": "^",
    }
    fallback_markers = ["s", "D", "P", "X", "v", "<", ">"]
    marker_map = {}
    fallback_index = 0
    for label in all_labels:
        if label in preferred_markers:
            marker_map[label] = preferred_markers[label]
        else:
            marker_map[label] = fallback_markers[
                fallback_index % len(fallback_markers)
            ]
            fallback_index += 1

    def stage_variance(stage_name):
        matched = variance[
            variance["Stage"].astype(str).eq(str(stage_name))
        ].copy()

        if matched.empty:
            keyword = (
                "before"
                if "before" in str(stage_name).lower()
                else "after"
            )
            matched = variance[
                variance["Stage"]
                .astype(str)
                .str.lower()
                .str.contains(keyword, regex=False)
            ].copy()

        if matched.empty:
            raise ValueError(
                f"No PCA variance rows found for stage {stage_name!r}. "
                "Available stages: "
                f"{variance['Stage'].drop_duplicates().tolist()}"
            )

        component_index = (
            matched["Component"]
            .astype(str)
            .str.upper()
        )
        values = matched.set_index(
            component_index
        )["Explained_variance_ratio"]

        if not {"PC1", "PC2"}.issubset(values.index):
            raise ValueError(
                f"PC1/PC2 variance values are incomplete for "
                f"stage {stage_name!r}."
            )

        pc1 = float(values.loc["PC1"])
        pc2 = float(values.loc["PC2"])

        if abs(pc1) <= 1:
            pc1 *= 100
        if abs(pc2) <= 1:
            pc2 *= 100

        return pc1, pc2

    before_stage = str(before["Stage"].iloc[0])
    after_stage = str(after["Stage"].iloc[0])

    before_pc1, before_pc2 = stage_variance(before_stage)
    after_pc1, after_pc2 = stage_variance(after_stage)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 6.5),
        constrained_layout=False,
    )

    panels = [
        (
            axes[0],
            before,
            before_stage,
            before_pc1,
            before_pc2,
        ),
        (
            axes[1],
            after,
            after_stage,
            after_pc1,
            after_pc2,
        ),
    ]

    for ax, table, stage_name, pc1_percent, pc2_percent in panels:
        for cohort in all_cohorts:
            for label in all_labels:
                subset = table[
                    table["GEO_ID"].eq(cohort)
                    & table["Label"].eq(label)
                ]
                if subset.empty:
                    continue

                ax.scatter(
                    subset["PC1"],
                    subset["PC2"],
                    s=30,
                    alpha=0.78,
                    color=color_map[cohort],
                    marker=marker_map[label],
                    edgecolors="none",
                )

        ax.set_xlabel(f"PC1 ({pc1_percent:.2f}%)")
        ax.set_ylabel(f"PC2 ({pc2_percent:.2f}%)")
        ax.set_title(stage_name, fontsize=12)
        ax.grid(alpha=0.20)

    cohort_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color_map[cohort],
            markeredgecolor="none",
            markersize=7,
            label=cohort,
        )
        for cohort in all_cohorts
    ]

    class_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_map[label],
            linestyle="",
            color="black",
            markerfacecolor="black",
            markersize=7,
            label=label,
        )
        for label in all_labels
    ]

    cohort_legend = axes[1].legend(
        handles=cohort_handles,
        title="GEO cohort (color)",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    axes[1].add_artist(cohort_legend)

    axes[1].legend(
        handles=class_handles,
        title="Class (marker)",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.42),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )

    fig.suptitle(
        "Principal-component analysis before and after harmonization",
        fontsize=14,
        y=1.01,
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.80,
        bottom=0.12,
        top=0.88,
        wspace=0.24,
    )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight")
    plt.close(fig)

def plot_expression_distributions(summary,output):
    stages=list(summary.Stage.drop_duplicates());cohorts=sorted(summary.GEO_ID.unique());fig,axes=plt.subplots(2,len(stages),figsize=(7*len(stages),9),squeeze=False)
    for j,stage in enumerate(stages):
      df=summary[summary.Stage==stage];axes[0,j].boxplot([df[df.GEO_ID==g].Median for g in cohorts],labels=cohorts,showfliers=False);axes[0,j].tick_params(axis='x',rotation=60);axes[0,j].set_title(f"Sample medians - {stage}");axes[0,j].set_ylabel("Expression")
      axes[1,j].boxplot([df[df.GEO_ID==g].IQR for g in cohorts],labels=cohorts,showfliers=False);axes[1,j].tick_params(axis='x',rotation=60);axes[1,j].set_title(f"Sample IQR - {stage}");axes[1,j].set_ylabel("IQR")
    _save(fig,output)

def plot_missing_values(summary,output):
    fig,ax=plt.subplots(figsize=(8,4.5));ax.bar(summary.Stage,summary.Missing_percent);ax.set_ylabel("Missing values (%)");ax.set_title("Missing-value audit by preprocessing stage");ax.tick_params(axis='x',rotation=25);_save(fig,output)

def plot_correlation_heatmap(corr,metadata,output):
    order=np.lexsort((metadata.Label.astype(str),metadata.GEO_ID.astype(str)));mat=corr.iloc[order,order].to_numpy();fig,ax=plt.subplots(figsize=(9,8));im=ax.imshow(mat,aspect="auto",vmin=np.nanquantile(mat,.02),vmax=1,cmap="viridis");ax.set_title("Sample-wise Pearson correlation after harmonization");ax.set_xlabel("Samples ordered by cohort and class");ax.set_ylabel("Samples ordered by cohort and class");fig.colorbar(im,ax=ax,label="Pearson correlation");_save(fig,output)

def plot_outlier_audit(qc,output):
    fig,ax=plt.subplots(figsize=(7,5.5));normal=~qc.Audit_flag;ax.scatter(qc.loc[normal,"PCA_distance_robust_z"],qc.loc[normal,"Mean_correlation_robust_z"],s=20,alpha=.55,label="Retained / not flagged");ax.scatter(qc.loc[~normal,"PCA_distance_robust_z"],qc.loc[~normal,"Mean_correlation_robust_z"],s=40,marker="x",label="Audit flag");ax.axhline(-3.5,ls="--",lw=1);ax.axvline(3.5,ls="--",lw=1);ax.set_xlabel("Robust z-score of PCA distance");ax.set_ylabel("Robust z-score of mean correlation");ax.set_title("Combined sample quality-control audit");ax.legend(frameon=False);_save(fig,output)

def plot_volcano(limma,selected,output):
    selected = list(selected)
    df=limma.copy();df["minus_log10_FDR"]=-np.log10(df.FDR.clip(lower=1e-300));sig=(df.FDR<=.05)&(df.logFC.abs()>1);sel=df.Gene.isin(selected)
    fig,ax=plt.subplots(figsize=(8,6));ax.scatter(df.loc[~sig,"logFC"],df.loc[~sig,"minus_log10_FDR"],s=7,alpha=.25,label="Not DEG");ax.scatter(df.loc[sig,"logFC"],df.loc[sig,"minus_log10_FDR"],s=9,alpha=.55,label="Descriptive DEG");ax.scatter(df.loc[sel,"logFC"],df.loc[sel,"minus_log10_FDR"],s=42,facecolors="none",edgecolors="black",linewidths=.8,label=f"Frozen NIBFS top-{len(selected)}")
    for r in df[sel].sort_values("Stat_score",ascending=False).head(12).itertuples(): ax.annotate(r.Gene,(r.logFC,r.minus_log10_FDR),fontsize=7,xytext=(3,3),textcoords="offset points")
    ax.axvline(-1,ls="--",lw=1);ax.axvline(1,ls="--",lw=1);ax.axhline(-np.log10(.05),ls="--",lw=1);ax.set_xlabel("log2 fold change (Cancer - Normal)");ax.set_ylabel("-log10(FDR)");ax.set_title("Differential-expression landscape in the model-development set");ax.legend(frameon=False,fontsize=8);_save(fig,output)

def plot_clustered_heatmap(X, metadata, genes, output, title):
    """Publication heatmap with sample dendrogram and class/GEO annotation bars.

    Samples are rows and genes are columns, matching the compact journal layout.
    """
    genes = [g for g in genes if g in X.columns]
    if not genes:
        raise ValueError("No requested genes are present in the expression matrix")
    meta = metadata.copy()
    if "GSM_ID" in meta.columns:
        meta = meta.set_index("GSM_ID").loc[X.index].reset_index()
    Z = (X[genes] - X[genes].mean()) / X[genes].std(ddof=1).replace(0, 1)
    if len(Z) > 1:
        sample_link = linkage(Z.to_numpy(), method="average", metric="correlation")
        sample_order = leaves_list(sample_link)
    else:
        sample_link = None
        sample_order = np.arange(len(Z))
    if len(genes) > 1:
        gene_order = leaves_list(linkage(Z.T.to_numpy(), method="average", metric="correlation"))
    else:
        gene_order = np.arange(len(genes))
    ordered = Z.iloc[sample_order, gene_order]
    ordered_meta = meta.iloc[sample_order].reset_index(drop=True)
    ordered_genes = np.array(genes)[gene_order]

    class_values = ordered_meta["Label"].map({"Normal": 0, "Cancer": 1}).fillna(-1).to_numpy()[:, None]
    geos = sorted(ordered_meta["GEO_ID"].astype(str).unique())
    geo_map = {g: i for i, g in enumerate(geos)}
    geo_values = ordered_meta["GEO_ID"].astype(str).map(geo_map).to_numpy()[:, None]

    height = max(8, min(18, 5 + len(ordered) * 0.035))
    fig = plt.figure(figsize=(14.5, height))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.35, 0.16, 0.16, 8.5, 1.7], wspace=0.03)
    ax_den = fig.add_subplot(gs[0, 0])
    ax_cls = fig.add_subplot(gs[0, 1])
    ax_geo = fig.add_subplot(gs[0, 2])
    ax = fig.add_subplot(gs[0, 3])
    ax_leg = fig.add_subplot(gs[0, 4])

    if sample_link is not None:
        dendrogram(sample_link, orientation="left", no_labels=True, color_threshold=0, above_threshold_color="black", ax=ax_den)
    ax_den.axis("off")
    ax_cls.imshow(class_values, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=0, vmax=1)
    ax_cls.set_xticks([0], ["Class"], rotation=90, fontsize=8)
    ax_cls.set_yticks([])
    ax_geo.imshow(geo_values, aspect="auto", interpolation="nearest", cmap=plt.get_cmap("tab20", max(len(geos), 1)), vmin=0, vmax=max(len(geos)-1, 1))
    ax_geo.set_xticks([0], ["GEO"], rotation=90, fontsize=8)
    ax_geo.set_yticks([])
    im = ax.imshow(ordered.to_numpy(), aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=-3, vmax=3)
    ax.set_xticks(np.arange(len(ordered_genes)), ordered_genes, rotation=90, fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel("Frozen candidate genes")
    ax.set_ylabel("Hierarchically clustered samples")
    ax.set_title(title, fontsize=12)

    ax_leg.axis("off")
    class_handles = [
        Line2D([0], [0], marker="s", linestyle="", label="Normal", markerfacecolor=plt.get_cmap("coolwarm")(0.0), markeredgecolor="none", markersize=8),
        Line2D([0], [0], marker="s", linestyle="", label="Cancer", markerfacecolor=plt.get_cmap("coolwarm")(1.0), markeredgecolor="none", markersize=8),
    ]
    geo_handles = [
        Line2D([0], [0], marker="s", linestyle="", label=g, markerfacecolor=plt.get_cmap("tab20", max(len(geos),1))(i), markeredgecolor="none", markersize=7)
        for i, g in enumerate(geos)
    ]
    leg1 = ax_leg.legend(handles=class_handles, title="Class", loc="upper left", frameon=False, fontsize=8)
    ax_leg.add_artist(leg1)
    ax_leg.legend(handles=geo_handles, title="GEO dataset", loc="center left", frameon=False, fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cbar.set_label("Gene-wise z-score")
    _save(fig, output)

def plot_stability_composite(pairwise,frequency,k,output):
    df=pairwise[pairwise.k==k];methods=sorted(df.Method.unique(),key=lambda x:(x!="NIBFS",x));freq=frequency[frequency.k==k].copy();genes=freq.groupby("Gene").Fold_Frequency.max().sort_values(ascending=False).head(35).index;mat=freq[freq.Gene.isin(genes)].pivot_table(index="Gene",columns="Method",values="Fold_Frequency",fill_value=0).reindex(index=genes,columns=methods)
    fig,axes=plt.subplots(1,2,figsize=(13,6),gridspec_kw={"width_ratios":[1,1.15]});axes[0].boxplot([df[df.Method==m].Jaccard for m in methods],labels=methods,showmeans=True);axes[0].set_ylim(0,1.03);axes[0].set_ylabel("Pairwise Jaccard");axes[0].set_title(f"Panel stability distributions (k={k})");axes[0].tick_params(axis='x',rotation=25);im=axes[1].imshow(mat.to_numpy(),aspect="auto",vmin=0,vmax=5,cmap="Blues");axes[1].set_xticks(range(len(methods)),methods,rotation=25,ha="right");axes[1].set_yticks(range(len(mat)),mat.index,fontsize=7);axes[1].set_title("Gene recurrence across five folds");fig.colorbar(im,ax=axes[1],label="Fold frequency");_save(fig,output)

def plot_stability_heatmap(pairwise,method,k,output):
    df=pairwise[(pairwise.Method==method)&(pairwise.k==k)];folds=sorted(set(df.Fold_1)|set(df.Fold_2));mat=np.eye(len(folds));idx={f:i for i,f in enumerate(folds)}
    for r in df.itertuples():mat[idx[r.Fold_1],idx[r.Fold_2]]=mat[idx[r.Fold_2],idx[r.Fold_1]]=r.Jaccard
    fig,ax=plt.subplots(figsize=(5.5,5));im=ax.imshow(mat,vmin=0,vmax=1,cmap="Blues");ax.set_xticks(range(len(folds)),[f"F{f}" for f in folds]);ax.set_yticks(range(len(folds)),[f"F{f}" for f in folds]);
    for i in range(len(folds)):
      for j in range(len(folds)):ax.text(j,i,f"{mat[i,j]:.2f}",ha="center",va="center",fontsize=8)
    ax.set_title(f"{method} pairwise Jaccard (k={k})");fig.colorbar(im,ax=ax);_save(fig,output)

def plot_cv_performance(metrics,output,k=20):
    summary=metrics[metrics.k==k].groupby(["Feature_selection_method","Classifier"]).ROC_AUC.agg(["mean","std"]).reset_index().sort_values("mean");labels=(summary.Feature_selection_method+" + "+summary.Classifier).tolist();y=np.arange(len(summary));fig,ax=plt.subplots(figsize=(9,7));ax.barh(y,summary["mean"],xerr=summary["std"],capsize=3);ax.set_yticks(y,labels);ax.set_xlabel("Mean cross-validated ROC-AUC");ax.set_xlim(max(.5,summary["mean"].min()-.04),1.005);ax.set_title(f"Cross-validation classification performance (k={k})");_save(fig,output)

def plot_roc_on_axis(ax,predictions,title):
    for key,g in predictions.groupby("Classifier",sort=True):fpr,tpr,_=roc_curve(g.True_Label,g.Probability);auc=roc_auc_score(g.True_Label,g.Probability);ax.plot(fpr,tpr,lw=2,label=f"{key} (AUC={auc:.3f})")
    ax.plot([0,1],[0,1],ls="--",lw=1);ax.set_xlabel("False positive rate");ax.set_ylabel("True positive rate");ax.set_title(title);ax.legend(frameon=False,fontsize=8)

def plot_roc(predictions,output,title):
    fig,ax=plt.subplots(figsize=(6.5,5.5));plot_roc_on_axis(ax,predictions,title);_save(fig,output)

def plot_calibration_on_axis(ax,calibration,title):
    for keys,g in calibration.groupby(["Feature_selection_method","Classifier"]):ax.plot(g.Mean_predicted_probability,g.Observed_fraction_positive,marker="o",label=" / ".join(keys))
    ax.plot([0,1],[0,1],ls="--",lw=1);ax.set_xlabel("Mean predicted probability");ax.set_ylabel("Observed fraction positive");ax.set_title(title);ax.legend(frameon=False,fontsize=7)

def plot_heldout_composite(predictions,calibration,output):
    fig,axes=plt.subplots(1,2,figsize=(12,5));plot_roc_on_axis(axes[0],predictions,"Post-harmonization held-out ROC");plot_calibration_on_axis(axes[1],calibration,"Held-out calibration");_save(fig,output)

def plot_sensitivity(stability,cv_summary,output,method="NIBFS"):
    fig,axes=plt.subplots(1,2,figsize=(12,5));
    for m,g in stability.groupby("Method"):g=g.sort_values("k");axes[0].plot(g.k,g.Mean_Jaccard,marker="o",label=m)
    axes[0].set_xlabel("Panel size (k)");axes[0].set_ylabel("Mean pairwise Jaccard");axes[0].set_ylim(0,1.02);axes[0].legend(frameon=False,fontsize=8);axes[0].set_title("Panel-size sensitivity of stability")
    sub=cv_summary[cv_summary.Feature_selection_method==method]
    for clf,g in sub.groupby("Classifier"):g=g.sort_values("k");axes[1].plot(g.k,g.ROC_AUC_mean,marker="o",label=clf)
    axes[1].set_xlabel("Panel size (k)");axes[1].set_ylabel("Mean CV ROC-AUC");axes[1].set_ylim(max(.5,sub.ROC_AUC_mean.min()-.03),1.01);axes[1].legend(frameon=False);axes[1].set_title(f"{method} predictive sensitivity");_save(fig,output)

def plot_rank_landscape(ranking,selected,output):
    df=ranking.copy();sel=df.Gene.isin(selected);fig,ax=plt.subplots(figsize=(7,6));ax.scatter(df.Rank_stat,df.Rank_topo,s=7,alpha=.25);ax.scatter(df.loc[sel,"Rank_stat"],df.loc[sel,"Rank_topo"],s=38,facecolors="none",edgecolors="black");
    for r in df[sel].itertuples():ax.annotate(r.Gene,(r.Rank_stat,r.Rank_topo),fontsize=7,xytext=(3,3),textcoords="offset points")
    ax.set_xscale("log");ax.set_yscale("log");ax.invert_xaxis();ax.invert_yaxis();ax.set_xlabel("Statistical rank (better toward upper right)");ax.set_ylabel("PPI topological rank");ax.set_title("NIBFS component-rank landscape");_save(fig,output)

def plot_external_validation(predictions,direction,output):
    fig,axes=plt.subplots(1,2,figsize=(12,5));plot_roc_on_axis(axes[0],predictions,"Independent validation ROC: GSE15852");consistent=direction.Direction_consistent.map({True:"Consistent",False:"Discordant"})
    for label,g in direction.groupby(consistent):axes[1].scatter(g.Training_logFC,g.External_logFC,s=35,label=label)
    axes[1].axhline(0,ls="--",lw=1);axes[1].axvline(0,ls="--",lw=1);lo=min(direction[["Training_logFC","External_logFC"]].min());hi=max(direction[["Training_logFC","External_logFC"]].max());axes[1].plot([lo,hi],[lo,hi],ls=":",lw=1)
    for r in direction.itertuples():axes[1].annotate(r.Gene,(r.Training_logFC,r.External_logFC),fontsize=7,xytext=(3,3),textcoords="offset points")
    axes[1].set_xlabel("Discovery log2FC");axes[1].set_ylabel("External log2FC");axes[1].set_title(f"Direction consistency: {direction.Direction_consistent.sum()}/{len(direction)}");axes[1].legend(frameon=False);_save(fig,output)

def ppi_graph_and_centrality(genes,edges,limma):
    genes=list(dict.fromkeys(map(str,genes)));G=nx.Graph();G.add_nodes_from(genes)
    for r in edges.itertuples(index=False):G.add_edge(r.Gene1,r.Gene2,weight=float(r.combined_score)/1000)
    degree=dict(G.degree());weighted=dict(G.degree(weight="weight"));between=nx.betweenness_centrality(G,weight=None,normalized=True);close=nx.closeness_centrality(G);component={g:i+1 for i,c in enumerate(sorted(nx.connected_components(G),key=len,reverse=True)) for g in c};logfc=limma.set_index("Gene").logFC.to_dict()
    central=pd.DataFrame({"Gene":genes,"Within_panel_degree":[degree[g] for g in genes],"Weighted_degree":[weighted[g] for g in genes],"Betweenness_centrality":[between[g] for g in genes],"Closeness_centrality":[close[g] for g in genes],"Component":[component[g] for g in genes],"Isolated":[degree[g]==0 for g in genes],"logFC":[logfc.get(g,np.nan) for g in genes]}).sort_values(["Within_panel_degree","Betweenness_centrality","Gene"],ascending=[False,False,True]);return G,central

def draw_ppi(ax,G,central,seed=42):
    pos=nx.spring_layout(G,seed=seed,weight="weight",k=.9);logfc=central.set_index("Gene").logFC.to_dict();values=[float(logfc.get(g,0)) for g in G.nodes()];vmax=max(max(abs(np.array(values))),1e-6);nodes=nx.draw_networkx_nodes(G,pos,node_size=[420+80*G.degree(g) for g in G.nodes()],node_color=values,cmap="coolwarm",vmin=-vmax,vmax=vmax,ax=ax);nx.draw_networkx_edges(G,pos,width=[1+2*G[u][v]["weight"] for u,v in G.edges()],alpha=.45,ax=ax);nx.draw_networkx_labels(G,pos,font_size=7,ax=ax);ax.axis("off");ax.set_title(f"STRING/PPI subnetwork of the frozen top-{G.number_of_nodes()}");return nodes

def plot_ppi_network(genes,edges,limma,output,seed=42):
    G,central=ppi_graph_and_centrality(genes,edges,limma);fig,ax=plt.subplots(figsize=(10,8));nodes=draw_ppi(ax,G,central,seed);fig.colorbar(nodes,ax=ax,label="Training log2FC");_save(fig,output);comps=sorted((len(c) for c in nx.connected_components(G)),reverse=True);summary=pd.DataFrame([{"Number_of_genes":G.number_of_nodes(),"Number_of_edges":G.number_of_edges(),"Network_density":nx.density(G),"Connected_components":nx.number_connected_components(G),"Largest_component_size":comps[0] if comps else 0,"Isolated_genes":";".join(sorted(nx.isolates(G)))}]);return summary,central

def top_terms(enrichment,source,n=8):return enrichment[enrichment.Database==source].sort_values("Adjusted_p_value").head(n).sort_values("minus_log10_adjusted_p")

def bar_terms(ax,df,title):
    if df.empty:ax.text(.5,.5,"No significant terms",ha="center",va="center");ax.set_title(title);return
    ax.barh(df.Term,df.minus_log10_adjusted_p);ax.set_xlabel("-log10(adjusted p)");ax.set_title(title);ax.tick_params(axis='y',labelsize=7)

def plot_enrichment(enrichment,output,top_n=20):
    fig,axes=plt.subplots(1,3,figsize=(16,6));bar_terms(axes[0],top_terms(enrichment,"GO:BP",7),"GO Biological Process");bar_terms(axes[1],top_terms(enrichment,"KEGG",7),"KEGG");bar_terms(axes[2],top_terms(enrichment,"REAC",7),"Reactome");_save(fig,output)

def plot_biological_interpretation(enrichment,genes,edges,limma,output,seed=42):
    G,central=ppi_graph_and_centrality(genes,edges,limma);fig=plt.figure(figsize=(15,10));gs=fig.add_gridspec(2,3,height_ratios=[1,1.25]);axes=[fig.add_subplot(gs[0,i]) for i in range(3)];bar_terms(axes[0],top_terms(enrichment,"GO:BP",6),"GO Biological Process");bar_terms(axes[1],top_terms(enrichment,"KEGG",6),"KEGG");bar_terms(axes[2],top_terms(enrichment,"REAC",6),"Reactome");axn=fig.add_subplot(gs[1,:]);nodes=draw_ppi(axn,G,central,seed);fig.colorbar(nodes,ax=axn,label="Training log2FC",fraction=.02);_save(fig,output)

def plot_km_forest(km, output):
    if km.empty:
        return
    parsed = km.copy()
    if not {"CI_low", "CI_high"}.issubset(parsed.columns):
        ci_col = "CI" if "CI" in parsed.columns else "95% confidence interval"
        ci = parsed[ci_col].astype(str).str.extract(r"([0-9.]+)\s*[-–,]\s*([0-9.]+)")
        parsed["CI_low"] = pd.to_numeric(ci[0], errors="coerce")
        parsed["CI_high"] = pd.to_numeric(ci[1], errors="coerce")
    parsed["Hazard_ratio"] = pd.to_numeric(parsed["Hazard_ratio"], errors="coerce")
    parsed["CI_low"] = pd.to_numeric(parsed["CI_low"], errors="coerce")
    parsed["CI_high"] = pd.to_numeric(parsed["CI_high"], errors="coerce")
    parsed = parsed.dropna(subset=["Hazard_ratio", "CI_low", "CI_high"])
    y = np.arange(len(parsed))
    fig, ax = plt.subplots(figsize=(7.5, max(5, .34 * len(parsed))))
    ax.errorbar(parsed.Hazard_ratio, y, xerr=[parsed.Hazard_ratio-parsed.CI_low, parsed.CI_high-parsed.Hazard_ratio], fmt="o", capsize=3)
    ax.axvline(1, ls="--", lw=1)
    ax.set_yticks(y, parsed.Gene)
    ax.set_xlabel("Hazard ratio (95% CI)")
    ax.set_title("Post hoc KM Plotter results (RFS)")
    _save(fig, output)

def plot_study_workflow(manifest, stability, external_metrics, output):
    primary_k = int(manifest.get("primary_k", 20))
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis("off")
    boxes = [
        ("A. Discovery", f"11 GPL570 cohorts\n{manifest['samples_total']} samples\n{manifest['cancer']} cancer / {manifest['normal']} normal"),
        ("B. Harmonization", f"HGNC probe mapping\nCommon-gene intersection\nQuantile normalization\nComBat\n{manifest['eligible_genes']} eligible genes"),
        ("C. Evaluation", f"80:20 split\n{manifest['training_samples']} development\n{manifest['heldout_samples']} held-out\n5-fold CV"),
        ("D. NIBFS", f"limma statistical rank\nSTRING degree rank\nDeterministic Borda\nFrozen top-{primary_k}"),
        ("E. Stability & models", f"NIBFS mean Jaccard\n{stability:.3f}\nLR / RF / LightGBM"),
        ("F. External & biology", f"GSE15852 cross-platform\nBest AUC {external_metrics:.3f}\nDirection consistency\nGO/KEGG/Reactome/PPI/RFS"),
    ]
    coords=[(.02,.58),(.35,.58),(.68,.58),(.18,.15),(.51,.15),(.80,.15)]
    w=.19; h=.25
    for (title,text),(x,y) in zip(boxes,coords):
        ax.add_patch(plt.Rectangle((x,y),w,h,fill=False,lw=1.5))
        ax.text(x+w/2,y+h-.04,title,ha="center",va="top",fontsize=12,fontweight="bold")
        ax.text(x+w/2,y+h/2-.02,text,ha="center",va="center",fontsize=9)
    arrows=[((.21,.705),(.35,.705)),((.54,.705),(.68,.705)),((.775,.58),(.275,.40)),((.37,.275),(.51,.275)),((.70,.275),(.80,.275))]
    for a,b in arrows:
        ax.annotate("",xy=b,xytext=a,arrowprops=dict(arrowstyle="->",lw=1.5))
    ax.set_title("Leakage-aware study workflow for stability-aware NIBFS in breast cancer transcriptomics",fontsize=17,fontweight="bold")
    ax.text(.5,.03,"Candidate diagnostic biomarker prioritization - not a clinically validated assay",ha="center",fontsize=11,fontweight="bold")
    _save(fig,output)

def plot_graphical_abstract(manifest, stability, best_auc, direction_count, output):
    primary_k = int(manifest.get("primary_k", 20))
    output = Path(output)
    fig, ax = plt.subplots(figsize=(13.28, 5.31))
    ax.axis("off")
    items = [
        ("Discovery resource", f"11 GPL570 cohorts\n{manifest['samples_total']} samples"),
        ("Harmonization", f"HGNC mapping\nQN + ComBat\n{manifest['eligible_genes']} genes"),
        ("Training-only NIBFS", "limma rank + STRING rank\nBorda aggregation"),
        ("Stable frozen panel", f"Top-{primary_k} candidates\nMean Jaccard {stability:.3f}"),
        ("Independent evidence", f"GSE15852 best AUC {best_auc:.3f}\n{direction_count}/{primary_k} directions consistent\nGO/KEGG/PPI/RFS"),
    ]
    xs = np.linspace(.02, .81, 5)
    w, h = .17, .58
    for i, ((title, text), x) in enumerate(zip(items, xs)):
        ax.add_patch(plt.Rectangle((x, .22), w, h, fill=False, lw=1.7))
        ax.text(x + w / 2, .70, title, ha="center", fontsize=11, fontweight="bold")
        ax.text(x + w / 2, .47, text, ha="center", va="center", fontsize=9.5)
        if i < 4:
            ax.annotate("", xy=(xs[i + 1], .51), xytext=(x + w, .51), arrowprops=dict(arrowstyle="->", lw=1.7))
    ax.set_title("Stability-aware NIBFS for reproducible breast cancer candidate-biomarker prioritization",fontsize=15,fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=600, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".tiff"), dpi=300, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

def plot_top_panel_barplot(final_panel, output):
    df = final_panel.sort_values("Rank_NIBFS", ascending=False).copy()
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.barh(df.Gene, df.Borda_score)
    ax.set_xlabel("NIBFS Borda score")
    ax.set_ylabel("Gene")
    ax.set_title("Final NIBFS candidate panel")
    _save(fig, output)

def plot_gene_occurrence_heatmap(fold_panels, output, method="NIBFS", k=20):
    df = fold_panels[(fold_panels.Method == method) & (fold_panels.k == k)].copy()
    if df.empty:
        return
    genes = (
        df.groupby("Gene").Fold.nunique().sort_values(ascending=False).index.astype(str).tolist()
    )
    folds = sorted(df.Fold.unique())
    mat = pd.DataFrame(0, index=genes, columns=[f"F{f}" for f in folds], dtype=int)
    for r in df.itertuples(index=False):
        mat.loc[str(r.Gene), f"F{int(r.Fold)}"] = 1
    fig, ax = plt.subplots(figsize=(7, max(6, .28 * len(mat))))
    im = ax.imshow(mat.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(mat.columns)), mat.columns)
    ax.set_yticks(range(len(mat.index)), mat.index, fontsize=7)
    ax.set_title(f"{method} gene occurrence across folds (k={k})")
    ax.set_xlabel("Validation fold")
    ax.set_ylabel("Selected gene")
    fig.colorbar(im, ax=ax, ticks=[0, 1], label="Selected")
    _save(fig, output)

def plot_preprocessing_qc_overview(distribution, missing, variance, qc, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    stages = list(distribution.Stage.drop_duplicates())
    med = distribution.groupby("Stage").Median.median().reindex(stages)
    axes[0, 0].bar(med.index, med.values)
    axes[0, 0].set_title("Median expression by stage")
    axes[0, 0].tick_params(axis="x", rotation=25)
    axes[0, 0].set_ylabel("Median expression")

    axes[0, 1].bar(missing.Stage, missing.Missing_percent)
    axes[0, 1].set_title("Missing-value audit")
    axes[0, 1].tick_params(axis="x", rotation=25)
    axes[0, 1].set_ylabel("Missing (%)")

    v = variance.pivot(index="Component", columns="Stage", values="Explained_variance_ratio")
    v.T.plot(kind="bar", ax=axes[1, 0])
    axes[1, 0].set_title("PCA variance before/after harmonization")
    axes[1, 0].set_ylabel("Explained variance ratio")
    axes[1, 0].tick_params(axis="x", rotation=0)

    normal = ~qc.Audit_flag
    axes[1, 1].scatter(qc.loc[normal, "PCA_distance_robust_z"], qc.loc[normal, "Mean_correlation_robust_z"], s=16, alpha=.5, label="Not flagged")
    axes[1, 1].scatter(qc.loc[~normal, "PCA_distance_robust_z"], qc.loc[~normal, "Mean_correlation_robust_z"], s=35, marker="x", label="Audit flag")
    axes[1, 1].axhline(-3.5, ls="--", lw=1)
    axes[1, 1].axvline(3.5, ls="--", lw=1)
    axes[1, 1].set_title("Combined sample audit")
    axes[1, 1].set_xlabel("PCA-distance robust z")
    axes[1, 1].set_ylabel("Mean-correlation robust z")
    axes[1, 1].legend(frameon=False)
    _save(fig, output)

def plot_heldout_roc_only(predictions, output):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    plot_roc_on_axis(ax, predictions, "Post-harmonization held-out ROC")
    _save(fig, output)

def plot_heldout_performance(metrics, output):
    df = metrics[(metrics.Feature_selection_method == "NIBFS") & (metrics.Evaluation_rule == "Default")].copy()
    if df.empty:
        return
    cols = ["ROC_AUC", "Accuracy", "F1", "MCC", "Sensitivity", "Specificity"]
    long = df.melt(id_vars="Classifier", value_vars=cols, var_name="Metric", value_name="Value")
    classifiers = sorted(long.Classifier.unique())
    x = np.arange(len(cols))
    width = .8 / max(1, len(classifiers))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, clf in enumerate(classifiers):
        g = long[long.Classifier == clf].set_index("Metric").reindex(cols)
        ax.bar(x + (i - (len(classifiers)-1)/2)*width, g.Value, width=width, label=clf)
    ax.set_xticks(x, cols, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Metric value")
    k_value = int(df["k"].iloc[0]) if "k" in df.columns and not df.empty else len(df)
    ax.set_title(f"Held-out performance of the frozen NIBFS top-{k_value} panel")
    ax.legend(frameon=False)
    _save(fig, output)

def plot_external_roc_only(predictions, output):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    plot_roc_on_axis(ax, predictions, "Independent external ROC: GSE15852")
    _save(fig, output)

def plot_external_direction_only(direction, output):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    consistent = direction.Direction_consistent.map({True: "Consistent", False: "Discordant"})
    for label, g in direction.groupby(consistent):
        ax.scatter(g.Training_logFC, g.External_logFC, s=38, label=label)
    ax.axhline(0, ls="--", lw=1)
    ax.axvline(0, ls="--", lw=1)
    lo = min(direction[["Training_logFC", "External_logFC"]].min())
    hi = max(direction[["Training_logFC", "External_logFC"]].max())
    ax.plot([lo, hi], [lo, hi], ls=":", lw=1)
    for r in direction.itertuples(index=False):
        ax.annotate(r.Gene, (r.Training_logFC, r.External_logFC), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Discovery log2FC")
    ax.set_ylabel("External log2FC")
    ax.set_title(f"Expression-direction consistency: {direction.Direction_consistent.sum()}/{len(direction)}")
    ax.legend(frameon=False)
    _save(fig, output)

# Clear dynamic alias for any panel size.


