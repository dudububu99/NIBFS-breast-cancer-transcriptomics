
from __future__ import annotations
from pathlib import Path
from itertools import combinations
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from .feature_selection import pairwise_jaccard_summary, select_top_k
from .modeling import create_models, fit_predict_panels, metrics_from_predictions
from .workflow import rankings, panels

RANK_COLUMNS = {
    'NIBFS': 'Rank_NIBFS', 'DEG-only': 'Rank_stat',
    'mRMR': 'Selection_Order', 'LASSO': 'Rank_LASSO',
}


def run_repeated_stability(
    X: pd.DataFrame,
    y: np.ndarray,
    degree: pd.DataFrame,
    cfg: dict,
    output_dir: str | Path,
    *,
    repeats: int = 10,
    folds: int = 5,
    include_lr_prediction: bool = False,
) -> Path:
    """Optional robustness analysis. It never writes into core result folders."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    k = int(cfg['project']['final_k'])
    base_seed = int(cfg['project']['random_state'])
    panel_rows, pair_rows, frequency_rows, metric_rows, prediction_rows = [], [], [], [], []

    for repeat in range(1, repeats + 1):
        print(
            f'Repeated stability: repeat {repeat}/{repeats}',
            flush=True,
        )
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=base_seed + repeat - 1)
        method_sets = {method: {} for method in cfg['feature_selection']['methods']}
        for fold, (tr, va) in enumerate(splitter.split(X, y), 1):
            ranking_tables = rankings(X.iloc[tr], y[tr], degree, cfg)
            selected = panels(ranking_tables, k)
            for method, genes in selected.items():
                method_sets[method][fold] = genes
                panel_rows.extend({
                    'Repeat': repeat, 'Fold': fold, 'Method': method,
                    'k': k, 'Selection_rank': rank, 'Gene': gene,
                } for rank, gene in enumerate(genes, 1))
            if include_lr_prediction:
                local_cfg = copy.deepcopy(cfg)
                local_cfg['models']['methods'] = ['LR']
                pred, _ = fit_predict_panels(X.iloc[tr], y[tr], X.iloc[va], y[va], selected, create_models(local_cfg), f'Repeat {repeat} fold {fold}')
                pred['Repeat'], pred['Fold'] = repeat, fold
                prediction_rows.append(pred)
                met = metrics_from_predictions(pred, default_threshold=float(cfg['models']['default_decision_threshold']))
                met['Repeat'], met['Fold'] = repeat, fold
                metric_rows.append(met)

        for method, fold_panels in method_sets.items():
            pairwise, frequency, summary = pairwise_jaccard_summary(fold_panels, method, k)
            pairwise['Repeat'] = repeat
            frequency['Repeat'] = repeat
            frequency['Method'] = method
            frequency['k'] = k
            pair_rows.append(pairwise)
            frequency_rows.append(frequency)

    panels_df = pd.DataFrame(panel_rows)
    pairs_df = pd.concat(pair_rows, ignore_index=True)
    frequency_df = pd.concat(frequency_rows, ignore_index=True)
    repeat_summary = pairs_df.groupby(['Repeat','Method','k']).Jaccard.agg(['mean','std','min','max']).reset_index()
    repeat_summary.columns = ['Repeat','Method','k','Mean_Jaccard','SD_Jaccard','Minimum_Jaccard','Maximum_Jaccard']
    overall = repeat_summary.groupby(['Method','k']).Mean_Jaccard.agg(['mean','std','median','min','max']).reset_index()
    overall.columns = ['Method','k','Mean_over_repeats','SD_over_repeats','Median_over_repeats','Minimum_repeat','Maximum_repeat']

    panels_df.to_csv(output/'repeated_selected_panels.csv', index=False)
    pairs_df.to_csv(output/'repeated_pairwise_jaccard.csv', index=False)
    frequency_df.to_csv(output/'repeated_gene_frequency.csv', index=False)
    repeat_summary.to_csv(output/'repeated_stability_per_repeat.csv', index=False)
    overall.to_csv(output/'repeated_stability_summary.csv', index=False)

    if metric_rows:
        metrics = pd.concat(metric_rows, ignore_index=True)
        predictions = pd.concat(prediction_rows, ignore_index=True)
        metrics.to_csv(output/'repeated_LR_metrics_by_fold.csv', index=False)
        predictions.to_csv(output/'repeated_LR_predictions.csv', index=False)
        metrics.groupby(['Repeat','Feature_selection_method','Classifier','k']).ROC_AUC.mean().reset_index().to_csv(
            output/'repeated_LR_auc_per_repeat.csv', index=False
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    methods = sorted(repeat_summary.Method.unique(), key=lambda x: (x != 'NIBFS', x))
    ax.boxplot([repeat_summary[repeat_summary.Method == m].Mean_Jaccard for m in methods], labels=methods, showmeans=True)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel('Mean Jaccard per repeat')
    ax.set_title(f'Optional repeated {folds}-fold stability ({repeats} repeats, k={k})')
    fig.tight_layout()
    fig.savefig(output/'figure_repeated_stability.png', dpi=600, bbox_inches='tight')
    fig.savefig(output/'figure_repeated_stability.pdf', bbox_inches='tight')
    plt.close(fig)
    return output
