
from __future__ import annotations
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _save_both(fig, stem: Path, dpi: int = 600) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(stem.with_suffix('.png'), dpi=dpi, bbox_inches='tight')
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def run_weight_sensitivity(
    project_dir: str | Path,
    *,
    primary_k: int,
    alphas: tuple[float, ...] = (0.25, 0.50, 0.75),
    folds: int = 5,
    random_state: int = 42,
) -> Path:
    """Run rank-weight sensitivity at the configured primary panel size.

    This reads fold-specific and full-development NIBFS ranking tables produced
    by the core pipeline. It never overwrites core outputs.
    """
    project_dir = Path(project_dir).resolve()
    tables = project_dir / 'results' / 'main' / 'tables'
    output = project_dir / 'results' / 'main' / 'sensitivity' / 'rank_weight'
    output.mkdir(parents=True, exist_ok=True)

    def select_panel(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
        d = df.copy()
        d['Weighted_Borda_score'] = alpha * d['Borda_stat'] + (1 - alpha) * d['Borda_topo']
        return d.sort_values(
            ['Weighted_Borda_score', 'Rank_stat', 'Gene'],
            ascending=[False, True, True],
        ).head(primary_k).copy()

    panel_rows: list[dict] = []
    selected_genes: set[str] = set()
    for alpha in alphas:
        for fold in range(1, folds + 1):
            rank_path = tables / f'fold{fold}_NIBFS_ranking.csv'
            if not rank_path.exists():
                raise FileNotFoundError(rank_path)
            panel = select_panel(pd.read_csv(rank_path), alpha)
            selected_genes.update(panel['Gene'])
            for rank, row in enumerate(panel.itertuples(index=False), 1):
                panel_rows.append({
                    'Alpha_statistical': alpha,
                    'Alpha_topological': 1 - alpha,
                    'Data_scope': f'Fold {fold}',
                    'Validation_fold': fold,
                    'Rank': rank,
                    'Gene': row.Gene,
                    'Weighted_Borda_score': row.Weighted_Borda_score,
                    'Rank_stat': row.Rank_stat,
                    'Rank_topo': row.Rank_topo,
                })
        full = select_panel(pd.read_csv(tables / 'full_training_NIBFS_ranking.csv'), alpha)
        selected_genes.update(full['Gene'])
        for rank, row in enumerate(full.itertuples(index=False), 1):
            panel_rows.append({
                'Alpha_statistical': alpha,
                'Alpha_topological': 1 - alpha,
                'Data_scope': 'Full development',
                'Validation_fold': np.nan,
                'Rank': rank,
                'Gene': row.Gene,
                'Weighted_Borda_score': row.Weighted_Borda_score,
                'Rank_stat': row.Rank_stat,
                'Rank_topo': row.Rank_topo,
            })

    panels = pd.DataFrame(panel_rows)
    panels.to_csv(output / f'weight_sensitivity_selected_panels_k{primary_k}.csv', index=False)

    pair_rows: list[dict] = []
    for alpha in alphas:
        sets = {
            fold: set(panels[(panels.Alpha_statistical == alpha) & (panels.Validation_fold == fold)].Gene)
            for fold in range(1, folds + 1)
        }
        for a, b in combinations(range(1, folds + 1), 2):
            inter, union = len(sets[a] & sets[b]), len(sets[a] | sets[b])
            pair_rows.append({
                'Alpha_statistical': alpha, 'Alpha_topological': 1-alpha,
                'Fold_1': a, 'Fold_2': b, 'Intersection': inter,
                'Union': union, 'Jaccard': inter / union,
            })
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(output / f'weight_sensitivity_pairwise_jaccard_k{primary_k}.csv', index=False)
    stability = pairs.groupby(['Alpha_statistical', 'Alpha_topological']).Jaccard.agg(
        ['mean', 'std', 'min', 'max']
    ).reset_index()
    stability.columns = [
        'Alpha_statistical', 'Alpha_topological', 'Mean_Jaccard',
        'SD_Jaccard', 'Minimum_Jaccard', 'Maximum_Jaccard',
    ]
    ref = set(panels[(panels.Alpha_statistical == 0.50) & (panels.Data_scope == 'Full development')].Gene)
    overlap_rows = []
    for alpha in alphas:
        genes = set(panels[(panels.Alpha_statistical == alpha) & (panels.Data_scope == 'Full development')].Gene)
        overlap_rows.append({
            'Alpha_statistical': alpha, 'Alpha_topological': 1-alpha,
            'Full_panel_overlap_with_alpha_0.50': len(genes & ref),
            'Full_panel_Jaccard_with_alpha_0.50': len(genes & ref) / len(genes | ref),
        })
    stability = stability.merge(pd.DataFrame(overlap_rows), on=['Alpha_statistical', 'Alpha_topological'])
    stability.to_csv(output / f'weight_sensitivity_stability_summary_k{primary_k}.csv', index=False)

    expr_path = tables / 'harmonized_expression_matrix.csv.gz'
    folds_path = tables / 'fold_assignments.csv'
    usecols = ['GSM_ID'] + sorted(selected_genes)
    X = pd.read_csv(expr_path, usecols=usecols).set_index('GSM_ID')
    fold_table = pd.read_csv(folds_path).set_index('GSM_ID')
    common = fold_table.index.intersection(X.index)
    X, fold_table = X.loc[common], fold_table.loc[common]

    metric_rows, pred_rows = [], []
    for alpha in alphas:
        for fold in range(1, folds + 1):
            genes = panels[(panels.Alpha_statistical == alpha) & (panels.Validation_fold == fold)].sort_values('Rank').Gene.tolist()
            va = (fold_table.Validation_fold == fold).to_numpy()
            tr = ~va
            ytr = fold_table.Label_binary.iloc[tr].to_numpy(int)
            yva = fold_table.Label_binary.iloc[va].to_numpy(int)
            model = Pipeline([
                ('scale', StandardScaler()),
                ('model', LogisticRegression(
                    solver='lbfgs', class_weight='balanced', max_iter=5000,
                    random_state=random_state,
                )),
            ])
            model.fit(X.loc[:, genes].iloc[tr], ytr)
            prob = model.predict_proba(X.loc[:, genes].iloc[va])[:, 1]
            pred = (prob >= 0.5).astype(int)
            metric_rows.append({
                'Alpha_statistical': alpha, 'Alpha_topological': 1-alpha,
                'Validation_fold': fold, 'Classifier': 'LR',
                'ROC_AUC': roc_auc_score(yva, prob),
                'Accuracy': accuracy_score(yva, pred),
                'Balanced_accuracy': balanced_accuracy_score(yva, pred),
                'Sensitivity': recall_score(yva, pred, pos_label=1),
                'Specificity': recall_score(yva, pred, pos_label=0),
                'Precision': precision_score(yva, pred, zero_division=0),
                'F1': f1_score(yva, pred),
                'MCC': matthews_corrcoef(yva, pred),
                'Brier_score': brier_score_loss(yva, prob),
            })
            for sid, truth, probability in zip(X.index[va], yva, prob):
                pred_rows.append({
                    'Alpha_statistical': alpha, 'Validation_fold': fold,
                    'Classifier': 'LR', 'GSM_ID': sid,
                    'True_Label': truth, 'Probability': probability,
                })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / f'weight_sensitivity_cv_metrics_by_fold_k{primary_k}.csv', index=False)
    pd.DataFrame(pred_rows).to_csv(output / f'weight_sensitivity_cv_predictions_k{primary_k}.csv', index=False)
    value_cols = ['ROC_AUC','Accuracy','Balanced_accuracy','Sensitivity','Specificity','Precision','F1','MCC','Brier_score']
    summary = metrics.groupby(['Alpha_statistical','Alpha_topological','Classifier'])[value_cols].agg(['mean','std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(output / f'weight_sensitivity_cv_summary_k{primary_k}.csv', index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].errorbar(stability.Alpha_statistical, stability.Mean_Jaccard, yerr=stability.SD_Jaccard, marker='o', capsize=3)
    axes[0].set_xlabel('Statistical-rank weight')
    axes[0].set_ylabel('Mean pairwise Jaccard')
    axes[0].set_title('A. Stability')
    auc = summary.sort_values('Alpha_statistical')
    axes[1].errorbar(auc.Alpha_statistical, auc.ROC_AUC_mean, yerr=auc.ROC_AUC_std, marker='o', capsize=3)
    axes[1].set_xlabel('Statistical-rank weight')
    axes[1].set_ylabel('Mean LR ROC-AUC')
    axes[1].set_title('B. CV discrimination')
    axes[2].bar(stability.Alpha_statistical.astype(str), stability['Full_panel_overlap_with_alpha_0.50'])
    axes[2].set_xlabel('Statistical-rank weight')
    axes[2].set_ylabel(f'Overlap with alpha=0.50 (of {primary_k})')
    axes[2].set_title('C. Frozen-panel overlap')
    _save_both(fig, output / f'Figure_S_weight_sensitivity_k{primary_k}')
    return output
