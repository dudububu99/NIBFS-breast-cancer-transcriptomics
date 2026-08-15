
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import copy
import hashlib
import json
import shutil
import time
import warnings
import zipfile

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from .configuration import load_config
from .hgnc import HGNCResolver
from .workflow import (
    dirs, references, prepare_discovery, prepare_ppi, rankings, panels,
    load_external, _long_panels, _full_panels, _source_subset,
)
from .feature_selection import prepare_ppi_rank_table, select_top_k, pairwise_jaccard_summary
from .modeling import (
    create_models, fit_predict_panels, metrics_from_predictions, summarize_cv,
    add_auc_confidence_intervals, calibration_table,
)
from .statistics import derive_youden_thresholds, stability_statistical_tests, cv_auc_statistical_tests
from .ppi import final_panel_subnetwork
from .enrichment import run_enrichment, standardize_enrichment
from .reporting import (
    choose_compact_heatmap_samples, clustered_heatmap_orders, build_kan_bridge,
    write_json,
)
from .figures import (
    plot_volcano, plot_clustered_heatmap, plot_stability_composite,
    plot_gene_occurrence_heatmap, plot_cv_performance, plot_sensitivity,
    plot_rank_landscape, plot_roc, plot_heldout_composite,
    plot_heldout_roc_only, plot_heldout_performance,
    plot_external_validation, plot_external_roc_only, plot_external_direction_only,
    plot_ppi_network, plot_enrichment, plot_biological_interpretation,
    plot_km_forest, plot_study_workflow, plot_graphical_abstract,
    plot_top_panel_barplot,
)
from .kmplotter import load_kmplotter_csv
from .sensitivity import run_weight_sensitivity


@dataclass
class RuntimeTracker:
    output_csv: Path
    rows: list[dict] = field(default_factory=list)
    wall_start: float = field(default_factory=time.perf_counter)

    def start(self, name: str) -> tuple[str, float]:
        print(f'\n{"="*88}\nSTART: {name}\n{"="*88}')
        return name, time.perf_counter()

    def stop(self, token: tuple[str, float]) -> None:
        name, started = token
        seconds = time.perf_counter() - started
        self.rows.append({
            'Stage': name, 'Seconds': seconds, 'Minutes': seconds/60,
            'Finished_UTC': datetime.now(timezone.utc).isoformat(),
        })
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows).to_csv(self.output_csv, index=False)
        print(f'FINISH {name}: {seconds:.2f} seconds ({seconds/60:.2f} minutes)')

    def finish_total(self) -> None:
        seconds = time.perf_counter() - self.wall_start
        self.rows.append({
            'Stage': 'TOTAL PIPELINE WALL TIME', 'Seconds': seconds,
            'Minutes': seconds/60, 'Finished_UTC': datetime.now(timezone.utc).isoformat(),
        })
        pd.DataFrame(self.rows).to_csv(self.output_csv, index=False)


class NIBFSPipeline:
    """Clean raw-to-results NIBFS implementation. No historical outputs are loaded."""

    def __init__(self, project_dir: str | Path, config_path: str | Path):
        self.project = Path(project_dir).resolve()
        self.project.mkdir(parents=True, exist_ok=True)
        config_path = Path(config_path).resolve()
        self.cfg = load_config(config_path)
        # Copy immutable run inputs into the evidence project so the run is self-contained.
        shutil.copy2(config_path, self.project/'config.yaml')
        source_root = config_path.parent
        accession_source = source_root/'data_accession_list.csv'
        if not accession_source.exists():
            raise FileNotFoundError(accession_source)
        shutil.copy2(accession_source, self.project/'data_accession_list.csv')
        source_manual = source_root/'manual_inputs'
        target_manual = self.project/'manual_inputs'
        target_manual.mkdir(parents=True, exist_ok=True)
        if source_manual.exists():
            for item in source_manual.iterdir():
                if item.is_file() and not (target_manual/item.name).exists():
                    shutil.copy2(item, target_manual/item.name)
        self.d = dirs(self.project)
        # New result layout; legacy helper folders remain for compatibility.
        self.main = self.project/'results'/'main'
        self.tables = self.main/'tables'
        self.figures = self.main/'figures'
        self.models_dir = self.main/'models'
        self.logs = self.main/'logs'
        self.supp_tables = self.main/'supplementary_tables'
        self.supp_figures = self.main/'supplementary_figures'
        self.kan_dir = self.project/'results'/'downstream_KAN'/'frozen_inputs'
        for p in [self.tables,self.figures,self.models_dir,self.logs,self.supp_tables,self.supp_figures,self.kan_dir]:
            p.mkdir(parents=True, exist_ok=True)
        # Redirect helper-generated core outputs into the new main folder.
        self.d['tables'] = self.tables
        self.d['figures'] = self.figures
        self.d['models'] = self.models_dir
        self.d['logs'] = self.logs
        self.d['supp'] = self.supp_tables
        self.runtime = RuntimeTracker(self.tables/'computation_time_summary.csv')
        self.primary_k = int(self.cfg['project']['final_k'])
        self.k_values = list(map(int, self.cfg['project']['sensitivity_k']))
        if self.primary_k not in self.k_values:
            raise ValueError('primary k must also appear in sensitivity_k')

    def _save_both(self, plotter, stem: str, *args, **kwargs):
        plotter(*args, self.figures/f'{stem}.png', **kwargs)
        plotter(*args, self.figures/f'{stem}.pdf', **kwargs)

    def run_all(self) -> dict:
        self.stage_references()
        self.stage_preprocessing()
        self.stage_split()
        self.stage_ppi()
        self.stage_cross_validation()
        self.stage_stability()
        self.stage_full_rankings()
        self.stage_heldout()
        self.stage_biology()
        self.stage_external()
        self.stage_weight_sensitivity()
        self.stage_kmplotter()
        self.stage_figures()
        self.stage_kan_bridge()
        self.stage_finalize()
        return self.summary()

    def stage_references(self):
        token = self.runtime.start('Reference downloads')
        self.hgnc_path, self.string_info_path, self.string_links_path = references(self.project, self.cfg, self.d)
        self.resolver = HGNCResolver.from_complete_set(self.hgnc_path)
        pd.DataFrame([
            {'Reference':'HGNC','Path':str(self.hgnc_path),'Size_MB':self.hgnc_path.stat().st_size/1e6},
            {'Reference':'STRING protein info','Path':str(self.string_info_path),'Size_MB':self.string_info_path.stat().st_size/1e6},
            {'Reference':'STRING links','Path':str(self.string_links_path),'Size_MB':self.string_links_path.stat().st_size/1e6},
        ]).to_csv(self.tables/'reference_download_summary.csv', index=False)
        self.runtime.stop(token)

    def stage_preprocessing(self):
        token = self.runtime.start('Discovery preprocessing and harmonization')
        self.X, self.y, self.metadata, self.prep, self.prep_artifacts = prepare_discovery(
            self.project, self.cfg, self.d, self.resolver
        )
        self.X.to_csv(self.tables/'harmonized_expression_matrix.csv.gz', index=True, index_label='GSM_ID', compression='gzip')
        self.metadata.to_csv(self.tables/'harmonized_metadata.csv', index=False)
        self.runtime.stop(token)

    def stage_split(self):
        token = self.runtime.start('Development-heldout split')
        train_idx, test_idx = train_test_split(
            np.arange(len(self.X)), test_size=float(self.cfg['project']['test_size']),
            stratify=self.y, random_state=int(self.cfg['project']['random_state']),
        )
        self.Xtr, self.Xte = self.X.iloc[train_idx], self.X.iloc[test_idx]
        self.ytr, self.yte = self.y[train_idx], self.y[test_idx]
        self.mtr = self.metadata.iloc[train_idx].reset_index(drop=True)
        self.mte = self.metadata.iloc[test_idx].reset_index(drop=True)
        assignments = pd.concat([
            self.mtr.assign(Set='Model-development', Label_binary=self.ytr),
            self.mte.assign(Set='Post-harmonization held-out', Label_binary=self.yte),
        ], ignore_index=True)
        assignments.to_csv(self.tables/'train_test_split_assignments.csv', index=False)
        assignments.groupby(['Set','Label'], as_index=False).size().rename(columns={'size':'Samples'}).to_csv(
            self.tables/'train_test_class_distribution.csv', index=False
        )
        self.Xtr.to_csv(self.tables/'development_expression_matrix.csv.gz', index=True, index_label='GSM_ID', compression='gzip')
        self.Xte.to_csv(self.tables/'heldout_expression_matrix.csv.gz', index=True, index_label='GSM_ID', compression='gzip')
        self.runtime.stop(token)

    def stage_ppi(self):
        token = self.runtime.start('STRING PPI construction')
        self.edges, self.degree = prepare_ppi(
            self.cfg, self.d, self.resolver, self.string_info_path, self.string_links_path, self.X.columns
        )
        self.ppi_rank = prepare_ppi_rank_table(self.degree, self.X.columns)
        self.ppi_rank.to_csv(self.tables/'ppi_rank_table_training_genes.csv', index=False)
        self.runtime.stop(token)

    def stage_cross_validation(self):
        token = self.runtime.start('Five-fold CV and panel-size sensitivity')
        self.models = create_models(self.cfg)
        self.default_threshold = float(self.cfg['models']['default_decision_threshold'])
        splitter = StratifiedKFold(
            n_splits=int(self.cfg['project']['cv_folds']), shuffle=True,
            random_state=int(self.cfg['project']['random_state']),
        )
        splits = list(splitter.split(self.Xtr, self.ytr))
        self.fold_rankings = {method:{} for method in self.cfg['feature_selection']['methods']}
        prediction_frames, ppi_prediction_frames = [], []
        fold_membership_rows, validation_rows, panel_rows = [], [], []

        for fold, (fit_idx, val_idx) in enumerate(splits, 1):
            ranking_tables = rankings(self.Xtr.iloc[fit_idx], self.ytr[fit_idx], self.degree, self.cfg)
            for method, table in ranking_tables.items():
                self.fold_rankings[method][fold] = table
                table.to_csv(self.tables/f'fold{fold}_{method.replace(" ", "_")}_ranking.csv', index=False)
            for k in self.k_values:
                selected = panels(ranking_tables, k)
                panel_rows.extend(_long_panels(fold, k, selected))
                pred, _ = fit_predict_panels(
                    self.Xtr.iloc[fit_idx], self.ytr[fit_idx], self.Xtr.iloc[val_idx], self.ytr[val_idx],
                    selected, self.models, f'CV fold {fold}',
                )
                pred['Fold'] = fold
                prediction_frames.append(pred)
                ppi_genes = select_top_k(self.ppi_rank, k, 'Rank_topo')
                ppred, _ = fit_predict_panels(
                    self.Xtr.iloc[fit_idx], self.ytr[fit_idx], self.Xtr.iloc[val_idx], self.ytr[val_idx],
                    {'PPI-only': ppi_genes}, self.models, f'CV fold {fold}',
                )
                ppred['Fold'] = fold
                ppi_prediction_frames.append(ppred)
            for idx in fit_idx:
                fold_membership_rows.append({'Fold':fold,'Subset':'Training',**self.mtr.iloc[idx][['GSM_ID','GEO_ID','Label']].to_dict()})
            for idx in val_idx:
                row = self.mtr.iloc[idx]
                fold_membership_rows.append({'Fold':fold,'Subset':'Validation',**row[['GSM_ID','GEO_ID','Label']].to_dict()})
                validation_rows.append({'GSM_ID':row.GSM_ID,'GEO_ID':row.GEO_ID,'Label':row.Label,'Label_binary':int(self.ytr[idx]),'Validation_fold':fold})

        self.cv_predictions = pd.concat(prediction_frames, ignore_index=True)
        self.cv_metrics = metrics_from_predictions(self.cv_predictions, default_threshold=self.default_threshold)
        self.cv_metrics['Fold'] = self.cv_metrics.Dataset.str.extract(r'(\d+)')[0].astype(int)
        self.cv_summary = summarize_cv(self.cv_metrics)
        self.ppi_predictions = pd.concat(ppi_prediction_frames, ignore_index=True)
        self.ppi_cv_metrics = metrics_from_predictions(self.ppi_predictions, default_threshold=self.default_threshold)
        self.ppi_cv_metrics['Fold'] = self.ppi_cv_metrics.Dataset.str.extract(r'(\d+)')[0].astype(int)
        self.ppi_cv_summary = summarize_cv(self.ppi_cv_metrics)
        self.fold_panels = pd.DataFrame(panel_rows)
        self.fold_membership = pd.DataFrame(fold_membership_rows)
        self.validation_assignments = pd.DataFrame(validation_rows).sort_values('GSM_ID')

        outputs = {
            'cross_validated_predictions_all_k.csv': self.cv_predictions,
            'cross_validated_performance_by_fold_all_k.csv': self.cv_metrics,
            'cross_validated_performance_summary_all_k.csv': self.cv_summary,
            f'cross_validated_performance_by_fold_k{self.primary_k}.csv': self.cv_metrics[self.cv_metrics.k==self.primary_k],
            f'cross_validated_performance_summary_k{self.primary_k}.csv': self.cv_summary[self.cv_summary.k==self.primary_k],
            f'foldwise_cv_predictions_k{self.primary_k}.csv': self.cv_predictions[self.cv_predictions.k==self.primary_k],
            'nibfs_panel_size_cv_metrics.csv': self.cv_metrics[self.cv_metrics.Feature_selection_method=='NIBFS'],
            'nibfs_panel_size_cv_summary.csv': self.cv_summary[self.cv_summary.Feature_selection_method=='NIBFS'],
            'foldwise_selected_gene_panels_all_k.csv': self.fold_panels,
            f'foldwise_selected_gene_panels_k{self.primary_k}.csv': self.fold_panels[self.fold_panels.k==self.primary_k],
            'fold_membership_all_folds.csv': self.fold_membership,
            'fold_assignments.csv': self.validation_assignments,
            'ppi_only_descriptive_cv_metrics.csv': self.ppi_cv_metrics,
            'ppi_only_descriptive_cv_summary.csv': self.ppi_cv_summary,
        }
        for name, frame in outputs.items(): frame.to_csv(self.tables/name, index=False)
        self.validation_assignments.groupby(['Validation_fold','Label'], as_index=False).size().rename(columns={'size':'Samples'}).to_csv(
            self.tables/'fold_class_distribution_summary.csv', index=False
        )
        pd.DataFrame({'Selection_rank':np.arange(1,self.primary_k+1),'Gene':select_top_k(self.ppi_rank,self.primary_k,'Rank_topo')}).to_csv(
            self.tables/f'ppi_only_descriptive_panel_k{self.primary_k}.csv', index=False
        )
        self.runtime.stop(token)

    def stage_stability(self):
        token = self.runtime.start('Stability and statistical tests')
        rank_columns = {'NIBFS':'Rank_NIBFS','DEG-only':'Rank_stat','mRMR':'Selection_Order','LASSO':'Rank_LASSO'}
        pair_frames, freq_frames, summaries = [], [], []
        for k in self.k_values:
            for method, fold_tables in self.fold_rankings.items():
                sets = {fold:select_top_k(table,k,rank_columns[method]) for fold,table in fold_tables.items()}
                pair, freq, summary = pairwise_jaccard_summary(sets, method, k)
                freq['Method'], freq['k'] = method, k
                pair_frames.append(pair); freq_frames.append(freq); summaries.append(summary)
        self.pairwise = pd.concat(pair_frames, ignore_index=True)
        self.frequency = pd.concat(freq_frames, ignore_index=True)
        self.stability = pd.DataFrame(summaries).sort_values(['k','Mean_Jaccard'], ascending=[True,False])
        self.pairwise.to_csv(self.tables/'stability_pairwise_jaccard.csv', index=False)
        self.frequency.to_csv(self.tables/'stability_gene_frequency.csv', index=False)
        self.stability.to_csv(self.tables/'stability_summary.csv', index=False)
        self.stability[self.stability.k==self.primary_k].to_csv(self.tables/f'stability_summary_k{self.primary_k}.csv', index=False)
        global_test, posthoc = stability_statistical_tests(self.pairwise, self.primary_k)
        cv_tests = cv_auc_statistical_tests(self.cv_metrics, self.primary_k)
        global_test.to_csv(self.tables/'stability_Friedman_test.csv', index=False)
        posthoc.to_csv(self.tables/'stability_posthoc_Wilcoxon_BH.csv', index=False)
        cv_tests.to_csv(self.tables/'CV_ROCAUC_Wilcoxon_BH.csv', index=False)
        self.runtime.stop(token)

    def stage_full_rankings(self):
        token = self.runtime.start('Full-development rankings and frozen panel')
        self.final_rankings = rankings(self.Xtr, self.ytr, self.degree, self.cfg)
        self.final_panels = {k:panels(self.final_rankings,k) for k in self.k_values}
        for method, table in self.final_rankings.items():
            table.to_csv(self.tables/f'full_training_{method.replace(" ", "_")}_ranking.csv', index=False)
        self.final = self.final_rankings['NIBFS'].head(self.primary_k).copy()
        freq_map = self.frequency[(self.frequency.Method=='NIBFS')&(self.frequency.k==self.primary_k)].set_index('Gene').Fold_Frequency
        self.final['Fold_frequency'] = self.final.Gene.map(freq_map).fillna(0).astype(int)
        self.final.to_csv(self.tables/f'final_NIBFS_gene_panel_k{self.primary_k}.csv', index=False)
        _full_panels(self.final_panels).to_csv(self.tables/'final_feature_panels_full_development_all_k.csv', index=False)
        volcano = self.final_rankings['DEG-only'].copy()
        volcano['Descriptive_DEG'] = (volcano.FDR<=0.05)&(volcano.logFC.abs()>1)
        volcano[f'Final_NIBFS_k{self.primary_k}'] = volcano.Gene.isin(self.final_panels[self.primary_k]['NIBFS'])
        volcano.to_csv(self.tables/'volcano_limma_full_training_table.csv', index=False)
        self.runtime.stop(token)

    def stage_heldout(self):
        token = self.runtime.start('Internal held-out evaluation')
        self.thresholds = derive_youden_thresholds(self.cv_predictions, 'NIBFS', self.primary_k)
        self.thresholds.to_csv(self.tables/'discovery_OOF_Youden_thresholds.csv', index=False)
        self.held_predictions, self.held_models = fit_predict_panels(
            self.Xtr,self.ytr,self.Xte,self.yte,self.final_panels[self.primary_k],self.models,
            'Post-harmonization held-out',
        )
        default = metrics_from_predictions(self.held_predictions, default_threshold=self.default_threshold, threshold_source='Default 0.5').assign(Evaluation_rule='Default')
        transfer = metrics_from_predictions(self.held_predictions, thresholds=self.thresholds, default_threshold=self.default_threshold).assign(Evaluation_rule='OOF-transferred')
        self.held_metrics = pd.concat([default,transfer], ignore_index=True)
        self.held_metrics = add_auc_confidence_intervals(
            self.held_metrics, self.held_predictions,
            int(self.cfg['external_validation']['bootstrap_iterations']),
            int(self.cfg['project']['random_state']),
        )
        self.held_predictions.to_csv(self.tables/'heldout_predictions.csv', index=False)
        self.held_metrics.to_csv(self.tables/'heldout_metrics_default_and_transferred.csv', index=False)
        self.held_metrics[self.held_metrics.Evaluation_rule=='Default'].to_csv(self.tables/'heldout_metrics.csv', index=False)
        calibration_table(self.held_predictions).to_csv(self.tables/'heldout_calibration_curve.csv', index=False)
        pd.DataFrame([{
            'Classifier':clf,'Brier_score':float(group.Brier_score.iloc[0]) if 'Brier_score' in group else np.nan
        } for clf,group in self.held_metrics.groupby('Classifier')]).to_csv(self.tables/'heldout_calibration_summary.csv', index=False)
        sens=[]
        for k in self.k_values:
            pred,_=fit_predict_panels(self.Xtr,self.ytr,self.Xte,self.yte,{'NIBFS':self.final_panels[k]['NIBFS']},self.models,'Post-harmonization held-out')
            sens.append(metrics_from_predictions(pred,default_threshold=self.default_threshold))
        pd.concat(sens,ignore_index=True).to_csv(self.tables/'nibfs_panel_size_heldout_metrics.csv',index=False)
        ppi_pred,_=fit_predict_panels(self.Xtr,self.ytr,self.Xte,self.yte,{'PPI-only':select_top_k(self.ppi_rank,self.primary_k,'Rank_topo')},self.models,'Post-harmonization held-out')
        metrics_from_predictions(ppi_pred,default_threshold=self.default_threshold).to_csv(self.tables/'ppi_only_descriptive_heldout_metrics.csv',index=False)
        if bool(self.cfg['outputs'].get('save_models',True)):
            for (method,clf),model in self.held_models.items():
                if method=='NIBFS': joblib.dump(model,self.models_dir/f'{clf}_full_development_k{self.primary_k}.joblib')
        self.runtime.stop(token)

    def stage_biology(self):
        token = self.runtime.start('Functional enrichment and selected-gene PPI')
        genes = self.final_panels[self.primary_k]['NIBFS']
        self.panel_edges = final_panel_subnetwork(self.edges, genes)
        self.panel_edges.to_csv(self.tables/f'biological_validation_STRING_edges_k{self.primary_k}.csv',index=False)
        self.network_summary,self.centrality = plot_ppi_network(genes,self.panel_edges,self.final_rankings['NIBFS'],self.figures/f'Figure_PPI_network_k{self.primary_k}.png',int(self.cfg['project']['random_state']))
        self.network_summary.to_csv(self.tables/f'biological_validation_STRING_network_summary_k{self.primary_k}.csv',index=False)
        self.centrality.to_csv(self.tables/f'biological_validation_STRING_centrality_k{self.primary_k}.csv',index=False)
        raw=run_enrichment(genes,self.Xtr.columns.tolist(),self.cfg['biological_interpretation']['enrichment_sources'])
        raw.to_csv(self.tables/'enrichment_gprofiler_raw.csv',index=False)
        self.enrichment=standardize_enrichment(raw,self.primary_k)
        self.enrichment.to_csv(self.tables/'biological_validation_enrichment_all_terms.csv',index=False)
        for token_name,filename in [('GO:BP','biological_validation_GO_BP_enrichment.csv'),('KEGG','biological_validation_KEGG_enrichment.csv'),('REAC','biological_validation_Reactome_enrichment.csv')]:
            _source_subset(self.enrichment,token_name).to_csv(self.tables/filename,index=False)
        self.enrichment.sort_values('Adjusted_p_value').groupby('Database',as_index=False).head(10).to_csv(self.tables/'biological_validation_main_enrichment_table.csv',index=False)
        self.runtime.stop(token)

    def stage_external(self):
        token = self.runtime.start('Independent GSE15852 external validation')
        self.Xext,self.yext,self.mext=load_external(self.cfg,self.d,self.resolver)
        genes=self.final_panels[self.primary_k]['NIBFS']
        availability=pd.DataFrame({'Gene':genes,'Available_in_GSE15852':[g in self.Xext.columns for g in genes]})
        availability.to_csv(self.tables/f'external_GSE15852_gene_availability_k{self.primary_k}.csv',index=False)
        missing=availability.loc[~availability.Available_in_GSE15852,'Gene'].tolist()
        if missing and bool(self.cfg['external_validation'].get('require_complete_frozen_panel',True)):
            raise ValueError(f'Frozen panel genes missing from GSE15852: {missing}')
        available=availability.loc[availability.Available_in_GSE15852,'Gene'].tolist()
        self.external_predictions,_=fit_predict_panels(self.Xtr,self.ytr,self.Xext,self.yext,{'NIBFS':available},self.models,'Independent external GSE15852')
        default=metrics_from_predictions(self.external_predictions,default_threshold=self.default_threshold,threshold_source='Default 0.5').assign(Evaluation_rule='Default')
        transfer=metrics_from_predictions(self.external_predictions,thresholds=self.thresholds,default_threshold=self.default_threshold).assign(Evaluation_rule='OOF-transferred')
        self.external_metrics=pd.concat([default,transfer],ignore_index=True)
        self.external_metrics=add_auc_confidence_intervals(self.external_metrics,self.external_predictions,int(self.cfg['external_validation']['bootstrap_iterations']),int(self.cfg['project']['random_state']))
        self.external_predictions.to_csv(self.tables/'external_GSE15852_predictions.csv',index=False)
        self.external_metrics.to_csv(self.tables/'external_GSE15852_metrics_default_and_transferred.csv',index=False)
        training_fc=self.final_rankings['NIBFS'].set_index('Gene').logFC
        rows=[]
        for gene in available:
            ext_fc=float(self.Xext.loc[self.yext==1,gene].mean()-self.Xext.loc[self.yext==0,gene].mean())
            tr_fc=float(training_fc.loc[gene])
            rows.append({'Gene':gene,'Training_logFC':tr_fc,'External_logFC':ext_fc,'Training_direction':'Up' if tr_fc>0 else 'Down','External_direction':'Up' if ext_fc>0 else 'Down','Direction_consistent':bool(np.sign(tr_fc)==np.sign(ext_fc))})
        self.direction=pd.DataFrame(rows)
        self.direction.to_csv(self.tables/'external_GSE15852_direction_consistency.csv',index=False)
        pd.DataFrame([{'Available_genes':len(self.direction),'Direction_consistent':int(self.direction.Direction_consistent.sum()),'Direction_discordant':int((~self.direction.Direction_consistent).sum()),'Consistency_fraction':float(self.direction.Direction_consistent.mean())}]).to_csv(self.tables/'external_GSE15852_direction_consistency_summary.csv',index=False)
        calibration_table(self.external_predictions).to_csv(self.tables/'external_GSE15852_calibration_curve.csv',index=False)
        self.runtime.stop(token)

    def stage_weight_sensitivity(self):
        token=self.runtime.start('Rank-weight sensitivity')
        run_weight_sensitivity(self.project, primary_k=self.primary_k, alphas=tuple(map(float, self.cfg['rank_weight_sensitivity']['statistical_weights'])), random_state=int(self.cfg['project']['random_state']))
        self.runtime.stop(token)

    def stage_kmplotter(self):
        token=self.runtime.start('KM Plotter post hoc integration')
        genes=self.final_panels[self.primary_k]['NIBFS']
        manual=self.project/self.cfg['biological_interpretation']['kmplotter_csv']
        template=self.project/'manual_inputs'/f'KMPlotter_RFS_final_k{self.primary_k}_TEMPLATE.csv'
        template.parent.mkdir(parents=True,exist_ok=True)
        if not template.exists():
            pd.DataFrame({'Gene':genes,'KM_gene':'','Probe':'','HR':np.nan,'CI_low':np.nan,'CI_high':np.nan,'p':np.nan,'direction':''}).to_csv(template,index=False)
        self.km=pd.DataFrame()
        if manual.exists():
            self.km=load_kmplotter_csv(manual)
            missing=sorted(set(genes)-set(self.km.Gene.astype(str))) if not self.km.empty else genes
            if missing:
                warnings.warn(f'KM Plotter file is incomplete for k={self.primary_k}; missing genes: {missing}')
            else:
                self.km=self.km[self.km.Gene.isin(genes)].copy()
                self.km.to_csv(self.tables/f'KMPlotter_RFS_standardized_k{self.primary_k}.csv',index=False)
                plot_km_forest(self.km,self.supp_figures/f'Figure_KMPlotter_RFS_forest_k{self.primary_k}.png')
                plot_km_forest(self.km,self.supp_figures/f'Figure_KMPlotter_RFS_forest_k{self.primary_k}.pdf')
        else:
            warnings.warn(f'KM Plotter input not found. Fill template: {template}')
        self.runtime.stop(token)

    def stage_figures(self):
        token=self.runtime.start('Generate all main and supplementary figures')
        genes=self.final_panels[self.primary_k]['NIBFS']
        # Recreate preprocessing figures in both PNG and PDF from cached tables.
        from .figures import (plot_pca_before_after, plot_expression_distributions, plot_missing_values, plot_correlation_heatmap, plot_outlier_audit, plot_preprocessing_qc_overview)
        before=pd.read_csv(self.tables/'PCA_before_coordinates.csv')
        after=pd.read_csv(self.tables/'PCA_after_coordinates.csv')
        variance=pd.read_csv(self.tables/'PCA_variance_explained.csv')
        distribution=pd.read_csv(self.tables/'sample_expression_distribution_summary.csv')
        missing=pd.read_csv(self.tables/'missing_value_summary.csv')
        corr=pd.read_csv(self.tables/'sample_correlation_matrix.csv.gz',index_col=0)
        qc=pd.read_csv(self.tables/'sample_QC_audit_metrics.csv')
        self._save_both(plot_pca_before_after,'Figure_2_PCA_before_after_harmonization',before,after,variance)
        self._save_both(plot_expression_distributions,'Figure_S1_Expression_distributions',distribution)
        self._save_both(plot_missing_values,'Figure_S2_Missing_value_audit',missing)
        self._save_both(plot_correlation_heatmap,'Figure_S3_Sample_correlation_heatmap',corr,self.metadata)
        self._save_both(plot_outlier_audit,'Figure_S4_Sample_outlier_audit',qc)
        self._save_both(plot_preprocessing_qc_overview,'Figure_S5_Preprocessing_QC_overview',distribution,missing,variance,qc)
        volcano=pd.read_csv(self.tables/'volcano_limma_full_training_table.csv')
        self._save_both(plot_volcano,f'Figure_3_Volcano_k{self.primary_k}',volcano,genes)
        chosen,counts=choose_compact_heatmap_samples(self.mtr,max_per_cohort_class=int(self.cfg['outputs']['compact_heatmap_max_per_cohort_class']),random_state=int(self.cfg['project']['random_state']))
        counts.to_csv(self.tables/f'compact_heatmap_selected_sample_counts_k{self.primary_k}.csv',index=False)
        compact_X=self.Xtr.loc[self.Xtr.index.intersection(chosen)]
        compact_m=self.mtr.set_index('GSM_ID').loc[compact_X.index].reset_index()
        self._save_both(plot_clustered_heatmap,f'Figure_4_Compact_heatmap_k{self.primary_k}',compact_X,compact_m,genes,title=f'Frozen NIBFS top-{self.primary_k} expression heatmap')
        sample_order,gene_order,_=clustered_heatmap_orders(compact_X,genes)
        pd.DataFrame({'GSM_ID':sample_order}).to_csv(self.tables/f'heatmap_sample_order_k{self.primary_k}.csv',index=False)
        pd.DataFrame({'Gene':gene_order}).to_csv(self.tables/f'heatmap_gene_order_k{self.primary_k}.csv',index=False)
        self._save_both(plot_clustered_heatmap,f'Figure_S_Full_heatmap_k{self.primary_k}',self.Xtr,self.mtr,genes,title=f'Full model-development heatmap, top-{self.primary_k}')
        self._save_both(plot_stability_composite,f'Figure_5_Stability_and_recurrence_k{self.primary_k}',self.pairwise,self.frequency,self.primary_k)
        self._save_both(plot_gene_occurrence_heatmap,f'Figure_S_Gene_occurrence_k{self.primary_k}',self.fold_panels,method='NIBFS',k=self.primary_k)
        self._save_both(plot_cv_performance,f'Figure_6_CV_performance_k{self.primary_k}',self.cv_metrics,k=self.primary_k)
        self._save_both(plot_sensitivity,'Figure_S_Panel_size_sensitivity',self.stability,self.cv_summary,method='NIBFS')
        self._save_both(plot_rank_landscape,f'Figure_S_Rank_landscape_k{self.primary_k}',self.final_rankings['NIBFS'],genes)
        self._save_both(plot_top_panel_barplot,f'Figure_S_Final_panel_barplot_k{self.primary_k}',self.final)
        held_cal=calibration_table(self.held_predictions)
        self._save_both(plot_roc,f'Figure_S_Heldout_ROC_k{self.primary_k}',self.held_predictions,title='Post-harmonization held-out ROC')
        self._save_both(plot_heldout_composite,f'Figure_S_Heldout_composite_k{self.primary_k}',self.held_predictions,held_cal)
        self._save_both(plot_heldout_roc_only,f'Figure_S_Heldout_ROC_only_k{self.primary_k}',self.held_predictions)
        self._save_both(plot_heldout_performance,f'Figure_S_Heldout_performance_k{self.primary_k}',self.held_metrics[self.held_metrics.Evaluation_rule=='Default'])
        self._save_both(plot_external_validation,f'Figure_7_External_validation_k{self.primary_k}',self.external_predictions,self.direction)
        self._save_both(plot_external_roc_only,f'Figure_S_External_ROC_k{self.primary_k}',self.external_predictions)
        self._save_both(plot_external_direction_only,f'Figure_S_External_direction_k{self.primary_k}',self.direction)
        self._save_both(plot_enrichment,f'Figure_8_Functional_enrichment_k{self.primary_k}',self.enrichment)
        # Dedicated PPI in both formats; the stage_biology PNG is retained as an early preview.
        plot_ppi_network(genes,self.panel_edges,self.final_rankings['NIBFS'],self.figures/f'Figure_9_PPI_network_k{self.primary_k}.png',int(self.cfg['project']['random_state']))
        plot_ppi_network(genes,self.panel_edges,self.final_rankings['NIBFS'],self.figures/f'Figure_9_PPI_network_k{self.primary_k}.pdf',int(self.cfg['project']['random_state']))
        self._save_both(plot_biological_interpretation,f'Figure_10_Biological_interpretation_k{self.primary_k}',self.enrichment,genes,self.panel_edges,self.final_rankings['NIBFS'],seed=int(self.cfg['project']['random_state']))
        manifest={'samples_total':len(self.X),'cancer':int((self.y==1).sum()),'normal':int((self.y==0).sum()),'eligible_genes':self.X.shape[1],'training_samples':len(self.Xtr),'heldout_samples':len(self.Xte),'primary_k':self.primary_k}
        nib_stab=float(self.stability[(self.stability.Method=='NIBFS')&(self.stability.k==self.primary_k)].Mean_Jaccard.iloc[0])
        best_auc=float(self.external_metrics[self.external_metrics.Evaluation_rule=='Default'].ROC_AUC.max())
        self._save_both(plot_study_workflow,'Figure_1_Study_workflow',manifest,nib_stab,best_auc)
        plot_graphical_abstract(manifest,nib_stab,best_auc,int(self.direction.Direction_consistent.sum()),self.figures/'Graphical_Abstract.png')
        self.runtime.stop(token)

    def stage_kan_bridge(self):
        token=self.runtime.start('Freeze RF-LightGBM bridge for downstream KAN')
        bridge=build_kan_bridge(
            self.cv_predictions,self.held_predictions,self.mtr,self.mte,k=self.primary_k,
            external_predictions=self.external_predictions,external_metadata=self.mext,
        )
        files={'train_oof':f'KAN_meta_train_OOF_k{self.primary_k}.csv','heldout':f'KAN_meta_heldout_k{self.primary_k}.csv','external':f'KAN_meta_external_GSE15852_k{self.primary_k}.csv'}
        for key,frame in bridge.items(): frame.to_csv(self.kan_dir/files[key],index=False)
        self.final.to_csv(self.kan_dir/f'KAN_frozen_gene_panel_k{self.primary_k}.csv',index=False)
        self.validation_assignments.to_csv(self.kan_dir/'KAN_fold_assignments.csv',index=False)
        audit=[]
        train=bridge['train_oof']
        checks={
            'OOF rows':(len(self.Xtr),len(train)),
            'Unique OOF samples':(len(self.Xtr),train.Sample_ID.nunique() if not train.empty else 0),
            'Missing RF probabilities':(0,int(train.p_RF.isna().sum()) if not train.empty else -1),
            'Missing LightGBM probabilities':(0,int(train.p_LightGBM.isna().sum()) if not train.empty else -1),
        }
        for item,(expected,observed) in checks.items(): audit.append({'Check':item,'Expected':expected,'Observed':observed,'Status':'PASS' if expected==observed else 'FAIL'})
        pd.DataFrame(audit).to_csv(self.kan_dir/'KAN_bridge_audit.csv',index=False)
        manifest={'panel_size':self.primary_k,'meta_features':['p_RF','p_LightGBM'],'development_samples':len(self.Xtr),'heldout_samples':len(self.Xte),'external_samples':len(self.Xext),'cv_folds':int(self.cfg['project']['cv_folds']),'random_state':int(self.cfg['project']['random_state'])}
        write_json(self.kan_dir/'KAN_bridge_manifest.json',manifest)
        self.runtime.stop(token)

    def stage_finalize(self):
        token=self.runtime.start('Audit, inventory, and evidence ZIP')
        structural=[]
        def check(item,condition,observed=''):
            structural.append({'Item':item,'Observed':observed,'Status':'PASS' if condition else 'FAIL'})
        check('Primary panel row count',len(self.final)==self.primary_k,len(self.final))
        check('Primary panel unique genes',self.final.Gene.nunique()==self.primary_k,self.final.Gene.nunique())
        check('OOF samples covered',self.cv_predictions[(self.cv_predictions.Feature_selection_method=='NIBFS')&(self.cv_predictions.Classifier=='RF')&(self.cv_predictions.k==self.primary_k)].Sample_ID.nunique()==len(self.Xtr))
        check('External frozen panel complete',set(self.final.Gene).issubset(self.Xext.columns))
        check('No repeated CV in core',not bool(self.cfg['optional_analyses']['repeated_stability']['enabled']))
        check('No LOCO in core',not bool(self.cfg['optional_analyses']['loco']['enabled']))
        pd.DataFrame(structural).to_csv(self.tables/'FINAL_STRUCTURAL_AUDIT.csv',index=False)
        self.runtime.stop(token)
        self.runtime.finish_total()
        inventory=[]
        for path in sorted(self.project.rglob('*')):
            if path.is_file():
                inventory.append({'Relative_path':str(path.relative_to(self.project)),'Bytes':path.stat().st_size,'SHA256':hashlib.sha256(path.read_bytes()).hexdigest()})
        pd.DataFrame(inventory).to_csv(self.tables/'OUTPUT_INVENTORY_SHA256.csv',index=False)
        manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'primary_k':self.primary_k,'sensitivity_k':self.k_values,'random_state':int(self.cfg['project']['random_state']),'samples':len(self.X),'eligible_genes':self.X.shape[1]}
        write_json(self.logs/'run_manifest.json',manifest)
        if bool(self.cfg['outputs'].get('create_evidence_zip',True)):
            zip_path=self.project.parent/f'{self.project.name}_EVIDENCE.zip'
            if zip_path.exists(): zip_path.unlink()
            with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as z:
                include_raw = bool(self.cfg['outputs'].get('evidence_zip_include_raw_downloads', False))
                for path in self.project.rglob('*'):
                    if not path.is_file():
                        continue
                    rel = path.relative_to(self.project)
                    if (not include_raw) and str(rel).startswith('data/raw/'):
                        continue
                    z.write(path, arcname=str(rel))
            print('Evidence ZIP:',zip_path)

    def summary(self) -> dict:
        return {'project_dir':str(self.project),'primary_k':self.primary_k,'samples':len(self.X),'development_samples':len(self.Xtr),'heldout_samples':len(self.Xte),'eligible_genes':self.X.shape[1]}
