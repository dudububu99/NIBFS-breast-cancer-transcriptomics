from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, friedmanchisquare, wilcoxon
from sklearn.base import clone
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, f1_score, matthews_corrcoef,
    recall_score, precision_score, confusion_matrix, brier_score_loss,
)
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.multitest import multipletests
from neuroCombat import neuroCombat, neuroCombatFromTraining

from .common import (
    AdditionalAnalysisPaths,
    add_repo_to_path,
    clean_gene_symbol,
    acquire_reference_geo_series_matrix,
    find_column,
    read_series_matrix_numeric,
    write_json,
)

DISCOVERY_GSE = [
    "GSE61304", "GSE42568", "GSE29044", "GSE3744", "GSE29431",
    "GSE26910", "GSE31138", "GSE71053", "GSE10780", "GSE30010",
    "GSE111662",
]
PLATFORM = "GPL570"
N_FOLDS = 5
N_REPEATS = 5
FINAL_K = 20
BASE_SEED = 42
LOG2_THRESHOLD = 100.0
BOTTOM_VARIANCE_FRACTION = 0.10
METHODS = ["NIBFS", "DEG-only", "mRMR", "LASSO"]
RANK_COLUMNS = {
    "NIBFS": "Rank_NIBFS",
    "DEG-only": "Rank_stat",
    "mRMR": "Selection_Order",
    "LASSO": "Rank_LASSO",
}


@dataclass
class FrozenQuantileNormalizer:
    reference: np.ndarray | None = None

    def fit(self, X: pd.DataFrame):
        self.reference = np.sort(X.to_numpy(dtype=float), axis=1).mean(axis=0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.reference is None:
            raise RuntimeError("Quantile normalizer is not fitted.")
        values = X.to_numpy(dtype=float)
        output = np.empty_like(values, dtype=float)
        positions = np.arange(len(self.reference), dtype=float)
        for row_index, row in enumerate(values):
            ranks = rankdata(row, method="average") - 1.0
            output[row_index] = np.interp(ranks, positions, self.reference)
        return pd.DataFrame(output, index=X.index, columns=X.columns)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


def _calculate_metrics(y_true, probability) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    predicted = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "ROC_AUC": float(roc_auc_score(y_true, probability)),
        "PR_AUC": float(average_precision_score(y_true, probability)),
        "Accuracy": float(accuracy_score(y_true, predicted)),
        "Balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "Sensitivity": float(recall_score(y_true, predicted, zero_division=0)),
        "Specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "Precision": float(precision_score(y_true, predicted, zero_division=0)),
        "F1": float(f1_score(y_true, predicted, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, predicted)),
        "Brier_score": float(brier_score_loss(y_true, probability)),
    }


def _standardize_method(value: object) -> str:
    text = str(value).strip().casefold().replace("_", "-")
    if text == "nibfs":
        return "NIBFS"
    if text in {"deg-only", "deg"}:
        return "DEG-only"
    if "mrmr" in text:
        return "mRMR"
    if "lasso" in text:
        return "LASSO"
    return str(value)


def _prepare_development_prefold_matrix(paths: AdditionalAnalysisPaths) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Recreate only the pre-fold, label-free portion of the archived strict protocol.

    Conditional log2, representative-probe choice, and common-gene intersection are
    based on the fixed 608-sample development subset, as in the reference 1x5
    fold-fitted sensitivity notebook. The result is cached only in the additional analysis
    workspace.
    """
    cache_dir = paths.workspace / "cache_fold_fitted"
    cache_dir.mkdir(parents=True, exist_ok=True)
    x_cache = cache_dir / "development_prefold_expression.pkl.gz"
    meta_cache = cache_dir / "development_prefold_metadata.csv"

    split_path = paths.tables_dir / "train_test_split_assignments.csv"
    probe_map_path = paths.tables_dir / "GPL570_probe_to_HGNC_mapping.csv"
    if not split_path.is_file() or not probe_map_path.is_file():
        raise FileNotFoundError("Reference split/probe mapping files are missing from the tables ZIP.")

    split = pd.read_csv(split_path)
    dev = split.loc[split["Set"].astype(str).eq("Model-development")].drop_duplicates("GSM_ID").copy()
    if len(dev) != 608:
        raise RuntimeError(f"Expected 608 development samples, found {len(dev)}")
    dev["GSM_ID"] = dev["GSM_ID"].astype(str)
    dev["Label_binary"] = pd.to_numeric(dev["Label_binary"]).astype(int)
    dev["GEO_ID"] = dev["GEO_ID"].astype(str)

    # If cache exists, verify sample identity before using it.
    if x_cache.is_file() and meta_cache.is_file():
        X = pd.read_pickle(x_cache, compression="gzip")
        cached_meta = pd.read_csv(meta_cache)
        if (
            len(X) == 608
            and set(X.index.astype(str)) == set(dev["GSM_ID"])
            and set(cached_meta["GSM_ID"].astype(str)) == set(dev["GSM_ID"])
        ):
            X = X.loc[dev["GSM_ID"].tolist()]
            y = dev.set_index("GSM_ID").loc[X.index, "Label_binary"].astype(int)
            batch = dev.set_index("GSM_ID").loc[X.index, "GEO_ID"].astype(str)
            return X, y, batch, dev

    probe_map_raw = pd.read_csv(probe_map_path)
    probe_col = find_column(probe_map_raw, ["ID_REF", "Probe_ID", "Probe_Set_ID", "Probe", "Affymetrix_Probe_Set_ID"])
    gene_col = find_column(probe_map_raw, ["Gene", "Gene_Symbol", "HGNC_Symbol", "Symbol", "gene_symbol"])
    if probe_col is None or gene_col is None:
        raise KeyError(f"Cannot detect probe mapping columns in {probe_map_path}")
    probe_map = probe_map_raw[[probe_col, gene_col]].copy()
    probe_map.columns = ["Probe", "Gene"]
    probe_map["Probe"] = probe_map["Probe"].astype(str).str.strip()
    probe_map["Gene"] = probe_map["Gene"].map(clean_gene_symbol)
    probe_map = (
        probe_map.loc[probe_map["Probe"].ne("") & probe_map["Gene"].ne("")]
        .sort_values(["Probe", "Gene"])
        .drop_duplicates("Probe", keep="first")
    )
    probe_to_gene = probe_map.set_index("Probe")["Gene"].to_dict()

    development_ids = set(dev["GSM_ID"])
    mapped_by_cohort: dict[str, pd.DataFrame] = {}
    sample_to_cohort: dict[str, str] = {}
    audit_rows = []

    for gse in DISCOVERY_GSE:
        series, raw_hash = acquire_reference_geo_series_matrix(
            paths.repo_dir, paths.tables_dir, gse, PLATFORM,
            paths.raw_geo_dir / f"{gse}_series_matrix.txt.gz",
        )
        raw = read_series_matrix_numeric(series)
        sample_columns = [c for c in raw.columns if c != "ID_REF" and str(c) in development_ids]
        expected_n = int((dev["GEO_ID"] == gse).sum())
        if len(sample_columns) != expected_n:
            raise RuntimeError(
                f"{gse}: expected {expected_n} development samples, found {len(sample_columns)} in series matrix"
            )
        numeric = raw[sample_columns].apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any().any():
            raise RuntimeError(f"{gse}: missing/non-numeric expression values detected")
        values = numeric.to_numpy(dtype=float)
        apply_log2 = bool(np.min(values) >= 0 and np.max(values) > LOG2_THRESHOLD)
        if apply_log2:
            numeric = np.log2(numeric + 1.0)
        numeric.index = raw["ID_REF"].astype(str)
        available = numeric.index.intersection(pd.Index(probe_to_gene.keys()))
        numeric = numeric.loc[available]
        variances = numeric.var(axis=1, ddof=1)
        selection = pd.DataFrame(
            {
                "Probe": numeric.index,
                "Gene": [probe_to_gene[p] for p in numeric.index],
                "Variance_all_development_within_cohort": variances.to_numpy(),
            }
        ).sort_values(
            ["Gene", "Variance_all_development_within_cohort", "Probe"],
            ascending=[True, False, True],
        ).drop_duplicates("Gene", keep="first")
        chosen = numeric.loc[selection["Probe"].tolist()].copy()
        chosen.index = selection["Gene"].tolist()
        gene_matrix = chosen.T
        gene_matrix.index = gene_matrix.index.astype(str)
        mapped_by_cohort[gse] = gene_matrix
        for sample_id in gene_matrix.index:
            if sample_id in sample_to_cohort:
                raise RuntimeError(f"Duplicate sample across discovery cohorts: {sample_id}")
            sample_to_cohort[sample_id] = gse
        audit_rows.append(
            {
                "GEO_ID": gse,
                "Development_samples": len(sample_columns),
                "Probe_rows_original": len(raw),
                "Probe_rows_mapped": len(numeric),
                "Genes_after_representative_probe": gene_matrix.shape[1],
                "Log2_applied": apply_log2,
                "Raw_series_SHA256_match": bool(raw_hash["match"]),
                "Raw_series_SHA256": raw_hash["observed_sha256"],
            }
        )

    common_genes = sorted(set.intersection(*[set(x.columns) for x in mapped_by_cohort.values()]))
    if len(common_genes) != 19134:
        # Do not fail for a tiny upstream GEO re-hosting change, but make it explicit.
        print(f"WARNING: expected 19,134 common genes from archived run; reconstructed {len(common_genes):,}.")
    X = pd.concat([mapped_by_cohort[g][common_genes] for g in DISCOVERY_GSE], axis=0)
    X.index = X.index.astype(str)
    X = X.loc[dev["GSM_ID"].tolist()]
    if X.shape[0] != 608 or X.isna().any().any():
        raise RuntimeError(f"Invalid pre-fold development matrix: shape={X.shape}")

    y = dev.set_index("GSM_ID").loc[X.index, "Label_binary"].astype(int)
    batch = pd.Series([sample_to_cohort[s] for s in X.index], index=X.index, name="GEO_ID")

    X.to_pickle(x_cache, compression="gzip")
    dev.to_csv(meta_cache, index=False)
    pd.DataFrame(audit_rows).to_csv(cache_dir / "prefold_reconstruction_audit.csv", index=False)
    pd.DataFrame({"Gene": common_genes}).to_csv(cache_dir / "common_genes.csv", index=False)
    return X, y, batch, dev


def _fold_assignments(dev: pd.DataFrame, archived_assignment: pd.DataFrame) -> pd.DataFrame:
    dev = dev.copy().reset_index(drop=True)
    y = dev["Label_binary"].to_numpy(dtype=int)
    rows = []
    # Repeat 1 must exactly reproduce archived primary five-fold assignment (seed 42).
    archived = archived_assignment[["GSM_ID", "Fold"]].copy()
    archived["GSM_ID"] = archived["GSM_ID"].astype(str)
    archived_map = archived.set_index("GSM_ID")["Fold"].astype(int).to_dict()
    skf1 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    reconstructed = {}
    for fold, (_, valid_idx) in enumerate(skf1.split(np.zeros(len(dev)), y), 1):
        for i in valid_idx:
            reconstructed[str(dev.iloc[i]["GSM_ID"])] = fold
    if set(reconstructed) != set(archived_map) or any(reconstructed[k] != archived_map[k] for k in reconstructed):
        raise RuntimeError("Archived strict 1x5 fold assignment does not match reference StratifiedKFold seed=42.")

    for repeat in range(1, N_REPEATS + 1):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=BASE_SEED + repeat - 1)
        for fold, (_, valid_idx) in enumerate(skf.split(np.zeros(len(dev)), y), 1):
            for i in valid_idx:
                rows.append(
                    {
                        "Repeat": repeat,
                        "Fold": fold,
                        "GSM_ID": str(dev.iloc[i]["GSM_ID"]),
                        "Label": int(y[i]),
                        "GEO_ID": str(dev.iloc[i]["GEO_ID"]),
                        "Random_state": BASE_SEED + repeat - 1,
                    }
                )
    out = pd.DataFrame(rows)
    if len(out) != 608 * N_REPEATS:
        raise RuntimeError("Unexpected repeated strict fold-assignment size.")
    return out


def _pairwise_repeat_stability(panels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_rows = []
    repeat_rows = []
    for (repeat, method), block in panels.groupby(["Repeat", "Method"], sort=True):
        fold_sets = {
            int(f): set(g["Gene"].astype(str))
            for f, g in block.groupby("Fold")
        }
        values = []
        for a, b in combinations(sorted(fold_sets), 2):
            left, right = fold_sets[a], fold_sets[b]
            j = len(left & right) / len(left | right)
            values.append(j)
            pair_rows.append(
                {
                    "Repeat": int(repeat), "Method": method,
                    "Fold_A": a, "Fold_B": b, "Jaccard": j,
                    "Intersection": len(left & right), "Union": len(left | right),
                }
            )
        repeat_rows.append(
            {
                "Repeat": int(repeat), "Method": method,
                "Mean_Jaccard": float(np.mean(values)),
                "SD_pairwise_Jaccard": float(np.std(values, ddof=1)),
            }
        )
    return pd.DataFrame(pair_rows), pd.DataFrame(repeat_rows)


def run(paths: AdditionalAnalysisPaths) -> dict:
    out = paths.results_dir / "02_repeated_fold_fitted_5x5"
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = paths.checkpoints_dir / "repeated_fold_fitted_5x5"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    add_repo_to_path(paths.repo_dir)
    import yaml
    from src.workflow import rankings as exact_rankings
    from src.feature_selection import select_top_k
    from src.modeling import create_models
    from src.stability_estimators import nogueira_stability

    cfg = yaml.safe_load((paths.repo_dir / "config.yaml").read_text(encoding="utf-8"))
    ppi_degree = pd.read_csv(paths.tables_dir / "ppi_degree_table.csv")
    exact_models = create_models(cfg)

    archived_dir = paths.repo_dir / "results" / "verification" / "fold_fitted"
    archived_assign = pd.read_csv(archived_dir / "fold_fitted_1x5_fold_assignments.csv")
    archived_metrics = pd.read_csv(archived_dir / "fold_fitted_all_methods_1x5_fold_metrics.csv")
    archived_panels = pd.read_csv(archived_dir / "fold_fitted_all_methods_1x5_selected_panels.csv")
    archived_metrics["Method"] = archived_metrics["Method"].map(_standardize_method)
    archived_panels["Method"] = archived_panels["Method"].map(_standardize_method)
    archived_metrics.insert(0, "Repeat", 1)
    archived_panels.insert(0, "Repeat", 1)

    X_global, y_global, batch_global, dev = _prepare_development_prefold_matrix(paths)
    assignments = _fold_assignments(dev, archived_assign)
    assignments.to_csv(out / "repeated_fold_fitted_5x5_fold_assignments.csv", index=False)

    # Repeat 1 is reference archived output; do not rerun its feature selection.
    new_panel_parts: list[pd.DataFrame] = []
    new_metric_parts: list[pd.DataFrame] = []
    new_pred_parts: list[pd.DataFrame] = []
    audit_rows = []

    for repeat in range(2, N_REPEATS + 1):
        for fold in range(1, N_FOLDS + 1):
            prefix = checkpoint_dir / f"repeat_{repeat:02d}_fold_{fold:02d}"
            panel_file = prefix.with_name(prefix.name + "_panels.csv")
            metric_file = prefix.with_name(prefix.name + "_metrics.csv")
            pred_file = prefix.with_name(prefix.name + "_predictions.csv.gz")
            audit_file = prefix.with_name(prefix.name + "_audit.json")
            if panel_file.is_file() and metric_file.is_file() and pred_file.is_file() and audit_file.is_file():
                print(f"[strict {repeat}/{N_REPEATS} fold {fold}/{N_FOLDS}] checkpoint found — skipped")
                new_panel_parts.append(pd.read_csv(panel_file))
                new_metric_parts.append(pd.read_csv(metric_file))
                new_pred_parts.append(pd.read_csv(pred_file))
                audit_rows.append(json.loads(audit_file.read_text(encoding="utf-8")))
                continue

            started = time.perf_counter()
            valid_ids = assignments.loc[
                assignments["Repeat"].eq(repeat) & assignments["Fold"].eq(fold), "GSM_ID"
            ].astype(str).tolist()
            valid_set = set(valid_ids)
            train_ids = [sid for sid in X_global.index.astype(str) if sid not in valid_set]
            X_train_raw = X_global.loc[train_ids].copy()
            X_valid_raw = X_global.loc[valid_ids].copy()
            y_train = y_global.loc[train_ids]
            y_valid = y_global.loc[valid_ids]
            batch_train = batch_global.loc[train_ids]
            batch_valid = batch_global.loc[valid_ids]
            missing_batches = set(batch_valid.unique()) - set(batch_train.unique())
            if missing_batches:
                raise RuntimeError(f"Repeat {repeat} fold {fold}: validation-only batches {sorted(missing_batches)}")

            # A. Fold-training quantile reference
            qn = FrozenQuantileNormalizer()
            X_train_qn = qn.fit_transform(X_train_raw)
            X_valid_qn = qn.transform(X_valid_raw)

            # B. Fold-training label-free ComBat
            combat_fit = neuroCombat(
                dat=X_train_qn.T.to_numpy(dtype=float),
                covars=pd.DataFrame({"batch": batch_train.astype(str).to_numpy()}),
                batch_col="batch", categorical_cols=[], continuous_cols=[],
                eb=True, parametric=True, mean_only=False,
            )
            X_train_combat = pd.DataFrame(
                combat_fit["data"].T, index=X_train_qn.index, columns=X_train_qn.columns
            )
            combat_valid = neuroCombatFromTraining(
                dat=X_valid_qn.T.to_numpy(dtype=float),
                batch=batch_valid.astype(str).to_numpy(),
                estimates=combat_fit["estimates"],
            )
            X_valid_combat = pd.DataFrame(
                combat_valid["data"].T, index=X_valid_qn.index, columns=X_valid_qn.columns
            )

            # C. Fold-training bottom 10% variance filter
            training_variance = X_train_combat.var(axis=0, ddof=1)
            variance_cutoff = float(training_variance.quantile(BOTTOM_VARIANCE_FRACTION))
            kept_genes = training_variance.loc[training_variance > variance_cutoff].index.tolist()
            X_train = X_train_combat[kept_genes].copy()
            X_valid = X_valid_combat[kept_genes].copy()

            # D. Exact archived feature-selector implementations
            ranking_output = exact_rankings(X_train, np.asarray(y_train, dtype=int), ppi_degree, cfg)
            selected: dict[str, list[str]] = {}
            panel_rows = []
            for method in METHODS:
                table = pd.DataFrame(ranking_output[method]).copy()
                genes = select_top_k(table, FINAL_K, RANK_COLUMNS[method])
                if len(genes) != FINAL_K or not set(genes).issubset(X_train.columns):
                    raise RuntimeError(f"Repeat {repeat} fold {fold} {method}: invalid top-20 panel")
                selected[method] = list(map(str, genes))
                for rank, gene in enumerate(genes, 1):
                    panel_rows.append(
                        {"Repeat": repeat, "Fold": fold, "Method": method, "k": FINAL_K,
                         "Selection_rank": rank, "Gene": str(gene)}
                    )

            metric_rows = []
            pred_rows = []
            for method, genes in selected.items():
                for classifier, estimator in exact_models.items():
                    model = clone(estimator)
                    model.fit(X_train[genes], np.asarray(y_train, dtype=int))
                    prob = model.predict_proba(X_valid[genes])[:, 1]
                    metric_rows.append(
                        {"Repeat": repeat, "Fold": fold, "Method": method, "Classifier": classifier,
                         "k": FINAL_K, **_calculate_metrics(y_valid, prob)}
                    )
                    pred_rows.extend(
                        {"Repeat": repeat, "Fold": fold, "Method": method, "Classifier": classifier,
                         "k": FINAL_K, "Sample_ID": str(sid), "True_Label": int(label),
                         "Probability": float(p)}
                        for sid, label, p in zip(valid_ids, y_valid.to_numpy(dtype=int), prob)
                    )

            panels_df = pd.DataFrame(panel_rows)
            metrics_df = pd.DataFrame(metric_rows)
            pred_df = pd.DataFrame(pred_rows)
            audit = {
                "Repeat": repeat, "Fold": fold,
                "Train_n": len(train_ids), "Validation_n": len(valid_ids),
                "Genes_before_fold_preprocessing": X_train_raw.shape[1],
                "Genes_after_variance_filter": X_train.shape[1],
                "Variance_cutoff": variance_cutoff,
                "Train_batches": sorted(map(str, pd.unique(batch_train))),
                "Validation_batches": sorted(map(str, pd.unique(batch_valid))),
                "Validation_labels_used_for_preprocessing": False,
                "Validation_samples_used_to_fit_quantile": False,
                "Validation_samples_used_to_fit_ComBat": False,
                "Validation_samples_used_to_fit_variance_filter": False,
                "Runtime_seconds": time.perf_counter() - started,
            }
            panels_df.to_csv(panel_file, index=False)
            metrics_df.to_csv(metric_file, index=False)
            pred_df.to_csv(pred_file, index=False, compression="gzip")
            write_json(audit, audit_file)
            new_panel_parts.append(panels_df)
            new_metric_parts.append(metrics_df)
            new_pred_parts.append(pred_df)
            audit_rows.append(audit)
            print(f"[strict {repeat}/{N_REPEATS} fold {fold}/{N_FOLDS}] complete in {audit['Runtime_seconds']/60:.1f} min")

    new_panels = pd.concat(new_panel_parts, ignore_index=True) if new_panel_parts else pd.DataFrame()
    new_metrics = pd.concat(new_metric_parts, ignore_index=True) if new_metric_parts else pd.DataFrame()
    new_predictions = pd.concat(new_pred_parts, ignore_index=True) if new_pred_parts else pd.DataFrame()

    panels = pd.concat([archived_panels, new_panels], ignore_index=True).sort_values(
        ["Repeat", "Fold", "Method", "Selection_rank"]
    )
    metrics = pd.concat([archived_metrics, new_metrics], ignore_index=True).sort_values(
        ["Repeat", "Fold", "Method", "Classifier"]
    )
    panels.to_csv(out / "repeated_fold_fitted_5x5_selected_panels.csv", index=False)
    metrics.to_csv(out / "repeated_fold_fitted_5x5_fold_metrics.csv", index=False)
    if not new_predictions.empty:
        new_predictions.to_csv(out / "repeated_fold_fitted_repeats2to5_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(audit_rows).to_csv(out / "repeated_fold_fitted_repeats2to5_preprocessing_audit.csv", index=False)

    pairwise, repeat_stability = _pairwise_repeat_stability(panels)
    pairwise.to_csv(out / "repeated_fold_fitted_5x5_pairwise_jaccard.csv", index=False)
    repeat_stability.to_csv(out / "repeated_fold_fitted_5x5_stability_by_repeat.csv", index=False)
    stability_summary = (
        repeat_stability.groupby("Method")["Mean_Jaccard"]
        .agg(["mean", "std", "median", "min", "max"])
        .reset_index()
        .rename(columns={"mean": "Mean_Jaccard", "std": "SD_between_repeats",
                         "median": "Median_Jaccard", "min": "Minimum_repeat_mean",
                         "max": "Maximum_repeat_mean"})
        .sort_values("Mean_Jaccard", ascending=False)
    )

    # Nogueira across all 25 selected panels per method, using the reference 17,220-gene universe for continuity.
    nog_rows = []
    for method, block in panels.groupby("Method"):
        selected_sets = [g["Gene"].astype(str).tolist() for _, g in block.groupby(["Repeat", "Fold"])]
        nog_rows.append(
            {"Method": method, "N_selected_panels": len(selected_sets), "p": 17220, "k": FINAL_K,
             "Nogueira_stability": nogueira_stability(selected_sets, p=17220, k=FINAL_K)}
        )
    nogueira_df = pd.DataFrame(nog_rows).sort_values("Nogueira_stability", ascending=False)
    stability_summary = stability_summary.merge(nogueira_df[["Method", "Nogueira_stability"]], on="Method", how="left")
    stability_summary.to_csv(out / "repeated_fold_fitted_5x5_stability_summary.csv", index=False)
    nogueira_df.to_csv(out / "repeated_fold_fitted_5x5_nogueira.csv", index=False)

    # Repeat-level inference; five repeats are the independent matched units.
    wide = repeat_stability.pivot(index="Repeat", columns="Method", values="Mean_Jaccard").dropna()
    if all(m in wide.columns for m in METHODS):
        fr_stat, fr_p = friedmanchisquare(*[wide[m].to_numpy() for m in METHODS])
        infer_rows = [{
            "Analysis": "Repeated strict fold-fitted panel stability", "Test": "Friedman",
            "Proposed": "NIBFS", "Comparator": "All methods", "Alternative": "two-sided",
            "N_repeats": len(wide), "Statistic": fr_stat, "P_value": fr_p,
            "Mean_difference": np.nan, "Median_difference": np.nan,
        }]
        post = []
        for comp in [m for m in METHODS if m != "NIBFS"]:
            stat, pval = wilcoxon(wide["NIBFS"], wide[comp], alternative="greater", zero_method="wilcox")
            delta = wide["NIBFS"] - wide[comp]
            post.append({
                "Analysis": "Repeated strict fold-fitted panel stability", "Test": "Paired Wilcoxon",
                "Proposed": "NIBFS", "Comparator": comp, "Alternative": "greater",
                "N_repeats": len(wide), "Statistic": stat, "P_value": pval,
                "Mean_difference": float(delta.mean()), "Median_difference": float(delta.median()),
            })
        adj = multipletests([r["P_value"] for r in post], method="fdr_bh")[1]
        for r, q in zip(post, adj):
            r["BH_adjusted_p"] = float(q)
        infer_rows[0]["BH_adjusted_p"] = np.nan
        infer = pd.DataFrame(infer_rows + post)
        infer.to_csv(out / "repeated_fold_fitted_5x5_repeat_level_inference.csv", index=False)
    else:
        infer = pd.DataFrame()

    # Predictive summaries are fold-level because archived repeat-1 OOF probabilities are intentionally not regenerated.
    perf_summary = (
        metrics.groupby(["Method", "Classifier"])[["ROC_AUC", "Brier_score"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    perf_summary.columns = [
        "_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else str(col)
        for col in perf_summary.columns
    ]
    perf_summary.to_csv(out / "repeated_fold_fitted_5x5_performance_summary.csv", index=False)

    summary = {
        "status": "PASS",
        "design": "5 repeats x 5 folds strict training-fold-fitted preprocessing",
        "repeat_1_source": "archived reference 1x5 outputs; not reselected",
        "new_feature_selection_runs": 20,
        "new_repeats": [2, 3, 4, 5],
        "seeds": [42, 43, 44, 45, 46],
        "methods": METHODS,
        "stability_summary": stability_summary.to_dict("records"),
        "performance_summary": perf_summary.to_dict("records"),
        "inference": infer.to_dict("records") if not infer.empty else [],
    }
    write_json(summary, out / "repeated_fold_fitted_5x5_summary.json")
    return summary
