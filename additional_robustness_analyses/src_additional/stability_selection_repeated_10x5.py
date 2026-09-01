from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from .common import AdditionalAnalysisPaths, add_repo_to_path, write_json

REPEATS = 10
FOLDS = 5
BASE_SEED = 42
FINAL_K = 20
P_UNIVERSE = 17220
METHOD_NAME = "Stability-selection"


def _load_development(paths: AdditionalAnalysisPaths) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    expression_path = paths.tables_dir / "harmonized_expression_matrix.csv.gz"
    if not expression_path.is_file():
        expression_path = paths.tables_dir / "harmonized_expression_matrix.csv"
    split_path = paths.tables_dir / "train_test_split_assignments.csv"
    if not expression_path.is_file() or not split_path.is_file():
        raise FileNotFoundError("Harmonized expression matrix / split assignments missing from reference tables ZIP.")
    expression = pd.read_csv(expression_path)
    sample_col = "GSM_ID" if "GSM_ID" in expression.columns else expression.columns[0]
    expression[sample_col] = expression[sample_col].astype(str)
    expression = expression.set_index(sample_col)
    split = pd.read_csv(split_path)
    dev = split.loc[split["Set"].astype(str).eq("Model-development")].drop_duplicates("GSM_ID").copy()
    dev["GSM_ID"] = dev["GSM_ID"].astype(str)
    X = expression.loc[dev["GSM_ID"].tolist()].copy()
    y = pd.to_numeric(dev["Label_binary"]).astype(int).to_numpy()
    if X.shape != (608, 17220):
        raise RuntimeError(f"Expected reference development matrix (608, 17220), found {X.shape}")
    return X, y, dev.reset_index(drop=True)


def _welch_screen(X: pd.DataFrame, y: np.ndarray, screen_k: int) -> list[str]:
    """Fast fold-local supervised screen used only to make the empirical comparator tractable.

    It uses absolute Welch-style standardized mean differences computed from the outer
    training fold. No validation sample is used. The stability-selection stage itself
    then operates on these screened genes under repeated half-sampling.
    """
    arr = X.to_numpy(dtype=np.float64, copy=False)
    y = np.asarray(y, dtype=int)
    x0 = arr[y == 0]
    x1 = arr[y == 1]
    n0, n1 = len(x0), len(x1)
    m0, m1 = np.nanmean(x0, axis=0), np.nanmean(x1, axis=0)
    v0, v1 = np.nanvar(x0, axis=0, ddof=1), np.nanvar(x1, axis=0, ddof=1)
    denom = np.sqrt(v0 / max(n0, 1) + v1 / max(n1, 1))
    score = np.abs(m1 - m0) / np.where(denom > 0, denom, np.inf)
    genes = np.asarray(X.columns.astype(str))
    # deterministic: decreasing score, alphabetical gene for exact ties
    order = np.lexsort((genes, -np.nan_to_num(score, nan=-np.inf)))
    return genes[order[: min(screen_k, len(genes))]].tolist()


def _stability_panel(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
    n_subsamples: int,
    screen_k: int,
    C: float,
    pi_threshold: float,
) -> tuple[list[str], pd.DataFrame, dict]:
    screened = _welch_screen(X_train, y_train, screen_k)
    Xs = X_train[screened].to_numpy(dtype=np.float64)
    # Outer-training scaling is valid because no outer validation samples are used.
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xs)
    splitter = StratifiedShuffleSplit(
        n_splits=n_subsamples,
        train_size=0.5,
        random_state=seed,
    )
    counts = np.zeros(len(screened), dtype=np.int32)
    abscoef_sum = np.zeros(len(screened), dtype=np.float64)
    converged = 0
    iterations_used = []

    for b, (sub_idx, _) in enumerate(splitter.split(Xs, y_train), 1):
        model = LogisticRegression(
            C=C,
            penalty="l1",
            solver="liblinear",
            class_weight="balanced",
            max_iter=3000,
            tol=1e-4,
            random_state=seed + b,
        )
        model.fit(Xs[sub_idx], y_train[sub_idx])
        coef = model.coef_.ravel()
        nonzero = np.abs(coef) > 1e-12
        counts += nonzero.astype(np.int32)
        abscoef_sum += np.abs(coef)
        nit = int(np.max(np.asarray(model.n_iter_)))
        iterations_used.append(nit)
        if nit < 3000:
            converged += 1

    prob = counts / float(n_subsamples)
    mean_abs = abscoef_sum / float(n_subsamples)
    rank = pd.DataFrame(
        {
            "Gene": screened,
            "Selection_probability": prob,
            "Mean_abs_coefficient": mean_abs,
            "Selected_subsamples": counts,
            "N_subsamples": n_subsamples,
        }
    ).sort_values(
        ["Selection_probability", "Mean_abs_coefficient", "Gene"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    rank["Stability_rank"] = np.arange(1, len(rank) + 1)
    top20 = rank.head(FINAL_K)["Gene"].astype(str).tolist()
    stable_threshold = rank.loc[rank["Selection_probability"] >= pi_threshold, "Gene"].astype(str).tolist()
    audit = {
        "screen_k": screen_k,
        "n_subsamples": n_subsamples,
        "subsample_fraction": 0.5,
        "base_selector": "L1 logistic regression",
        "solver": "liblinear",
        "C": C,
        "selection_probability_threshold": pi_threshold,
        "thresholded_set_size": len(stable_threshold),
        "converged_subsamples": converged,
        "max_iterations_used": max(iterations_used) if iterations_used else None,
        "fixed_k_reporting": FINAL_K,
        "fixed_k_rule": "top 20 by selection probability, then mean absolute coefficient, then gene symbol",
        "screen_rule": "outer-training absolute Welch-style standardized mean difference",
    }
    return top20, rank, audit


def _within_repeat_stability(panels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_rows = []
    repeat_rows = []
    for (repeat, method), block in panels.groupby(["Repeat", "Method"], sort=True):
        fold_sets = {int(f): set(g["Gene"].astype(str)) for f, g in block.groupby("Fold")}
        values = []
        for a, b in combinations(sorted(fold_sets), 2):
            A, B = fold_sets[a], fold_sets[b]
            j = len(A & B) / len(A | B)
            values.append(j)
            pair_rows.append({
                "Repeat": int(repeat), "Method": method, "Fold_A": a, "Fold_B": b,
                "Intersection": len(A & B), "Union": len(A | B), "Jaccard": j,
            })
        repeat_rows.append({
            "Repeat": int(repeat), "Method": method,
            "Mean_Jaccard": float(np.mean(values)),
            "SD_pairwise_Jaccard": float(np.std(values, ddof=1)),
        })
    return pd.DataFrame(pair_rows), pd.DataFrame(repeat_rows)


def run(
    paths: AdditionalAnalysisPaths,
    n_subsamples: int = 50,
    screen_k: int = 1000,
    C: float = 1.0,
    pi_threshold: float = 0.90,
) -> dict:
    out = paths.results_dir / "03_stability_selection_repeated_10x5"
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = paths.checkpoints_dir / "stability_selection_repeated_10x5"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    add_repo_to_path(paths.repo_dir)
    from src.stability_estimators import nogueira_stability

    X, y, dev = _load_development(paths)
    existing_path = paths.repo_dir / "results" / "verification" / "stability" / "repeated_10x5_selected_panels_k20.csv"
    existing_panels = pd.read_csv(existing_path)
    existing_panels["Method"] = existing_panels["Method"].astype(str).replace({"mRMR-inspired": "mRMR"})
    if len(existing_panels) != 4000:
        raise RuntimeError(f"Expected 4,000 archived repeated panel rows, found {len(existing_panels)}")

    all_panel_parts = []
    all_pred_parts = []
    audit_rows = []

    for repeat in range(1, REPEATS + 1):
        skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=BASE_SEED + repeat - 1)
        for fold, (train_idx, valid_idx) in enumerate(skf.split(np.zeros(len(y)), y), 1):
            prefix = checkpoint_dir / f"repeat_{repeat:02d}_fold_{fold:02d}"
            panel_file = prefix.with_name(prefix.name + "_panel.csv")
            pred_file = prefix.with_name(prefix.name + "_lr_predictions.csv")
            rank_file = prefix.with_name(prefix.name + "_stability_ranking.csv.gz")
            audit_file = prefix.with_name(prefix.name + "_audit.json")
            if panel_file.is_file() and pred_file.is_file() and rank_file.is_file() and audit_file.is_file():
                print(f"[stability-selection {repeat}/{REPEATS} fold {fold}/{FOLDS}] checkpoint found — skipped")
                all_panel_parts.append(pd.read_csv(panel_file))
                all_pred_parts.append(pd.read_csv(pred_file))
                audit_rows.append(json.loads(audit_file.read_text(encoding="utf-8")))
                continue

            started = time.perf_counter()
            X_train = X.iloc[train_idx]
            X_valid = X.iloc[valid_idx]
            y_train = y[train_idx]
            y_valid = y[valid_idx]
            seed = BASE_SEED + (repeat - 1) * FOLDS + fold

            genes, ranking, audit = _stability_panel(
                X_train, y_train, seed=seed,
                n_subsamples=n_subsamples, screen_k=screen_k,
                C=C, pi_threshold=pi_threshold,
            )
            panel = pd.DataFrame({
                "Repeat": repeat, "Fold": fold, "Method": METHOD_NAME, "k": FINAL_K,
                "Selection_rank": np.arange(1, FINAL_K + 1), "Gene": genes,
            })

            # Same downstream LR family as the reference repeated analysis.
            lr = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    penalty="l2", solver="lbfgs", max_iter=5000,
                    class_weight="balanced", random_state=BASE_SEED,
                )),
            ])
            lr.fit(X_train[genes], y_train)
            prob = lr.predict_proba(X_valid[genes])[:, 1]
            pred = pd.DataFrame({
                "Repeat": repeat, "Fold": fold, "Method": METHOD_NAME,
                "Classifier": "LR", "k": FINAL_K,
                "Sample_ID": X_valid.index.astype(str),
                "True_Label": y_valid.astype(int),
                "Probability": prob.astype(float),
            })
            audit.update({
                "Repeat": repeat, "Fold": fold, "outer_seed": BASE_SEED + repeat - 1,
                "selector_seed": seed, "outer_train_n": len(train_idx), "outer_valid_n": len(valid_idx),
                "runtime_seconds": time.perf_counter() - started,
                "outer_validation_used_for_screening_or_selection": False,
            })
            panel.to_csv(panel_file, index=False)
            pred.to_csv(pred_file, index=False)
            ranking.to_csv(rank_file, index=False, compression="gzip")
            write_json(audit, audit_file)
            all_panel_parts.append(panel)
            all_pred_parts.append(pred)
            audit_rows.append(audit)
            print(f"[stability-selection {repeat}/{REPEATS} fold {fold}/{FOLDS}] complete in {audit['runtime_seconds']/60:.1f} min")

    ss_panels = pd.concat(all_panel_parts, ignore_index=True).sort_values(["Repeat", "Fold", "Selection_rank"])
    ss_predictions = pd.concat(all_pred_parts, ignore_index=True).sort_values(["Repeat", "Fold", "Sample_ID"])
    ss_panels.to_csv(out / "stability_selection_repeated_10x5_selected_panels.csv", index=False)
    ss_predictions.to_csv(out / "stability_selection_repeated_10x5_lr_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(audit_rows).to_csv(out / "stability_selection_repeated_10x5_audit.csv", index=False)

    combined_panels = pd.concat([existing_panels, ss_panels], ignore_index=True)
    pairwise, repeat_stability = _within_repeat_stability(combined_panels)
    pairwise.to_csv(out / "all_methods_plus_stability_selection_pairwise_jaccard.csv", index=False)
    repeat_stability.to_csv(out / "all_methods_plus_stability_selection_stability_by_repeat.csv", index=False)

    summary_stability = (
        repeat_stability.groupby("Method")["Mean_Jaccard"]
        .agg(["mean", "std", "median", "min", "max"])
        .reset_index()
        .rename(columns={"mean": "Mean_Jaccard", "std": "SD_between_repeats",
                         "median": "Median_Jaccard", "min": "Minimum_repeat_mean",
                         "max": "Maximum_repeat_mean"})
        .sort_values("Mean_Jaccard", ascending=False)
    )
    nog_rows = []
    for method, block in combined_panels.groupby("Method"):
        sets = [g["Gene"].astype(str).tolist() for _, g in block.groupby(["Repeat", "Fold"])]
        nog_rows.append({
            "Method": method, "N_selected_panels": len(sets), "p": P_UNIVERSE, "k": FINAL_K,
            "Nogueira_stability": nogueira_stability(sets, p=P_UNIVERSE, k=FINAL_K),
        })
    nog = pd.DataFrame(nog_rows)
    summary_stability = summary_stability.merge(nog[["Method", "Nogueira_stability"]], on="Method", how="left")
    summary_stability.to_csv(out / "all_methods_plus_stability_selection_stability_summary.csv", index=False)
    nog.to_csv(out / "all_methods_plus_stability_selection_nogueira.csv", index=False)

    # Repeat-level matched stability tests, now including the new comparator.
    wide = repeat_stability.pivot(index="Repeat", columns="Method", values="Mean_Jaccard").dropna()
    methods = [m for m in ["NIBFS", "DEG-only", "mRMR", "LASSO", METHOD_NAME] if m in wide.columns]
    fr_stat, fr_p = friedmanchisquare(*[wide[m].to_numpy() for m in methods])
    infer_rows = [{
        "Analysis": "Repeated panel stability with stability-selection comparator",
        "Test": "Friedman", "Proposed": "NIBFS", "Comparator": "All methods",
        "Alternative": "two-sided", "N_repeats": len(wide), "Statistic": fr_stat,
        "P_value": fr_p, "Mean_difference": np.nan, "Median_difference": np.nan,
        "BH_adjusted_p": np.nan,
    }]
    post = []
    for comp in [m for m in methods if m != "NIBFS"]:
        stat, pval = wilcoxon(wide["NIBFS"], wide[comp], alternative="greater", zero_method="wilcox")
        delta = wide["NIBFS"] - wide[comp]
        post.append({
            "Analysis": "Repeated panel stability with stability-selection comparator",
            "Test": "Paired Wilcoxon", "Proposed": "NIBFS", "Comparator": comp,
            "Alternative": "greater", "N_repeats": len(wide), "Statistic": stat,
            "P_value": pval, "Mean_difference": float(delta.mean()),
            "Median_difference": float(delta.median()),
        })
    qvals = multipletests([r["P_value"] for r in post], method="fdr_bh")[1]
    for row, q in zip(post, qvals):
        row["BH_adjusted_p"] = float(q)
    inference = pd.DataFrame(infer_rows + post)
    inference.to_csv(out / "all_methods_plus_stability_selection_repeat_level_inference.csv", index=False)

    # New comparator OOF AUC per repeat. Existing method predictions are not rerun.
    auc_rows = []
    for repeat, block in ss_predictions.groupby("Repeat"):
        if block["Sample_ID"].duplicated().any() or len(block) != 608:
            raise RuntimeError(f"Repeat {repeat}: expected exactly one OOF prediction per development sample")
        auc_rows.append({
            "Repeat": int(repeat),
            "Method": METHOD_NAME,
            "LR_OOF_ROC_AUC": float(roc_auc_score(block["True_Label"], block["Probability"])),
            "LR_OOF_Brier": float(brier_score_loss(block["True_Label"], block["Probability"])),
        })
    auc_by_repeat = pd.DataFrame(auc_rows)
    auc_by_repeat.to_csv(out / "stability_selection_lr_oof_metrics_by_repeat.csv", index=False)
    ss_auc_mean = float(auc_by_repeat["LR_OOF_ROC_AUC"].mean())
    ss_auc_sd = float(auc_by_repeat["LR_OOF_ROC_AUC"].std(ddof=1))

    # Reconstruct the already-reported existing repeated LR means without rerunning them.
    rwr_ref = pd.read_csv(paths.repo_dir / "supplementary_data" / "Table_S2A_RWR_Comparison.csv")
    nibfs_row = rwr_ref.loc[rwr_ref.iloc[:, 0].astype(str).str.contains("NIBFS", case=False)].iloc[0]
    # Expected columns: Method, Restart, Jaccard, SD, LR OOF AUC, SD
    numeric = pd.to_numeric(nibfs_row, errors="coerce").dropna().to_numpy(dtype=float)
    # The last two numeric values are AUC and its SD in the reference table.
    nibfs_auc_mean = float(numeric[-2])
    nibfs_auc_sd = float(numeric[-1])
    paired_tests = pd.read_csv(paths.repo_dir / "results" / "verification" / "stability" / "repeated_paired_statistical_tests.csv")
    auc_test = paired_tests.loc[
        paired_tests["Analysis"].astype(str).eq("Repeated OOF ROC-AUC")
        & paired_tests["Test"].astype(str).eq("Paired Wilcoxon")
    ].copy()
    existing_auc = {"NIBFS": nibfs_auc_mean}
    for row in auc_test.itertuples(index=False):
        existing_auc[str(row.Comparator)] = nibfs_auc_mean - float(row.Mean_difference)
    auc_compare_rows = [
        {"Method": m, "LR_OOF_ROC_AUC_mean": v, "Source": "archived reference repeated result"}
        for m, v in existing_auc.items()
    ]
    auc_compare_rows.append({
        "Method": METHOD_NAME, "LR_OOF_ROC_AUC_mean": ss_auc_mean,
        "Source": "additional analysis; 10x5 OOF predictions",
    })
    auc_compare = pd.DataFrame(auc_compare_rows)
    auc_compare.to_csv(out / "reported_existing_plus_stability_selection_lr_auc.csv", index=False)

    summary = {
        "status": "PASS",
        "design": "Empirical screened L1-logistic stability-selection comparator under the same 10 repeated five-fold evaluation partitions",
        "existing_methods_rerun": False,
        "new_outer_folds": 50,
        "subsamples_per_outer_fold": n_subsamples,
        "screen_k": screen_k,
        "C": C,
        "pi_threshold": pi_threshold,
        "stability_summary": summary_stability.to_dict("records"),
        "repeat_level_inference": inference.to_dict("records"),
        "new_comparator_lr_oof_auc_mean": ss_auc_mean,
        "new_comparator_lr_oof_auc_sd": ss_auc_sd,
        "archived_nibfs_lr_oof_auc_mean": nibfs_auc_mean,
        "archived_nibfs_lr_oof_auc_sd": nibfs_auc_sd,
        "method_label_note": (
            "For manuscript wording, describe this transparently as a screened L1-logistic "
            "stability-selection comparator with fixed-k top-20 reporting; do not imply formal "
            "family-wise error control from the pi-threshold audit."
        ),
    }
    write_json(summary, out / "stability_selection_repeated_10x5_summary.json")
    return summary
