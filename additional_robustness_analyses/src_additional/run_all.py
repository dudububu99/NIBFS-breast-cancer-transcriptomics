from __future__ import annotations

import argparse
import json
import os
import sys
import shutil
from pathlib import Path

from .common import prepare_workspace, sha256_file, write_json
from .gse15852_pair_bootstrap import run as run_gse15852
from .repeated_fold_fitted_5x5 import run as run_strict
from .stability_selection_repeated_10x5 import run as run_stability_selection
from .consolidate import run as run_consolidate


def main(
    repo_zip: str,
    tables_zip: str,
    workspace: str,
    ss_subsamples: int = 50,
    ss_screen_k: int = 1000,
) -> dict:
    paths = prepare_workspace(repo_zip, tables_zip, workspace)
    input_manifest = {
        "repo_zip": str(Path(repo_zip).resolve()),
        "repo_zip_sha256": sha256_file(repo_zip),
        "tables_zip": str(Path(tables_zip).resolve()),
        "tables_zip_sha256": sha256_file(tables_zip),
        "workspace": str(paths.workspace),
        "old_inputs_modified": False,
        "policy": "All new outputs are written only under the additional-analysis workspace.",
    }
    write_json(input_manifest, paths.workspace / "INPUT_LOCK.json")

    required = [
        paths.repo_dir / "config.yaml",
        paths.repo_dir / "results" / "verification" / "fold_fitted" / "fold_fitted_1x5_fold_assignments.csv",
        paths.repo_dir / "results" / "verification" / "fold_fitted" / "fold_fitted_all_methods_1x5_selected_panels.csv",
        paths.repo_dir / "results" / "verification" / "fold_fitted" / "fold_fitted_all_methods_1x5_fold_metrics.csv",
        paths.repo_dir / "results" / "verification" / "stability" / "repeated_10x5_selected_panels_k20.csv",
        paths.repo_dir / "results" / "verification" / "stability" / "repeated_paired_statistical_tests.csv",
        paths.repo_dir / "results" / "verification" / "tcga_brca" / "TCGA_BRCA_RNAseq_performance_with_pair_bootstrap_CI.csv",
        paths.tables_dir / "external_GSE15852_predictions.csv",
        paths.tables_dir / "harmonized_expression_matrix.csv.gz",
        paths.tables_dir / "train_test_split_assignments.csv",
        paths.tables_dir / "GPL570_probe_to_HGNC_mapping.csv",
        paths.tables_dir / "ppi_degree_table.csv",
        paths.tables_dir / "OUTPUT_INVENTORY_SHA256.csv",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("CBC additional analysis preflight failed; missing required reference inputs:\n" + "\n".join(missing))

    print("=" * 92)
    print("CBC ADDITIONAL ANALYSIS RUN-ALL")
    print("Reference results are used as fixed inputs for the matched additional analyses.")
    print("Workspace:", paths.workspace)
    print("=" * 92)

    print("\n[1/4] GSE15852 paired bootstrap from fixed prediction probabilities")
    gse_summary = run_gse15852(paths, iterations=2000, seed=20260829)

    print("\n[2/4] Repeated strict training-fold-fitted analysis (5x5; repeat 1 reused)")
    strict_summary = run_strict(paths)

    print("\n[3/4] Stability-selection comparator under reference repeated 10x5")
    ss_summary = run_stability_selection(
        paths,
        n_subsamples=ss_subsamples,
        screen_k=ss_screen_k,
        C=1.0,
        pi_threshold=0.90,
    )

    print("\n[4/4] Consolidating submission-ready tables, audit, and manuscript update guide")
    master = run_consolidate(paths, gse_summary, strict_summary, ss_summary)

    # Create a compact ZIP containing the additional-analysis results (excluding raw GEO cache/checkpoints).
    review_zip_base = paths.workspace / "CBC_ADDITIONAL ANALYSIS_RESULTS_FOR_REVIEW"
    review_zip = Path(shutil.make_archive(str(review_zip_base), "zip", root_dir=paths.results_dir))
    master["review_zip"] = str(review_zip)
    write_json(master, paths.results_dir / "99_submission_ready_summary" / "CBC_ADDITIONAL ANALYSIS_MASTER_SUMMARY.json")

    print("\n" + "=" * 92)
    print("ALL ADDITIONAL ANALYSIS ANALYSES COMPLETE")
    print("Results:", paths.results_dir)
    print("Review ZIP:", review_zip)
    print("Master summary:", paths.results_dir / "99_submission_ready_summary" / "CBC_ADDITIONAL ANALYSIS_MASTER_SUMMARY.json")
    print("Manuscript guide:", paths.results_dir / "99_submission_ready_summary" / "MANUSCRIPT_UPDATE_GUIDE.md")
    print("=" * 92)
    return master


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-zip", required=True)
    parser.add_argument("--tables-zip", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--ss-subsamples", type=int, default=50)
    parser.add_argument("--ss-screen-k", type=int, default=1000)
    args = parser.parse_args()
    main(
        repo_zip=args.repo_zip,
        tables_zip=args.tables_zip,
        workspace=args.workspace,
        ss_subsamples=args.ss_subsamples,
        ss_screen_k=args.ss_screen_k,
    )
