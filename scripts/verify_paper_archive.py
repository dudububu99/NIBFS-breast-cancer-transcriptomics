#!/usr/bin/env python3
"""Verify machine-readable manuscript results and frozen-panel consistency."""

from __future__ import annotations

from pathlib import Path
import json
import math
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def fail(message: str) -> None:
    raise AssertionError(message)


def close(actual: float, expected: float, tol: float = 5e-4) -> None:
    if not math.isfinite(float(actual)) or abs(float(actual) - expected) > tol:
        fail(f"Expected {expected} ± {tol}, found {actual}")


def main() -> None:
    checks: list[tuple[str, bool]] = []

    # Exact primary-development fold assignment archive.
    d1 = pd.read_csv(ROOT / "supplementary_data" / "Supplementary_Data_File_D1_Fold_Assignments_608.csv")
    checks.append(("D1 has 608 rows", len(d1) == 608))
    checks.append(("D1 has 608 unique GSM IDs", d1["GSM_ID"].nunique() == 608))
    checks.append(("D1 has folds 1..5", set(d1["Validation_fold"].astype(int)) == {1, 2, 3, 4, 5}))

    # Frozen panel: source expectation, executed TCGA output, and KM input must agree.
    expected = pd.read_csv(ROOT / "results" / "verification" / "tcga_brca" / "frozen_top20_expected.csv")
    expected_genes = expected.sort_values("Rank_NIBFS")["Gene"].astype(str).tolist()
    used = pd.read_csv(ROOT / "results" / "verification" / "tcga_brca" / "frozen_NIBFS_top20_used.csv")
    used_genes = used.sort_values("Rank_NIBFS")["Gene"].astype(str).tolist()
    km = pd.read_csv(ROOT / "manual_inputs" / "KMPlotter_RFS_final_k20.csv")
    checks.append(("Frozen panel has 20 genes", len(expected_genes) == 20 and len(set(expected_genes)) == 20))
    checks.append(("TCGA used exact frozen panel", used_genes == expected_genes))
    checks.append(("KM Plotter covers exact frozen panel", km["Gene"].astype(str).tolist() == expected_genes))
    checks.append(("KM Plotter has 16 BH-significant genes", int(km["BH_significant"].astype(bool).sum()) == 16))

    # Nogueira stability.
    nog = pd.read_csv(ROOT / "supplementary_data" / "Table_S8_Nogueira_Stability.csv").set_index("Evaluation_setting")
    expected_nog = {
        "Primary 5-fold": {"NIBFS": 0.949942, "DEG-only": 0.899884, "mRMR-inspired": 0.784750, "LASSO": 0.469384},
        "Repeated 10x5-fold": {"NIBFS": 0.953660, "DEG-only": 0.927630, "mRMR-inspired": 0.780480, "LASSO": 0.528963},
        "Single strict training-fold-fitted 5-fold sensitivity": {"NIBFS": 0.939930, "DEG-only": 0.829802, "mRMR-inspired": 0.359256, "LASSO": 0.309198},
    }
    for setting, vals in expected_nog.items():
        for method, value in vals.items():
            close(nog.loc[setting, method], value, tol=1e-6)
    checks.append(("Nogueira stability values", True))

    # Repeat-level inference.
    inf = pd.read_csv(ROOT / "supplementary_data" / "Table_S8B_Repeated_Stability_Inference.csv")
    wil = inf.loc[inf["Test"].eq("Paired Wilcoxon")].set_index("Comparator")
    for comparator, delta in {"DEG-only": 0.053683, "mRMR": 0.274768, "LASSO": 0.575660}.items():
        close(wil.loc[comparator, "Mean_difference"], delta, tol=1e-6)
        close(wil.loc[comparator, "BH_adjusted_p"], 0.000977, tol=1e-6)
    checks.append(("Repeated matched stability inference", True))

    # LASSO nonzero audit.
    la = pd.read_csv(ROOT / "supplementary_data" / "Table_S9_LASSO_Nonzero_Audit.csv")
    top20 = la["Top20_all_nonzero"].astype(str).str.lower().isin({"true", "1", "yes"})
    checks.append(("LASSO audit reports all top-20 nonzero", bool(top20.all())))
    rep = la.loc[la["Setting"].astype(str).str.contains("Repeated", case=False, na=False)]
    if not rep.empty and "Rank20_abs_coefficient" in rep.columns:
        checks.append(("Repeated LASSO rank-20 coefficient remains nonzero", float(pd.to_numeric(rep["Rank20_abs_coefficient"]).min()) > 0.0))

    # Fold-fitted all-comparator stability reported in the manuscript.
    ff = pd.read_csv(ROOT / "results" / "verification" / "fold_fitted" / "fold_fitted_all_methods_stability_summary.csv").set_index("Method")
    for method, expected_value in {"NIBFS": 0.8883, "DEG-only": 0.7120, "mRMR": 0.2225, "LASSO": 0.1879}.items():
        close(ff.loc[method, "Mean_Jaccard"], expected_value)
    checks.append(("Fold-fitted manuscript stability values", True))

    # Degree-preserving audit.
    degree = json.loads((ROOT / "results" / "verification" / "degree_preserving" / "degree_preserving_null_summary.json").read_text())
    checks.append(("Degree null completed 100 replicates", int(degree["n_nulls_completed"]) == 100))
    checks.append(("Degrees preserved exactly", degree["all_degrees_preserved_exactly"] is True))
    checks.append(("Topology ranks identical", degree["all_topology_ranks_identical"] is True))
    checks.append(("Adjacency changed", float(degree["mean_sampled_neighbor_change_fraction"]) > 0.90))

    # Topology-gene permutation control B=1000.
    perm = pd.read_csv(ROOT / "supplementary_data" / "Table_S2B_Topology_Permutation.csv")
    checks.append(("Topology permutation uses B=1000", set(perm["N_permutations"].astype(int)) == {1000}))
    stab_row = perm.loc[perm["Outcome"].eq("Panel stability")].iloc[0]
    close(stab_row["Observed_real_STRING"], 0.905628, tol=1e-6)

    # RWR-DEG reference comparison.
    rwr = pd.read_csv(ROOT / "supplementary_data" / "Table_S2A_RWR_Comparison.csv")
    rwr50 = rwr.loc[(rwr["Method"].eq("RWR-DEG")) & (rwr["Restart"].astype(str).eq("0.50"))].iloc[0]
    close(rwr50["Mean_Jaccard"], 0.83878, tol=1e-6)
    close(rwr50["LR_OOF_ROC_AUC"], 0.993929, tol=1e-6)
    checks.append(("RWR-DEG r=0.50 comparison", True))

    # External summary and TCGA pair audit.
    ext = pd.read_csv(ROOT / "supplementary_data" / "Table_S5A_External_Summary.csv")
    checks.append(("External summary contains GSE15852, GSE70947, TCGA-BRCA", set(ext["Dataset"]) == {"GSE15852", "GSE70947", "TCGA-BRCA"}))
    tcga = json.loads((ROOT / "results" / "verification" / "tcga_brca" / "TCGA_BRCA_RNAseq_validation_summary.json").read_text())
    checks.append(("TCGA has 113 pairs / 226 samples", tcga["external_pairs"] == 113 and tcga["external_samples"] == 226))
    checks.append(("TCGA has 20/20 panel coverage", tcga["complete_panel_coverage"] is True and tcga["direction_concordant_genes"] == 20))
    checks.append(("TCGA labels unused for fitting", tcga["external_labels_used_for_fitting"] is False))

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(("PASS" if ok else "FAIL") + " - " + name)
    if failed:
        print("\nArchive verification FAILED:")
        for name in failed:
            print(" -", name)
        sys.exit(1)
    print(f"\nARCHIVE VERIFICATION PASS ({len(checks)} checks)")


if __name__ == "__main__":
    main()
