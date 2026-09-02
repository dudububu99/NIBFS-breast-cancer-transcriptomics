# NIBFS v1.2.5 submission-sync patch audit — 2026-09-02

## Scope

This patch synchronizes the GitHub/reviewer repository with the canonical synchronized CBC manuscript and Supplementary Material while preserving the reported scientific analysis.

## Scientific outputs intentionally preserved

The repeated strict training-fold-fitted 5x5 results remain:

- NIBFS mean Jaccard: 0.8442348955392435 (reported 0.8442 ± 0.0385)
- NIBFS Nogueira stability: 0.9189058139534884 (reported 0.9189)
- NIBFS LR ROC-AUC: 0.8647040426402794 ± 0.04139727041460157
- NIBFS RF ROC-AUC: 0.9320594758998169 ± 0.025663779040762506
- NIBFS LightGBM ROC-AUC: 0.9381175595371474 ± 0.020196400164936987

No archived result under `results/` or `additional_robustness_analyses/results_additional/` was modified.
No core scientific implementation under `src/` was modified, except `src/__init__.py` version metadata.

## Changes made

1. Fixed the syntax typo in the additional-analysis Run-All notebook: `from src_additional.run_all import main`.
2. Fixed the displayed output folder from `results_additional analysis` to `results_additional`.
3. Aligned README/config/package metadata with the already declared release version 1.2.5.
4. Added the repeated strict 5x5 row already reported in the Supplementary PDF to `Table_S8_Nogueira_Stability.csv`.
5. Added the already archived LR/RF/LightGBM ROC-AUC mean/SD columns to the machine-readable Table S10 source.
6. Added `Table_S10B_Repeated_Strict_FoldFitted_Inference.csv` from the archived exact repeat-level inference output.
7. Extended the archive verifier to check the synchronized Table S8/S10 values.

## Validation

Baseline before patch:
- original file manifest: PASS (193 files)
- paper/reviewer archive verifier: PASS (22 checks)
- unit tests: PASS (4 tests)

After patch:
- Python source syntax: PASS
- notebook code-cell syntax: PASS
- paper/reviewer archive verifier: PASS (24 checks)
- unit tests: PASS (4 tests)
- regenerated SHA-256 file manifest: PASS (196 files)

## Interpretation

This is a synchronization/reproducibility patch only. It does not replace the manuscript's prespecified fixed-structural-anchor 5x5 analysis with the later post-hoc structural-coverage sensitivity audit.
