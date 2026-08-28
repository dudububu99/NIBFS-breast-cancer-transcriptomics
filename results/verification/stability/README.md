# Stability verification archive

This directory contains compact selected-panel source tables and post-processing outputs for the manuscript's stability additions.

- `primary_5fold_selected_panels_k20.csv`: 5 folds × 4 selectors × 20 genes.
- `repeated_10x5_selected_panels_k20.csv`: 50 folds × 4 selectors × 20 genes.
- `nogueira_stability_reported.csv`: values synchronized with Supplementary Table S8A.
- `repeated_paired_statistical_tests.csv`: executed repeat-level Friedman/Wilcoxon output.
- `repeated_stability_inference_reported.csv`: manuscript-facing Table S8B.

Run `python scripts/postprocess_nogueira_stability.py` to recompute the Nogueira table from archived selected panels. The training-fold-fitted selected panels are stored in `results/verification/fold_fitted/`.
