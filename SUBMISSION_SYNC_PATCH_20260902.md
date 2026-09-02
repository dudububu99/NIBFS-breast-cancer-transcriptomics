# Submission-sync patch — 2026-09-02

This patch does **not** change the NIBFS scientific implementation, selected panels, fold assignments, classifier predictions, or manuscript values.

Changes are limited to:

1. repair a syntax typo in `additional_robustness_analyses/00_RUN_ALL_ADDITIONAL_ROBUSTNESS_ANALYSES.ipynb` (`src_additional.run_all`) and its displayed output-directory name;
2. align repository-version metadata to the already declared release `1.2.5`;
3. add the repeated strict 5×5 Nogueira row to the machine-readable Table S8 source;
4. expose the already archived LR/RF/LightGBM mean/SD values in the machine-readable Table S10 source;
5. add a machine-readable Table S10 Panel B inference CSV; and
6. extend `scripts/verify_paper_archive.py` to verify these synchronized supplementary values.

The repeated strict training-fold-fitted analysis continues to report the prespecified legacy result used by the synchronized manuscript and Supplementary Material: NIBFS Jaccard 0.8442 ± 0.0385, Nogueira 0.9189, LR ROC-AUC 0.8647 ± 0.0414, RF 0.9321 ± 0.0257, and LightGBM 0.9381 ± 0.0202. The fixed structural source is not re-estimated from fold-specific phenotype labels or expression data.
