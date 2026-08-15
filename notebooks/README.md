# Notebooks

1. `01_main_NIBFS_core.ipynb` — fresh raw-to-core NIBFS analysis, primary five-fold evaluation, frozen top-20 panel, held-out assessment, GSE15852, sensitivity, biological context, and KAN bridge export.
2. `02_fold_fitted_all_comparators_1x5.ipynb` — stricter training-fold-fitted preprocessing comparison for NIBFS, DEG-only, mRMR, and LASSO. Run after notebook 01.
3. `03_TCGA_BRCA_RNAseq_external_validation.ipynb` — paired TCGA-BRCA GDC STAR-Counts cross-technology validation. Run after notebook 01.

The notebooks in this folder are paper-facing cleaned copies. Historical path-recovery cells, failed resume attempts, and obsolete Drive-specific wrappers were intentionally excluded. Machine-readable outputs from the executed analyses are retained under `results/verification/`.
