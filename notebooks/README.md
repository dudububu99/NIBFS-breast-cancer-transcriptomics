# Notebooks

1. `01_main_NIBFS_core.ipynb` — raw-to-core NIBFS analysis, primary five-fold evaluation, frozen top-20 panel, held-out assessment, GSE15852, sensitivity, biological context, and KAN bridge export.
2. `02_fold_fitted_all_comparators_1x5.ipynb` — training-fold-fitted preprocessing comparison for NIBFS, DEG-only, mRMR, and LASSO. Run after notebook 01.
3. `03_TCGA_BRCA_RNAseq_external_validation.ipynb` — paired TCGA-BRCA GDC STAR-Counts cross-technology validation. Run after notebook 01.

The notebooks use repository-relative source modules and package discovery. Machine-readable outputs from the completed analyses are retained under `results/verification/`.
