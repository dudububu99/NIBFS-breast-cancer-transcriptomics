# additional CBC analyses: manuscript update guide

This file is generated from the additional-analysis outputs. It is intentionally separate from the v1.1.0 reference archive. Do not overwrite the old result folders.

## 1. GSE15852 paired correction
GSE15852 contains 43 matched tumor-normal pairs (86 samples). The previously saved NIBFS model probabilities were reused without rerunning feature selection or fitting models during the bootstrap calculation. Pair-cluster bootstrap (2,000 replicates) should replace the earlier class-stratified confidence intervals.

Updated ROC-AUC results:
LR 0.8697 (0.8031-0.9324); RF 0.9283 (0.8729-0.9757); LGBM 0.9281 (0.8680-0.9740)

Suggested Methods sentence:
"GSE15852 comprises 43 matched tumor-normal pairs. Confidence intervals for ROC-AUC were therefore recomputed using a 2,000-iteration patient-pair cluster bootstrap while retaining the previously frozen panel and saved discovery-model predictions."

## 2. Repeated strict training-fold-fitted sensitivity analysis
Design: 5 repeats x 5 folds. Repeat 1 corresponds to the previously generated strict 1x5 analysis; repeats 2-5 use the same protocol with random states 43-46. Previously generated results are not overwritten.

NIBFS mean repeat-level Jaccard: 0.8442 ± 0.0385
DEG-only: 0.6992 ± 0.0267
mRMR: 0.2366 ± 0.0169
LASSO: 0.2041 ± 0.0253
NIBFS Nogueira across 25 panels: 0.9189

Suggested Results framing:
"The stricter training-fold-fitted analysis was extended to five repeated five-fold allocations. The ordering of panel stability remained [describe from table], indicating that the earlier result was not specific to a single fold allocation."

Use the exact values in `repeated_fold_fitted_5x5_stability_summary.csv`; do not claim statistical significance unless supported by `repeated_fold_fitted_5x5_repeat_level_inference.csv`.

## 3. Empirical stability-selection comparator
The new comparator is a screened L1-logistic subsampling stability-selection procedure evaluated under the existing same repeated 10x5 evaluation partitions. The comparison uses the corresponding NIBFS/DEG/mRMR/LASSO repeated results from the same evaluation partitions.

NIBFS repeated mean Jaccard: 0.9056
Stability-selection repeated mean Jaccard: 0.3356
NIBFS Nogueira: 0.9537
Stability-selection Nogueira: 0.5324
Stability-selection LR OOF ROC-AUC: 0.9935 ± 0.0013
Archived NIBFS LR OOF ROC-AUC: 0.9898 ± 0.0010

Recommended terminology:
"screened L1-logistic stability-selection comparator with fixed-k top-20 reporting"

Do not describe the thresholded pi>=0.90 set as formal error-controlled stability selection. The top-20 comparison is based on selection probability, then mean absolute coefficient, under a prespecified 50 half-sample resampling scheme and fold-local 1,000-gene screen.

## 4. TCGA-BRCA confidence intervals
No new TCGA training or validation run was required. The reference archive already contains participant-pair bootstrap confidence intervals. Updated ROC-AUC display:
Signed 0.9919 (0.9839-0.9977); LR 0.9987 (0.9964-0.9999); RF 0.9981 (0.9951-0.9999); LGBM 0.9973 (0.9934-0.9996)

## 5. Sample identity audit
Total discovery rows: 760
Unique GEO sample accessions: 760
Exact duplicate GSM rows: 0
Cross-cohort metadata-fingerprint flag rows: 0

Use cautious wording: the audit establishes whether GEO sample accession IDs are duplicated. Public metadata alone cannot definitively prove or exclude shared biological specimens across studies.

## 6. What remains unchanged
Do NOT rerun or replace the primary 5-fold analysis, repeated 10x5 NIBFS/DEG/mRMR/LASSO analysis, random-anchor 1,000-control experiment, degree-preserving rewiring, LOCO, frozen top-20 panel, GSE70947 evaluation, or original TCGA model predictions. Those remain the v1.1.0 reference results.
