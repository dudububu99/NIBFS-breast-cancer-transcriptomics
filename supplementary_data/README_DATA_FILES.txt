Machine-readable Supplementary files for the CBC submission

Tables S1-S13 are provided as CSV files where applicable. Table S10 Panel B is additionally supplied as Table_S10B_Repeated_Strict_FoldFitted_Inference.csv.
Data File D1: exact primary fold assignments for 608 development samples.
Data File D2: complete enrichment output.
Data File D3A-D3C: sample-identity check outputs (duplicate GSM rows, metadata flags, and JSON summary).
Data File D4: official GSE15852 matched-pair mapping used for patient-pair bootstrap.
Figure-source CSV files are included for Supplementary Figures S5-S8, covering repeated strict fold-fitted stability, the stability-selection comparator, paired GSE15852 bootstrap summaries, and the stability-discrimination scatter.

The repeated strict fold-fitted analysis uses five repeated five-fold allocations under the same protocol. The stability-selection comparator uses the same repeated evaluation partitions as the other selectors; within each outer-training fold, it applies a 1,000-gene screen based on absolute Welch-style standardized mean differences, followed by 50 stratified half-sample L1-logistic resamples. The pi >= 0.90 set is descriptive and does not define the reported fixed-k top-20 panel. GSE15852 confidence intervals are computed from fixed prediction probabilities using patient-pair cluster bootstrap. Figure-source files are plotting-only and do not alter any scientific result.


Submission-sync note (2026-09-02): Table S8 now includes the repeated 5x5 Nogueira row already reported in the Supplementary PDF, and Table S10 includes the LR/RF/LightGBM ROC-AUC mean/SD columns already reported in the same PDF. No scientific result was recomputed or changed.
