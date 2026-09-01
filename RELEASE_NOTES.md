# Release notes - v1.2.5

**Date:** 2026-08-31

This is a documentation-synchronization release accompanying the final manuscript and Supplementary Material. Scientific outputs, predictions, selected panels, fold assignments, bootstrap distributions, and analysis code are unchanged from v1.2.4.

Documentation was synchronized to the executed code by clarifying that:

- the empirical stability-selection comparator uses an outer-training 1,000-gene screen based on absolute Welch-style standardized mean differences, followed by 50 stratified half-sample L1-logistic resamples;
- the `pi >= 0.90` set is descriptive and does not define the reported fixed-k top-20 panel or imply a formal error-control guarantee;
- primary LOCO ROC-AUC eligibility requires both classes and at least 15 held-out samples; and
- TCGA-BRCA uses STAR-Counts `gene_name`, within-sample ranks over the shared gene universe, and the explicitly documented rank-space classifier configurations.

The release also retains the exact p-value granularity, post-harmonization internal-assessment caveat, fixed-random-anchor interpretation, and public-secondary-data ethics wording used in the manuscript.
