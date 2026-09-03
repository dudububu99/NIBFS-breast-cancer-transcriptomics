# Release notes - v1.2.5

**Date:** 2026-08-31

This is a documentation-synchronization release accompanying the final manuscript and Supplementary Material. Scientific outputs, predictions, selected panels, fold assignments, bootstrap distributions, and analysis code are unchanged from v1.2.4.

Documentation was synchronized to the executed code by clarifying that:

- the empirical stability-selection comparator uses an outer-training 1,000-gene screen based on absolute Welch-style standardized mean differences, followed by 50 stratified half-sample L1-logistic resamples;
- the `pi >= 0.90` set is descriptive and does not define the reported fixed-k top-20 panel or imply a formal error-control guarantee;
- primary LOCO ROC-AUC eligibility requires both classes and at least 15 held-out samples; and
- TCGA-BRCA uses STAR-Counts `gene_name`, within-sample ranks over the shared gene universe, and the explicitly documented rank-space classifier configurations.

The release also retains the exact p-value granularity, post-harmonization internal-assessment caveat, fixed-random-anchor interpretation, and public-secondary-data ethics wording used in the manuscript.

## Pre-publication GitHub packaging QA (2026-09-03)

Before the first public GitHub commit, repository packaging was cleaned without changing any scientific result, prediction, selected panel, fold assignment, analysis algorithm, or archived manuscript evidence. The README/config/package version metadata were harmonized to v1.2.5; Python/pytest cache artifacts were removed; deterministic manifest-build and stricter manifest/release verification were added; and small unit tests were added for existing core utility functions.


### Reviewer-facing verification convenience layer (2026-09-03)

The same v1.2.5 scientific release now includes a reviewer-first verification entry point (`scripts/verify_repository.py`), `START_HERE_REVIEWERS.md`, and isolated Windows/macOS/Linux launchers. These additions only orchestrate existing integrity/archive/unit checks; they do not rerun or alter manuscript experiments or scientific outputs.

A separate `requirements-verify.txt` is provided for reviewer/supervisor QA so that integrity checks and deterministic tests can be run without installing the full R/rpy2-enabled reproduction environment.
