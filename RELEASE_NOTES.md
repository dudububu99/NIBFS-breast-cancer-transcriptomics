# Release notes — v1.0.0 paper-facing

Date: 2026-08-15

This release is the curated repository intended to accompany the manuscript *Stability-aware feature selection with fixed-rank fusion for reproducible breast cancer gene prioritization*.

Curation actions:

- normalized duplicate filenames and kept one canonical copy of each source module;
- retained the final paper-facing source for repeated CV, transfer-safe LOCO, RWR-DEG, 1,000-permutation topology control, degree-preserving rewiring, GSE70947, and TCGA-BRCA;
- converted the main, fold-fitted, and TCGA notebooks into repository-relative paper-facing copies and removed historical failed-resume output cells;
- retained exact primary fold assignments and machine-readable supplementary tables;
- retained compact executed-output verification tables while excluding public raw downloads, backup archives, checkpoints, and the large TCGA full processed TPM matrix;
- normalized repository release metadata to v1.0.0 without changing the scientific configuration parameters;
- added archive consistency checks and tests.

Validation performed before packaging:

- Python source compilation: PASS;
- archive verification: 18/18 checks PASS;
- tests: 4/4 PASS.
