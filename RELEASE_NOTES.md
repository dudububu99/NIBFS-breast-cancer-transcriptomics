# Release notes — v1.1.0

Date: 2026-08-28

This manuscript-synchronization release adds the robustness and audit outputs incorporated into the final CBC revision without changing the locked primary NIBFS configuration.

Additions:

- Nogueira chance-corrected stability implementation and compact archived source panels for primary 5-fold, repeated 10×5, and training-fold-fitted 5-fold analyses;
- repeat-level Friedman and paired Wilcoxon/BH stability inference;
- conservative 760-sample retrospective subject/source audit and exact 608-sample group-aware fold assignment;
- executed post-harmonization group-aware five-fold sensitivity outputs, including zero reconstructed train-validation group overlap;
- machine-readable Supplementary Tables S8--S10 and Supplementary Data Files D3--D4;
- LASSO nonzero-coefficient audit supporting the deterministic top-20 comparator implementation;
- repository verification checks updated to cover the added manuscript-facing results.

The group-aware sensitivity is explicitly post-harmonization and does not replace the primary analysis or the training-fold-fitted preprocessing sensitivity.

---

# Release notes — v1.0.0

Date: 2026-08-15

This release is the curated repository intended to accompany the manuscript *Stability-aware feature selection with fixed-rank fusion for reproducible breast cancer gene prioritization*.

Curation actions:

- normalized duplicate filenames and kept one canonical copy of each source module;
- retained the analysis source for repeated CV, transfer-safe LOCO, RWR-DEG, 1,000-permutation topology control, degree-preserving rewiring, GSE70947, and TCGA-BRCA;
- converted the main, fold-fitted, and TCGA notebooks into repository-relative analysis copies and removed obsolete recovery output cells;
- retained exact primary fold assignments and machine-readable supplementary tables;
- retained compact executed-output verification tables while excluding public raw downloads, backup archives, checkpoints, and the large TCGA full processed TPM matrix;
- normalized repository release metadata to v1.0.0 without changing the scientific configuration parameters;
- added archive consistency checks and tests.

Validation performed before packaging:

- Python source compilation: PASS;
- archive verification: 18/18 checks PASS;
- tests: 4/4 PASS.
