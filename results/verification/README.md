# Verification outputs

This folder contains compact machine-readable outputs from analyses reported in the manuscript. It is intentionally smaller than the complete runtime archive.

- `fold_fitted/`: all-comparator fold-fitted preprocessing results and audit tables.
- `degree_preserving/`: 100-replicate degree-preserving rewiring audit summaries.
- `tcga_brca/`: paired TCGA-BRCA frozen-panel validation summaries, predictions, manifest, and direction-replication tables. The ~34 MB full processed TPM matrix is intentionally excluded because the source data are public and the repository contains the analysis engine and manifest needed to reconstruct it.
- `../supplementary_data/`: manuscript-facing machine-readable supporting tables, including the 608-sample exact primary fold assignments and B=1000 topology-permutation summary.

Run `python scripts/verify_paper_archive.py` for consistency checks against the archived manuscript values.
