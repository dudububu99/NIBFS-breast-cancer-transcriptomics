# NIBFS reproducibility repository

**Manuscript:** Stability-aware feature selection with fixed-rank fusion for reproducible breast cancer gene prioritization

**Authors:** Dian Yuliati, Mohammad Isa Irawan, Muhammad Syifa’ul Mufid

**Version:** 1.0.0 (paper-facing release, 2026-08-14)


## Purpose

This repository is a paper-facing, auditable implementation of Network-Informed Borda Feature Selection (NIBFS) and a verification archive for the machine-readable results reported with the manuscript. It intentionally does **not** claim a cold-start reproduction from redistributed raw GEO, STRING, or TCGA data. Raw/public source data should be obtained from their original repositories using the accession identifiers documented in `docs/DATA_PROVENANCE.md`.

## What is included

- core NIBFS rank-fusion implementation;
- R/limma fold-local statistical ranking script;
- classifier configurations matching the manuscript;
- a primary-CV reference runner that accepts a supplied preprocessed development matrix and fixed STRING ranking;
- exact sample-level five-fold assignments for the 608 development samples;
- machine-readable Supplementary Tables S1-S7 and Data Files D1-D2;
- an archived-result audit that checks the headline values against those files;
- manuscript and Supplementary Material PDFs;
- exact LaTeX source archives for both the main manuscript and Supplementary Material.

## Quick verification

```bash
python -m pip install -r requirements.txt
python scripts/01_audit_archived_results.py
python -m pytest -q
```

A successful verification ends with `ARCHIVED RESULT AUDIT PASSED`.

## Full analysis prerequisites

The primary-CV reference runner requires a sample-by-gene preprocessed development matrix and the fixed STRING-derived structural ranking. These input-dependent objects are not redistributed or reconstructed in this release; they should be obtained or generated according to the documented provenance and preprocessing requirements. See `docs/FULL_RERUN_INPUT_REQUIREMENTS.md`.

## Citation and archival metadata

`CITATION.cff` is provided for GitHub. `.zenodo.json` is provided for Zenodo. The Zenodo DOI and GitHub URL are intentionally absent until they are actually created.

## License

## License

The software and code in this repository are released under the MIT License. See the [`LICENSE`](LICENSE) file for details.

Copyright © 2026 Dian Yuliati, Mohammad Isa Irawan, and Muhammad Syifa’ul Mufid.
