# Reproducibility and verification

This repository supports two complementary activities:

1. **Release verification** checks that the archived files are intact and that machine-readable evidence agrees with the values reported in the accompanying manuscript. This does not rerun the computational experiments.
2. **End-to-end reproduction** reruns the analysis workflow from public data/resources using the source code and instructions in `README.md`.

## Quick release verification

From the repository root:

```bash
python -m pip install -r requirements-verify.txt
python scripts/verify_repository.py --with-tests
```

Expected status for release v1.2.5:

```text
PASS  Release metadata consistency
PASS  Cryptographic file-manifest integrity
PASS  Machine-readable paper/archive verification
PASS  Deterministic unit/smoke tests

REPOSITORY VERIFICATION PASS
```

The archive verifier currently performs **22 manuscript-evidence checks**, and the deterministic test suite contains **8 tests**.

### Windows

Double-click `VERIFY_REPOSITORY_WINDOWS.bat`. The launcher creates an isolated `.venv_verify` environment, installs only the lightweight verification dependencies, and runs the same checks.

### macOS / Linux

```bash
bash verify_repository_unix.sh
```

## What the checks verify

- `scripts/verify_release_metadata.py`: release-version/date consistency and repository hygiene.
- `scripts/verify_file_manifest.py`: SHA-256 hashes, byte sizes, path safety, and exact non-transient file coverage.
- `scripts/verify_paper_archive.py`: frozen-panel identity, primary fold assignments, stability values, selected comparator audits, network controls, and external-evaluation evidence.
- `pytest -q`: deterministic tests for core ranking, stability, metric, and TCGA helper behavior.

These checks do not download GEO, STRING, or TCGA resources and do not rerun feature selection, classification, repeated cross-validation, LOCO, or external-validation experiments. Passing them demonstrates release integrity and internal consistency; it is not a substitute for an independent scientific replication.

## End-to-end reproduction

The complete scientific workflow is described in `README.md`. The main entry points are:

- `notebooks/01_main_NIBFS_core.ipynb` for the core discovery/development workflow;
- `notebooks/02_fold_fitted_all_comparators_1x5.ipynb` for the fold-fitted sensitivity analysis;
- `notebooks/03_TCGA_BRCA_RNAseq_external_validation.ipynb` for the cross-technology TCGA-BRCA analysis;
- `additional_robustness_analyses/` for additional robustness analyses reported in the manuscript.

Public raw data are intentionally not vendored. Accession identifiers are provided in `data_accession_list.csv`, and the repository contains the code/configuration needed to reconstruct the analyses from the corresponding public resources.
