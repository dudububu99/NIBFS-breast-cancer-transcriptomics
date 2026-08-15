# NIBFS reproducibility repository

**Manuscript:** *Stability-aware feature selection with fixed-rank fusion for reproducible breast cancer gene prioritization*  
**Authors:** Dian Yuliati, Mohammad Isa Irawan, Muhammad Syifa'ul Mufid  
**Repository release:** 1.0.0 (2026-08-15)

## Purpose

This repository provides the reference implementation and verification archive for Network-Informed Borda Feature Selection (NIBFS). It contains the feature-selection logic, preprocessing and evaluation workflow, robustness analyses, machine-readable supplementary tables, and compact verification outputs supporting the accompanying manuscript.

The frozen top-20 panel is presented as a computationally prioritized candidate panel, not as a clinically validated diagnostic or prognostic signature. The study does not assert predictive superiority over all comparator methods; its primary methodological emphasis is panel reproducibility under resampling and the behavior of fixed-rank fusion.

## What is included

- modular core implementation in `src/`;
- analysis notebooks in `notebooks/`;
- locked analysis configuration in `config.yaml`;
- discovery and external accession manifest in `data_accession_list.csv`;
- exact 608-sample primary fold assignments and machine-readable supplementary tables in `supplementary_data/`;
- compact verification outputs for fold-fitted preprocessing, degree-preserving rewiring, and TCGA-BRCA in `results/verification/`;
- the KM Plotter RFS input used for post-selection survival context in `manual_inputs/`;
- environment helpers, tests, an archive-verification script, and a manuscript-to-code map.

Raw GEO, STRING, HGNC, and GDC files are not committed. The supplied code downloads public resources or reconstructs them from public identifiers.

## Frozen top-20 panel

The frozen top-20 NIBFS panel used in the manuscript is, in rank order:

`CDK1, EGFR, CCNB1, BUB1B, FN1, CDC20, EZH2, STAT1, TOP2A, CAV1, RRM2, GNAI1, KIT, PPARG, CCNA2, UBE2C, FGF2, CCNB2, MAD2L1, FOXO1`.

The archive-verification script checks that the expected panel, TCGA analysis, and KM Plotter input agree exactly.

## Recommended environment

Google Colab provides a convenient environment for reproducing the analyses because the workflow requires both Python and R/limma. For the core dependencies:

```bash
python install_environment_colab.py
```

For a local Python environment:

```bash
python -m pip install -r requirements.txt
```

R and Bioconductor `limma` are required for fold-local differential-expression ranking. `install_environment_colab.py` installs or checks them on Colab-compatible Linux environments.

## Core analysis

Recommended execution path:

1. place or extract this repository as a single folder in Google Drive;
2. keep `NIBFS_REPRODUCIBILITY_PACKAGE.marker` at the repository root;
3. open `notebooks/01_main_NIBFS_core.ipynb` in Colab and run it from top to bottom.

The notebook creates a timestamped `runs/NIBFS_RAW_RUN_*` directory and performs discovery ingestion, probe-to-HGNC mapping, common-gene intersection, joint quantile normalization, ComBat harmonization with class preservation, variance filtering, primary five-fold feature-selection evaluation, stability analysis, full-development frozen-panel construction, post-harmonization held-out evaluation, rank-weight sensitivity, GSE15852 evaluation, enrichment/network context, RFS integration, and frozen KAN bridge export.

The modular alternative is:

```bash
python scripts/run_core_pipeline.py
```

## Additional analyses reported in the manuscript

Run these after a completed core run. In Colab, the standalone repeated/LOCO scripts are easiest to run with `%run -i` so they can reuse active notebook objects when needed.

### Repeated 10×5 CV

```python
REPEATED_PROJECT_DIR = str(PACKAGE_DIR)
REPEATED_MAX_NEW_FOLDS = None
REPEATED_FORCE_RERUN = False
%run -i str(PACKAGE_DIR / "src" / "repeated_10x5_k20_lr_V2.py")
```

### Training-fold-fitted preprocessing

Run `notebooks/02_fold_fitted_all_comparators_1x5.ipynb`. It uses the latest completed core run and compares NIBFS, DEG-only, mRMR, and LASSO under fold-fitted quantile normalization, label-free ComBat transfer, variance filtering, feature selection, and model fitting.

### Transfer-safe LOCO

The LOCO script requires the active raw cohort objects created by notebook 01 (`all_expression`, `all_metadata`, `probe_map`, `string_edges`). It excludes the held-out cohort from representative-probe selection and all training-fitted steps and does not apply ordinary ComBat to the unseen cohort.

```python
%run -i str(PACKAGE_DIR / "src" / "full_transfer_safe_loco_GENELEVEL_V2.py")
```

### RWR-DEG network baseline

Complete repeated 10×5 first so the exact repeated fold assignments exist, then run:

```python
RWR_PROJECT_DIR = str(PACKAGE_DIR)
RWR_MAX_NEW_FOLDS = None
RWR_FORCE_RERUN = False
%run -i str(PACKAGE_DIR / "src" / "rwr_deg_network_baseline_10x5_V1.py")
```

### Gene-label/topology permutation control

The archived analysis uses 1,000 permutations. `permuted_topology_control_100x_V1.py` creates the initial 100-permutation run, and `permuted_topology_control_extend_100_to_1000_V3.py` extends or reuses it to 1,000.

### Degree-preserving rewiring audit

```bash
python scripts/run_degree_preserving_null.py   --edge-file <STRING_gene_edges_eligible_genes.csv>   --output-dir results/DEGREE_PRESERVING_NULL_100   --project-dir .
```

This audit is adjacency-sensitive but degree-preserving. Because NIBFS uses degree rank as its structural component, exact invariance of the topology rank and NIBFS panel under degree-preserving rewiring is expected by construction; the informative check is that adjacency changes while degree is preserved.

### GSE70947

```python
GSE70947_PROJECT_DIR = str(PACKAGE_DIR)
GSE70947_FORCE_RERUN = False
%run -i str(PACKAGE_DIR / "src" / "external_validation_GSE70947_V5.py")
```

The script evaluates the frozen panel and models without feature reselection, model refitting, hyperparameter tuning, or external threshold optimization.

### TCGA-BRCA RNA-seq

Run `notebooks/03_TCGA_BRCA_RNAseq_external_validation.ipynb`. The reusable analysis engine is `src/tcga_brca_rnaseq_external_validation.py`. The analysis uses participant-matched primary-tumor/solid-tissue-normal STAR-Counts, converts both development microarray and TCGA RNA-seq expression to within-sample percentile ranks over the shared gene universe, and evaluates the frozen panel in cross-technology rank space. External labels are not used for feature selection or model fitting.

## Verification

Run:

```bash
python scripts/verify_paper_archive.py
```

The script checks frozen-panel identity, exact primary fold-assignment dimensions, fold-fitted stability values, the 100-replicate degree-preserving audit, the B=1000 topology-permutation summary, RWR-DEG reference values, external-dataset coverage, TCGA pair counts and panel coverage, and the KM Plotter panel input.

## Key archived values

The compact verification archive includes the manuscript-reported fold-fitted mean Jaccard values: NIBFS 0.8883, DEG-only 0.7120, mRMR 0.2225, and LASSO 0.1879. It also records the 100-replicate degree-preserving audit and the paired TCGA-BRCA analysis (113 pairs / 226 samples, with all 20 frozen genes available and direction-concordant).

Machine-readable values are provided in `results/verification/`.

## Repository map

See `docs/PAPER_TO_CODE_MAP.md` for a manuscript-analysis-to-source mapping and `docs/REPOSITORY_AUDIT.md` for the curation decisions used to assemble this reproducibility archive.

## Tests

TCGA helper smoke tests are included under `tests/`:

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```

## Data availability

The GEO accessions are listed in `data_accession_list.csv`. Independent microarray evaluation uses GSE15852 and GSE70947; cross-technology evaluation uses public TCGA-BRCA RNA-seq data. Protein-protein interaction information is obtained from STRING v12.0. Public raw data are intentionally not vendored into this repository.

## License

Source code in this repository is released under the MIT License. Third-party datasets and resources remain subject to the terms and licenses of their respective providers. See `LICENSE` for details.
