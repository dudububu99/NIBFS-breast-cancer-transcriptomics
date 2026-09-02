# NIBFS reproducibility repository

**Manuscript:** *Stability-aware feature selection with fixed-rank fusion for reproducible breast cancer gene prioritization*  
**Authors:** Dian Yuliati, Mohammad Isa Irawan, Muhammad Syifa'ul Mufid  
**Repository release:** 1.2.5 (2026-08-31)

## Purpose

This repository provides the reference implementation and verification archive for Network-Informed Borda Feature Selection (NIBFS). It contains the feature-selection logic, preprocessing and evaluation workflow, robustness analyses, machine-readable supplementary tables, and compact verification outputs supporting the accompanying manuscript.

The frozen top-20 panel is presented as a computationally prioritized candidate panel, not as a clinically validated diagnostic or prognostic signature. The study does not assert predictive superiority over all comparator methods; its primary methodological emphasis is panel reproducibility under resampling and the behavior of fixed-rank fusion.

## What is included

- modular core implementation in `src/`;
- analysis notebooks in `notebooks/`;
- analysis configuration in `config.yaml`;
- discovery and external accession manifest in `data_accession_list.csv`;
- exact primary fold assignments plus machine-readable Supplementary Tables S1--S13 and Data Files D1--D4 in `supplementary_data/`;
- compact verification outputs for fold-fitted preprocessing, Nogueira stability, repeated-resampling inference, degree-preserving rewiring, and TCGA-BRCA in `results/verification/`;
- the KM Plotter RFS input used for post-selection survival context in `manual_inputs/`;
- environment helpers, tests, an archive-verification script, and a manuscript-to-code map.

Raw GEO, STRING, HGNC, and GDC files are not committed. The supplied code downloads public resources or reconstructs them from public identifiers.

## Additional robustness analyses

The directory `additional_robustness_analyses/` contains the pair-aware GSE15852 uncertainty analysis, repeated strict training-fold-fitted evaluation, fixed-k stability-selection comparison, sample-identity checks, and publication-figure source data reported in the manuscript. The accompanying run-all notebook and source modules reproduce these analyses from the supplied reference inputs.

The stability-selection comparator uses a fold-local 1,000-gene screen based on absolute Welch-style standardized mean differences computed from the outer-training samples, followed by 50 stratified half-sample L1-logistic resamples. The fixed top-20 panel is used only to match panel size for overlap comparison; the `pi >= 0.90` set is descriptive, does not define the reported top-20 panel, and is not interpreted as a formal error-control guarantee. GSE15852 pair-aware uncertainty is computed from fixed prediction probabilities without rerunning feature selection or fitting models during the bootstrap calculation.

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

### Chance-corrected Nogueira stability

The manuscript also reports the Nogueira chance-corrected stability estimator for the primary, repeated, and training-fold-fitted settings. The estimator is recomputed from compact selected-panel tables without rerunning preprocessing, feature selection, or classifiers:

```bash
python scripts/postprocess_nogueira_stability.py
```

The implementation is in `src/stability_estimators.py`; source panels and the reported/recomputed tables are in `results/verification/stability/`.

### Training-fold-fitted preprocessing

Run `notebooks/02_fold_fitted_all_comparators_1x5.ipynb`. It uses the latest completed core run and compares NIBFS, DEG-only, mRMR, and LASSO under fold-fitted quantile normalization, label-free ComBat transfer, variance filtering, feature selection, and model fitting.

### Transfer-safe LOCO

The LOCO script requires the active raw cohort objects created by notebook 01 (`all_expression`, `all_metadata`, `probe_map`, `string_edges`). It excludes the held-out cohort from representative-probe selection and all training-fitted steps and does not apply ordinary ComBat to the unseen cohort. Primary ROC-AUC summaries use cohorts containing both classes and at least 15 held-out samples (`PRIMARY_MIN_TOTAL = 15`).

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

The reported analysis uses 1,000 permutations. `permuted_topology_control_100x_V1.py` creates the initial 100-permutation run, and `permuted_topology_control_extend_100_to_1000_V3.py` extends or reuses it to 1,000.

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

Run `notebooks/03_TCGA_BRCA_RNAseq_external_validation.ipynb`. The reusable analysis engine is `src/tcga_brca_rnaseq_external_validation.py`. The analysis uses participant-matched primary-tumor/solid-tissue-normal STAR-Counts, takes gene symbols from the STAR-Counts `gene_name` field, converts both development microarray and TCGA RNA-seq expression to within-sample percentile ranks over the shared gene universe, and evaluates the frozen panel in cross-technology rank space. Rank-space LR uses L2/lbfgs with `C=1` and `max_iter=5000`; RF uses 500 trees, square-root feature sampling, and balanced class weights; LightGBM uses 300 estimators, learning rate 0.05, 31 leaves, unit subsampling/column sampling, and `is_unbalance=True`. External labels are not used for feature selection, model fitting, or tuning.

## Verification

Run:

```bash
python scripts/verify_paper_archive.py
```

The script checks frozen-panel identity, the exact primary fold assignment, Nogueira stability, repeated stability inference, LASSO nonzero-coefficient auditing, fold-fitted stability, the 100-replicate degree-preserving control, the B=1000 topology-permutation summary, RWR-DEG reference values, external-dataset coverage, TCGA pair counts and panel coverage, and the KM Plotter panel input.

## Key verification values

The compact verification materials include the single strict five-fold mean Jaccard values (NIBFS 0.8883, DEG-only 0.7120, mRMR 0.2225, LASSO 0.1879). The repeated strict 5x5 analysis reports mean repeat-level Jaccard values of NIBFS 0.8442, DEG-only 0.6992, mRMR 0.2366, and LASSO 0.2041, with the same stability ordering. The archive also contains the Nogueira stability estimates and repeat-level stability inference. It also records the 100-replicate degree-preserving control and the paired TCGA-BRCA analysis (113 pairs / 226 samples, with all 20 frozen genes available and direction-concordant).

Machine-readable values are provided in `results/verification/`.

## Repository map

See `docs/PAPER_TO_CODE_MAP.md` for a manuscript-analysis-to-source mapping and `docs/REPOSITORY_QA.md` for repository verification information.

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
