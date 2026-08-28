# Repository curation notes

This repository contains the analysis paths and compact verification materials supporting the manuscript. To keep the repository reproducible and manageable, the following are not included:

- duplicate local copies of source files;
- superseded exploratory or recovery notebooks;
- raw GEO, HGNC, STRING, and GDC downloads that can be retrieved from public sources by the supplied code;
- the large processed TCGA full TPM matrix; compact predictions, manifests, validation summaries, and gene-direction tables are retained instead;
- backup ZIP archives and temporary checkpoint folders;
- obsolete environment-specific paths that are not required by the current analysis workflow.

The repository notebooks use repository-relative package discovery and the latest completed core run where required. Scientific parameters in `config.yaml` are preserved, and the repository version is `1.1.0`.


## v1.1.0 manuscript-synchronization additions

Version 1.1.0 adds only manuscript-facing verification and sensitivity materials; it does not alter the locked primary scientific configuration. Added materials include compact selected-panel tables and code for Nogueira stability post-processing, the retrospective subject/source audit, the executed post-harmonization group-aware five-fold sensitivity, repeat-level stability inference, the LASSO nonzero-coefficient audit, and Supplementary Tables/Data Files S8--S10 and D3--D4. Large generated harmonized matrices remain excluded from GitHub and are reconstructed by the core workflow.
