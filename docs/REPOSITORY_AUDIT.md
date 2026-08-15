# Repository curation notes

This repository contains the analysis paths and compact verification materials supporting the manuscript. To keep the repository reproducible and manageable, the following are not included:

- duplicate local copies of source files;
- superseded exploratory or recovery notebooks;
- raw GEO, HGNC, STRING, and GDC downloads that can be retrieved from public sources by the supplied code;
- the large processed TCGA full TPM matrix; compact predictions, manifests, validation summaries, and gene-direction tables are retained instead;
- backup ZIP archives and temporary checkpoint folders;
- obsolete environment-specific paths that are not required by the current analysis workflow.

The repository notebooks use repository-relative package discovery and the latest completed core run where required. Scientific parameters in `config.yaml` are preserved, and the repository version is `1.0.0`.
