# Repository curation audit

The repository was curated from the analysis files used for the manuscript. The following were intentionally excluded from the public-facing package:

- duplicate files whose names differed only by `(1)`, `(2)`, `(3)`, or `(4)`;
- failed TCGA path-recovery notebooks and obsolete resume wrappers;
- historical diagnostic/remount cells from the main notebook;
- raw GEO, HGNC, STRING, and GDC downloads that can be obtained from public sources by the supplied code;
- the large processed TCGA full TPM matrix; compact predictions, manifests, validation summaries, and gene-direction tables are retained instead;
- backup ZIPs and temporary checkpoint folders;
- old/alternative notebook paths that are not needed to reproduce the paper-facing analyses.

The cleaned notebooks preserve the scientific analysis code while replacing obsolete absolute Google Drive paths with repository-relative discovery of the package marker and latest completed core run. Historical error outputs were removed from the paper-facing notebook copies.

Scientific parameters in `config.yaml` were preserved. The repository release label was normalized to `1.0.0-paper-facing`; this is release metadata rather than an analytical change.
