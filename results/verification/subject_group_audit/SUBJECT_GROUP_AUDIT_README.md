# Subject/group audit for the NIBFS discovery resource

Audit basis: archived `harmonized_metadata.csv`, `train_test_split_assignments.csv`,
`fold_assignments.csv`, and `repeated_fold_assignments.csv`. No model fitting,
feature selection, normalization, or existing result file was modified.

## Confirmed groupable cohorts
- GSE10780: numeric patient code parsed from sample title; 90 unique codes for 185 tissues.
- GSE111662: explicit `subject id` metadata; 9 subjects / 27 arrays.
- GSE26910: six breast tumor-stroma / matched-normal-stroma pairs; title pair number used.
- GSE29044: explicit `sample id` metadata; repeated IDs link tumor and adjacent disease-free tissues.
- GSE71053: explicit `patient number`; 3 patients / 18 biopsies.

Other discovery cohorts were conservatively treated as one GSM = one group because no shared
subject identifier was established from the provided metadata. This is not proof that every
remaining sample is biologically independent.

## Main audit findings
- 760 GSM accessions are unique.
- No exact normalized sample-title duplicate occurs across different GEO accessions.
- Conservative known-group mapping yields 595 biological-source groups across 760 samples.
- 97 confirmed groups contain >1 array/tissue; 50 confirmed groups contain both Cancer and Normal labels.
- 37 confirmed groups span the current 608-development / 152-held-out split.
- 40/152 held-out samples (26.3%) have a confirmed group mate in development.
- Within the 608 development samples, 82 groups contain >1 sample and
  71 of them are split across >1 current primary CV fold.
- In the current primary CV, 180/608 validation-sample events (29.6%) have a confirmed group mate
  in the corresponding training portion.
- Across the existing 10 repeated five-fold partitions, a mean of 181.3/608 samples per repeat
  (29.8%) are linked across folds by the conservative known-group mapping.

## Interpretation
The current sample-level CV and post-harmonization internal held-out assessment are not
subject/group-isolated for the confirmed repeated-sample cohorts. This does not invalidate
the independent GSE15852, GSE70947, TCGA-BRCA, or leave-one-cohort-out analyses, but it
creates a credible reviewer concern for the primary/repeated sample-level CV and the internal
held-out AUC.

The supplied proposed group-aware fold files retain the original 608 development samples and
do not alter the frozen manuscript results. They are intended only as candidate assignments for
a sensitivity analysis. Every known Subject_Group is confined to exactly one validation fold
within each repeat.

## Public evidence used to validate grouping rules
- GSE10780 source article: https://pmc.ncbi.nlm.nih.gov/articles/PMC2796276/
- GSE111662 GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE111662
- GSE26910 GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE26910
- GSE29044 GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE29044
- GSE71053 GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE71053
- GSE61304 GEO (used to avoid false grouping by repeated numeric title suffixes):
  https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE61304
