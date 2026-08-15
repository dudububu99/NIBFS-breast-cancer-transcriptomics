# Paper-to-code map

| Manuscript component | Paper-facing implementation | Compact verification data |
|---|---|---|
| Discovery preprocessing and QC | `notebooks/01_main_NIBFS_core.ipynb`, `src/preprocessing.py`, `src/quality_control.py`, `src/workflow.py` | `supplementary_data/` |
| NIBFS / DEG-only / mRMR / LASSO | `src/feature_selection.py`, `src/modeling.py` | `supplementary_data/`, fold-fitted verification tables |
| Primary 5-fold evaluation | `notebooks/01_main_NIBFS_core.ipynb` | `supplementary_data/Supplementary_Data_File_D1_Fold_Assignments_608.csv` |
| Repeated 10×5 CV | `src/repeated_10x5_k20_lr_V2.py` | manuscript-facing supplementary summaries |
| Training-fold-fitted preprocessing | `notebooks/02_fold_fitted_all_comparators_1x5.ipynb` | `results/verification/fold_fitted/` |
| Transfer-safe LOCO | `src/full_transfer_safe_loco_GENELEVEL_V2.py` | `supplementary_data/Table_S4_LOCO_Eligibility.csv` and manuscript tables |
| RWR-DEG baseline | `src/rwr_deg_network_baseline_10x5_V1.py` | `supplementary_data/Table_S2A_RWR_Comparison.csv` |
| Gene-label/topology permutation | `src/permuted_topology_control_100x_V1.py`, `src/permuted_topology_control_extend_100_to_1000_V3.py` | `supplementary_data/Table_S2B_Topology_Permutation.csv` |
| Degree-preserving rewiring | `src/degree_preserving_null.py`, `scripts/run_degree_preserving_null.py` | `results/verification/degree_preserving/`, `supplementary_data/Table_S2C_Degree_Preserving_Rewiring.csv` |
| GSE15852 | core notebook and `src/workflow.py` | `supplementary_data/Table_S5A_External_Summary.csv` |
| GSE70947 | `src/external_validation_GSE70947_V5.py` | `supplementary_data/Table_S5A_External_Summary.csv`, `Table_S5B_Gene_Direction_Concordance.csv` |
| TCGA-BRCA RNA-seq | `notebooks/03_TCGA_BRCA_RNAseq_external_validation.ipynb`, `src/tcga_brca_rnaseq_external_validation.py` | `results/verification/tcga_brca/`, `Table_S5A`, `Table_S5B` |
| Rank-weight sensitivity | `src/sensitivity.py` | `supplementary_data/Table_S3B_Rank_Weight_Sensitivity.csv` |
| Enrichment | `src/enrichment.py`, `src/biological_figures.py` | `Supplementary_Data_File_D2_Enrichment_All_Terms.csv`, `Table_S6_Selected_Enrichment.csv` |
| Gene-wise RFS context | `src/kmplotter.py`, `manual_inputs/KMPlotter_RFS_final_k20.csv` | `supplementary_data/Table_S7_Gene_Wise_RFS.csv` |
