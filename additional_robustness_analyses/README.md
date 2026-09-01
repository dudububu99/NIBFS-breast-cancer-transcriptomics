# Additional robustness analyses

This directory contains the additional robustness and validation analyses reported in the manuscript: pair-aware uncertainty for GSE15852, repeated strict training-fold-fitted evaluation, the fixed-k stability-selection comparator, sample-identity checks, and publication-figure source data.

Run `00_RUN_ALL_ADDITIONAL_ROBUSTNESS_ANALYSES.ipynb` to reproduce the tabular analyses from the supplied repository/results inputs. The plotting notebook under `plotting_only/` regenerates the associated publication figures from the result tables.

The fixed-k stability-selection analysis is intended for matched panel-overlap comparison and is not presented as the classical threshold-based error-control formulation. Within each outer-training fold it applies a 1,000-gene screen based on absolute Welch-style standardized mean differences, followed by 50 stratified half-sample L1-logistic resamples; the `pi >= 0.90` set is descriptive and does not define the reported top-20 panel. Numerical result files in `results_additional/` are the analysis outputs used for the manuscript.
