# musical-timbre-studies

Supporting scripts for the submitted paper: "Learning metrics on spectrotemporal modulations reveals the perception of musical instrument timbre" by Thoret E., Caramiaux B., Depalle P., McAdams S.

This project is separated into two parts corresponding to the two main different parts of the paper:
1. a mds based analysis of the 17 datasets in folder `'./mds_based_analysis/'` 
2. a part concerning the optimisation of gaussian kernels in `'./optimized_metrics/'`

## Instructions

1. Paste the `'./ext/'` folder (private link provided upon request) in the main repo `'./musical-timbre-studies/'`

2. MDS based analysis (MATLAB)
    
   * Run analyses with `main_MDS_BASED_ANALYSIS.m`
   * Generate figure by running `Figure2.m`
    
2. Metric learning: optimisation of gaussian kernels written in python (python 3.X). We are providing 6 different scripts corresponding to the different analysis of the paper.

   * `'01_optimize_metrics.py'`: run optimisation of between-sounds metrics (optimised metrics are logged in folder `'./out_folder'`)
   * `'02_optimize_decimated_metrics.py'`: run optimisation of between-sounds metrics with decimated representation.
   * `'03_optimized_kernels_analysis.py'`: run analysis of the optimised metrics
   * `'04_EuclideanDistance.py'`: compute correlations between perceptual dissimilarities and euclidean distances between STRFs
   * `'05_acoustic_interpretation.py'`: generalizability analysis
   * `'06_correlation_with_variability.py'`: correlations between optimised metrics and the variability of sound representations

## Depedencies

Python dependencies: `tensorly`, `numpy`, `matplotlib`, `aifc`, `tensorflow`

