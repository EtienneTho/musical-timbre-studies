# musical-timbre-studies

Supporting scripts for the paper: "Learning metrics on spectrotemporal modulations reveals the perception of musical instrument timbre" by Thoret E., Caramiaux B., Depalle P., McAdams S.

This project is separated into two main parts corresponding to the two main different analyses of the paper:
1. a mds based analysis of the 17 datasets in folder `'./mds_based_analysis/'` written in Matlab
2. a part concerning the optimisation of gaussian kernels in `'./optimized_metrics/'` written in Python

## Instructions
1. Paste the `'./ext/'` folder (private link provided upon request) in the main repo `'./musical-timbre-studies/'` (the results can be reproduced without this).
2. Download the `metricPython.zip` at the link https://osf.io/qu7vc/
3. Unzip `metricPython.zip` and then drag and drop the folders in the folder `'./python/'`

### Multidimensional scaling analysis
`'./mds_based_analysis/matlab/'` 

* Paste the `'./ext/'` folder (private link provided upon request) in the main repo `'./musical-timbre-studies/'`
* Run analyses with `main_MDS_BASED_ANALYSIS.m`
* Generate figure by running `Figure2.m`

Metric learning: optimisation of gaussian kernels written in python (python 3.X). We detail below the correspondance between the scripts and the results reported in the paper. As the computation of the optimizations (scripts 010_XX 011_XX 012_XX 013_XX) is time consuming, the optimized metrics are provided in the other repository subfolders and do not need to be necessarily re-run.

### Optimized metrics simulating human dissimilarity ratings
`'./optimized_metrics/python/'`:
   * `'010_optimize_metrics.py'`: run optimisation of between-sounds metrics for Full STMF, scale-rate, freq-rate, and freq-scale representations.
   * `'011_optimize_metrics_spectrum.py'`: run optimisation of between-sounds metrics for the Auditory Spectrum.
   * `'012_optimize_euclidean_distance.py'`: run the pairwise euclidean distances between Full STMFs representation.
   * `'013_optimize_decimated_metrics.py'`: run optimisation of between-sounds metrics for Full STMF, scale-rate, freq-rate, and freq-scale decimated representations.
   * `'014_acoustic_interpretation_results_light.py'`: Cross-validation analysis of the metrics
   * `'015_acoustic_interpretation_results_light_tabs.py'`: Cross-validation analysis of the metrics with full statistics
   * `'020_optimized_kernels_analysis.py'`: analysis of the optimized metrics for Full STMF, scale-rate, freq-rate, and freq-scale representations and their decimated ones.   
   * `'021_optimized_kernels_analysis_spectrum.py'`: analysis of the optimized metrics for the Auditory Spectrum.
   * `'022_EuclideanDistanceAnalyses.py'`: analysis of the euclidean distances pairwise correlations.

### Clustering of optimized metrics
`'./optimized_metrics/python/'`:
   * `'030_acoustic_interpretation_dendrograms.py'`: run the clustering analysis.
### Acoustic interpretations of the metrics
`'./optimized_metrics/python/'`:
   * `'040_acoustic_interpretation_tabs.py'`: this script computes the pairwise correlation between representations for the generalizability analysis with full statistics.
   * `'041_acoustic_interpretation.py'`: this script computes the pairwise correlation between representations for the generalizability analysis.
### Timbre perceptual metrics are experiment-specific
`'./optimized_metrics/python/'`:
   * `'050_correlation_with_variability.py'`: this script computes the correlation between the stimuli variability and the optimized metrics.
   * `'051_regression_between_metrics_and_stimuli.py'`: this script computes the multiple linear regression between the optimized metrics and the stimuli representation.
   * `'052_regression_between_metrics_and_stimuli_fullStats.py'`: this script computes the multiple linear regression between the optimized metrics and the stimuli representation.   

## Contact
Please feel free to contact me (etienne thoret) for any question(s), bug(s), request(s), suggestion(s): [firstname][name][AT]gmail[dot]com

## Depedencies

Python dependencies: `tensorly`, `numpy`, `matplotlib`, `aifc`, `tensorflow`, `scipy`, `random`, `pingouin`, `math`, `pylab`, `tslearn`, `sklearn`, `statsmodels`

