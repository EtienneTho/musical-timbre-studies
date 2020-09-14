### Copyright (c) Baptiste Caramiaux, Etienne Thoret
### All rights reserved
###
### this script computes the euclidean distances between the representations in sigma_path and then computes the correlation with perceptual dissimilarties
### change sigmasPath (line 15) with values on line 16 ou 17 to do it with decimated or no decimated version
### Full statistics and medians are computed
### 

import os
import numpy as np  
import pickle
import math
import matplotlib.pyplot as plt 
from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr, iqr

sigmasPath = './out_aud_STRF_euclidean_distance_avg_time_all11_WithFullStats/'
# out_aud_STRF_euclidean_distance_avg_time_all11_WithFullStats
# out_aud_STRF_euclidean_distance_avg_time_WithFullStats
sigmasTab = []

# load sigmas
tabCorr = []
ind = 1;
for r, d, f in os.walk(sigmasPath):
 for file in sorted(f):
  if 'resultsOptims' in file:
   sigmas = pickle.load(open(os.path.join(sigmasPath, file), 'rb'))
   # print(file+' '+str(sigmas['correlations']**2))
   # print(file+' r^2='+str(sigmas['correlations']**2)+' p='+str(sigmas['p'])+' lower='+str(sigmas['lower'])+' upper='+str(sigmas['upper'])+' power='+str(sigmas['power'])+' df='+str(sigmas['df']))
   print(str(sigmas['df'])+' '+"%.2f" %(sigmas['correlations']**2)+' '+"%.3f" %sigmas['p']+' ['+"%.3f" %(sigmas['lower'])+';'+"%.3f" %(sigmas['upper'])+'] '+"%.3f" %(sigmas['power']))

   tabCorr.append(sigmas['correlations']**2)
   ind+=1

print()
print(tabCorr)
print('Median explained variance : '  + str(np.median(tabCorr)) + ' (+/-) '+ str(iqr(tabCorr)))
print()

