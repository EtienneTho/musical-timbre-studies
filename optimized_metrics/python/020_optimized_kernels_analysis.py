### Thoret, Caramiaux (c) 2020
###
### This script analyzes the optimized metrics obtained with 01X_XXX scripts
### Both full statitistics and averaged values are provided
###
### This script replicates the results of the section "Optimized metrics simulating human dissimilarity ratings" for the Full STMF, scale-rate, 
### freq-rate, freq-scale representations and their decimated version
###

import os
import numpy as np  
import pickle
import math
import matplotlib.pyplot as plt 
from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr, iqr
import string
import pingouin as pg


# avgXXX : XXX correspond to the averaged dimension, if XXX is not time, time has been also averaged
sigmasPath = ['./out_aud_STRF_avgTime/', './out_aud_STRF_avg_time_decimate_allAt11/', 
              './out_aud_STRF_avgFreq/', './out_aud_STRF_avg_time_avg_freq_decimate_allAt11/',
              './out_aud_STRF_avgScale/', './out_aud_STRF_avg_time_avg_scale_decimate_allAt11/',
              './out_aud_STRF_avgRate/', './out_aud_STRF_avg_time_avg_rate_decimate_allAt11/']
# sigmasPath = {'out_aud_STRF_avg_time_decimate_allAt11', 'out_aud_STRF_avg_time_avg_scale_decimate_allAt11', 'out_aud_STRF_avg_time_avg_rate_decimate_allAt11',
#         'out_aud_STRF_avg_time_avg_freq_decimate_allAt11'}

sigmasTab = []

def corr(input_data,sigmas,dissimilarityMatrix):
    ndims, ninstrus = input_data.shape[0], input_data.shape[1]    
    idx_triu = np.triu_indices(dissimilarityMatrix.shape[0], k=1)
    target_v = dissimilarityMatrix[idx_triu]    
    mean_target = np.mean(target_v)
    std_target = np.std(target_v)    
    no_samples = ninstrus * (ninstrus - 1) / 2    
    kernel = np.zeros((ninstrus, ninstrus))
    idx = [i for i in range(len(input_data))]
    sigmas = np.clip(sigmas, a_min=1.0, a_max=1e15)
    for i in range(ninstrus):
            for j in range(i + 1, ninstrus):
                kernel[i, j] = -np.sum(
                    np.power(
                        np.divide(input_data[idx, i] - input_data[idx, j],
                                  (sigmas[idx] + np.finfo(float).eps)), 2))
    kernel_v = kernel[idx_triu]
    mean_kernel = np.mean(kernel_v)
    std_kernel = np.std(kernel_v)
    Jn = np.sum(np.multiply(kernel_v - mean_kernel, target_v - mean_target))
    Jd = no_samples * std_target * std_kernel
    # corr_, p___ = pearsonr(np.asarray(target_v),np.asarray(kernel_v))
    pg_ = pg.corr(np.asarray(target_v), np.asarray(kernel_v))
    aa = pg_['CI95%'][0][0]**2
    bb = pg_['CI95%'][0][1]**2
    lower = np.min([aa,bb])
    upper = np.max([aa,bb])
    p = pg_['p-val'][0]
    r = pg_['r'][0]
    power = pg_['power'][0]
    df = pg_['n'][0]-2
    return r, p, lower, upper, power, df


# load sigmas
# FULL REPRESENTATIONS
tabCorr = []
tabCorrOrganised = []
ind = 1;
print(sigmasPath)
for ii,path in enumerate(sigmasPath):
    print(path)
    tabCorr = []
    for r, d, f in os.walk(path):
     for file in sorted(f):
      if 'resultsOptims' in file: # 
       sigmas = pickle.load(open(os.path.join(path, file), 'rb'))
       sigmasTab.append(sigmas['sigmas'].flatten())
       r, p, lower, upper, power, df = corr(sigmas['representations'],sigmas['sigmas'],sigmas['dissimilarities'])

       corr_value_explained_variance= math.pow(corr(sigmas['representations'],sigmas['sigmas'],sigmas['dissimilarities'])[0],2)
       # print(file+' r^2='+str(corr_value_explained_variance)+' p='+str(p)+' lower='+str(lower)+' upper='+str(upper)+' power='+str(power)+' df='+str(df))
       print(str(df)+' '+"%.2f" %(corr_value_explained_variance)+' '+"%.3f" %p+' ['+"%.3f" %(lower)+';'+"%.3f" %(upper)+'] '+"%.3f" %(power))


       tabCorr.append(corr_value_explained_variance)       
       ind+=1
    tabCorrOrganised.append(tabCorr)

    print()
    print(tabCorr)
    # print('Averaged explained variance :' + str(np.mean(tabCorr)) + ' (+/-) '  + str(np.std(tabCorr)))
    print('Median explained variance : '  + str(np.median(tabCorr)) + ' (+/-) '+ str(iqr(tabCorr)))
    print()
print(np.asarray(tabCorrOrganised).shape)
print(np.transpose(np.asarray(tabCorrOrganised)))
print()
print(np.median(np.transpose(np.asarray(tabCorrOrganised)), axis=0))
print(iqr(np.transpose(np.asarray(tabCorrOrganised)), axis=0))


