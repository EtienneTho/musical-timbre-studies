### Thoret, Caramiaux (c) 2020
###
### This script computes the pairwise correlation between representations for the generalizability analysis
###
### This script replicates the results of the section "Acoustic interpretations of the metrics"
###

import os
import numpy as np  
import pickle
import math
import matplotlib.pyplot as plt 
from scipy.stats.stats import spearmanr
from scipy.stats import spearmanr, iqr
import pingouin as pg

sigmasTab = []

def strfvec2strftensor(strfvec,nbChannels=128,nbRates=22,nbScales=11):
    strftensor = np.reshape(strfvec,(nbChannels,nbScales,nbRates))


def strf2avgvec(strf,timeDim=False):
    if timeDim:
        strf_scale_rate = np.mean(np.abs(strf), axis=(0, 1))
        strf_freq_rate  = np.mean(np.abs(strf), axis=(0, 2))
        strf_freq_scale = np.mean(np.abs(strf), axis=(0, 3))
    else:
        strf_scale_rate = np.mean(np.abs(strf), axis=0)
        strf_freq_rate  = np.mean(np.abs(strf), axis=1)
        strf_freq_scale = np.mean(np.abs(strf), axis=2)        
    avgvec = np.concatenate((np.ravel(strf_scale_rate), np.ravel(strf_freq_rate), np.ravel(strf_freq_scale)))
    return avgvec

def avgvec2strfavg(avgvec,nbChannels=128,nbRates=22,nbScales=11):
    idxSR = nbRates*nbScales
    idxFR = nbChannels*nbRates
    idxFS = nbChannels*nbScales
    strf_scale_rate = np.reshape(avgvec[:idxSR],(nbScales,nbRates))
    strf_freq_rate = np.reshape(avgvec[idxSR:idxSR+idxFR],(nbChannels,nbRates))
    strf_freq_scale = np.reshape(avgvec[idxSR+idxFR:],(nbChannels,nbScales))
    return strf_scale_rate, strf_freq_rate, strf_freq_scale

def plotStrfavg(strf_scale_rate, strf_freq_rate, strf_freq_scale,aspect_='auto', interpolation_='none',figname='defaut',show='true'):
    plt.suptitle(figname, fontsize=10)
    plt.subplot(1,3,1)
    plt.imshow(strf_scale_rate, aspect=aspect_, interpolation=interpolation_,origin='lower')
    plt.xlabel('Rates (Hz)', fontsize=10)
    plt.ylabel('Scales (c/o)', fontsize=10)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,3,2)
    plt.imshow(strf_freq_rate, aspect=aspect_, interpolation=interpolation_,origin='lower')
    plt.xticks([])
    plt.yticks([]) 
    plt.xlabel('Rates (Hz)', fontsize=10)
    plt.ylabel('Freq Channel', fontsize=10)    
    plt.subplot(1,3,3)
    plt.imshow(np.transpose(strf_freq_scale), aspect=aspect_, interpolation=interpolation_,origin='lower')
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('Scales (c/o)', fontsize=10)
    plt.xlabel('Freq Channel', fontsize=10)
    plt.savefig(figname+'.png')
    if show=='true':
        plt.show()
def plotStrfavgN(strf_scale_rate, strf_freq_rate, strf_freq_scale,aspect_='auto', interpolation_='none',figname='defaut',show='true', NPlots=1, indPlot=1):
    plt.suptitle(figname, fontsize=10)
    plt.subplot(NPlots,3,1+(indPlot-1)*3)
    plt.imshow(strf_scale_rate, aspect=aspect_, interpolation=interpolation_,origin='lower')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(NPlots,3,2+(indPlot-1)*3)
    plt.imshow(strf_freq_rate, aspect=aspect_, interpolation=interpolation_,origin='lower')
    plt.xticks([])
    plt.yticks([])   
    plt.subplot(NPlots,3,3+(indPlot-1)*3)
    plt.imshow(strf_freq_scale, aspect=aspect_, interpolation=interpolation_,origin='lower')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(figname+'.png')
    if show=='true':
        plt.show()        

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
    return Jn/Jd

###############################
##### All Datasets (128x22x11)
sigmasPath = './out_aud_STRF_avgTime/'
tabCorr = []
ind = 1;
for r, d, f in os.walk(sigmasPath):
 for file in sorted(f):
  if 'resultsOptims' in file: # 
   sigmas = pickle.load(open(os.path.join(sigmasPath, file), 'rb'))
   sigmasTab.append(sigmas['sigmas'].flatten())
   corr_value_explained_variance = math.pow(corr(sigmas['representations'],sigmas['sigmas'],sigmas['dissimilarities']),2)
   tabCorr.append(corr_value_explained_variance)
   ind+=1
plt.show()


# compute pairwise correlations => generalizability
nbDatasets = sigmasTab.__len__()
pairwisePearsonMatrixFullTensor = np.zeros((nbDatasets,nbDatasets))
pairwisePearsonMatrixScaleRate  = np.zeros((nbDatasets,nbDatasets))
pairwisePearsonMatrixFreqScale  = np.zeros((nbDatasets,nbDatasets))
pairwisePearsonMatrixFreqRate   = np.zeros((nbDatasets,nbDatasets))

print()
print('##################################')
print('All Datasets (128x22x11)')
print('Mutual Median pairwise correlations')
print('Full STRF')
p_tab = []
power_tab = []
CI95range_tab = []
for iDataset in range(nbDatasets):
    for jDataset in range(iDataset+1,nbDatasets):
        strfTensorI = np.reshape(sigmasTab[iDataset],(128,11,22))
        strf_scale_rateI, strf_freq_rateI, strf_freq_scaleI = avgvec2strfavg(strf2avgvec(strfTensorI))
        strfTensorJ = np.reshape(sigmasTab[jDataset],(128,11,22))
        strf_scale_rateJ, strf_freq_rateJ, strf_freq_scaleJ = avgvec2strfavg(strf2avgvec(strfTensorJ))
        # pairwisePearsonMatrixFullTensor[iDataset][jDataset]  = spearmanr(strfTensorI.flatten(),strfTensorJ.flatten())[0]**2     
        pg_ = pg.corr(strfTensorI.flatten(), strfTensorJ.flatten(), method="spearman")
        aa = pg_['CI95%'][0][0]**2
        bb = pg_['CI95%'][0][1]**2
        lower = np.min([aa,bb])
        upper = np.max([aa,bb])
        p = pg_['p-val'][0]
        r = pg_['r'][0]
        power = pg_['power'][0]
        df = pg_['n'][0]-2
        # print(str(iDataset)+' '+str(jDataset)+' '+str(df)+' '+"%.2f" %(r**2)+' '+"%.3f" %p+' ['+"%.3f" %(lower)+';'+"%.3f" %(upper)+'] '+"%.3f" %(power)+' '+"%.3f" %(upper-lower))
        p_tab.append(p)
        power_tab.append(power)
        CI95range_tab.append(upper-lower)
print('Full Statistics Ranges')
print('p_median='+"%.2f" %(np.median(p_tab))+' p_min='+"%.2f" %(np.min(p_tab)) + ' p_max='+"%.2f" %(np.max(p_tab)))
print('power_median='+"%.2f" %(np.median(power_tab))+' power_min='+"%.2f" %(np.min(power_tab)) + ' power_max='+"%.2f" %(np.max(power_tab)))
print('CI95range_tab_median='+"%.2f" %(np.median(p_tab))+' CI95range_tab_min='+"%.2f" %(np.min(CI95range_tab)) + ' CI95range_tab_max='+"%.2f" %(np.max(CI95range_tab)))
print()


print('Scale/Rate')
p_tab = []
power_tab = []
CI95range_tab = []
for iDataset in range(nbDatasets):
    for jDataset in range(iDataset+1,nbDatasets):
        strfTensorI = np.reshape(sigmasTab[iDataset],(128,11,22))
        strf_scale_rateI, strf_freq_rateI, strf_freq_scaleI = avgvec2strfavg(strf2avgvec(strfTensorI))
        strfTensorJ = np.reshape(sigmasTab[jDataset],(128,11,22))
        strf_scale_rateJ, strf_freq_rateJ, strf_freq_scaleJ = avgvec2strfavg(strf2avgvec(strfTensorJ))
        pg_ = pg.corr(strf_scale_rateI.flatten(), strf_scale_rateJ.flatten(), method="spearman")
        aa = pg_['CI95%'][0][0]**2
        bb = pg_['CI95%'][0][1]**2
        lower = np.min([aa,bb])
        upper = np.max([aa,bb])
        p = pg_['p-val'][0]
        r = pg_['r'][0]
        power = pg_['power'][0]
        df = pg_['n'][0]-2
        # print(str(iDataset)+' '+str(jDataset)+' '+str(df)+' '+"%.2f" %(r**2)+' '+"%.3f" %p+' ['+"%.3f" %(lower)+';'+"%.3f" %(upper)+'] '+"%.3f" %(power)+' '+"%.3f" %(upper-lower))
        p_tab.append(p)
        power_tab.append(power)
        CI95range_tab.append(upper-lower)
print('Full Statistics Ranges')
print('p_median='+"%.2f" %(np.median(p_tab))+' p_min='+"%.2f" %(np.min(p_tab)) + ' p_max='+"%.2f" %(np.max(p_tab)))
print('power_median='+"%.2f" %(np.median(power_tab))+' power_min='+"%.2f" %(np.min(power_tab)) + ' power_max='+"%.2f" %(np.max(power_tab)))
print('CI95range_tab_median='+"%.2f" %(np.median(p_tab))+' CI95range_tab_min='+"%.2f" %(np.min(CI95range_tab)) + ' CI95range_tab_max='+"%.2f" %(np.max(CI95range_tab)))
print()

print('Freq/Rate')
p_tab = []
power_tab = []
CI95range_tab = []
for iDataset in range(nbDatasets):
    for jDataset in range(iDataset+1,nbDatasets):
        strfTensorI = np.reshape(sigmasTab[iDataset],(128,11,22))
        strf_scale_rateI, strf_freq_rateI, strf_freq_scaleI = avgvec2strfavg(strf2avgvec(strfTensorI))
        strfTensorJ = np.reshape(sigmasTab[jDataset],(128,11,22))
        strf_scale_rateJ, strf_freq_rateJ, strf_freq_scaleJ = avgvec2strfavg(strf2avgvec(strfTensorJ))
        pairwisePearsonMatrixFreqRate[iDataset][jDataset]    = spearmanr(strf_freq_rateI.flatten(),strf_freq_rateJ.flatten())[0]**2
        pg_ = pg.corr(strf_freq_rateI.flatten(), strf_freq_rateJ.flatten(), method="spearman")
        aa = pg_['CI95%'][0][0]**2
        bb = pg_['CI95%'][0][1]**2
        lower = np.min([aa,bb])
        upper = np.max([aa,bb])
        p = pg_['p-val'][0]
        r = pg_['r'][0]
        power = pg_['power'][0]
        df = pg_['n'][0]-2
        # print(str(iDataset)+' '+str(jDataset)+' '+str(df)+' '+"%.2f" %(r**2)+' '+"%.3f" %p+' ['+"%.3f" %(lower)+';'+"%.3f" %(upper)+'] '+"%.3f" %(power)+' '+"%.3f" %(upper-lower))        
        p_tab.append(p)
        power_tab.append(power)
        CI95range_tab.append(upper-lower)
print('Full Statistics Ranges')
print('p_median='+"%.2f" %(np.median(p_tab))+' p_min='+"%.2f" %(np.min(p_tab)) + ' p_max='+"%.2f" %(np.max(p_tab)))
print('power_median='+"%.2f" %(np.median(power_tab))+' power_min='+"%.2f" %(np.min(power_tab)) + ' power_max='+"%.2f" %(np.max(power_tab)))
print('CI95range_tab_median='+"%.2f" %(np.median(p_tab))+' CI95range_tab_min='+"%.2f" %(np.min(CI95range_tab)) + ' CI95range_tab_max='+"%.2f" %(np.max(CI95range_tab)))
print()

print('Freq/Scale')
p_tab = []
power_tab = []
CI95range_tab = []
for iDataset in range(nbDatasets):
    for jDataset in range(iDataset+1,nbDatasets):
        strfTensorI = np.reshape(sigmasTab[iDataset],(128,11,22))
        strf_scale_rateI, strf_freq_rateI, strf_freq_scaleI = avgvec2strfavg(strf2avgvec(strfTensorI))
        strfTensorJ = np.reshape(sigmasTab[jDataset],(128,11,22))
        strf_scale_rateJ, strf_freq_rateJ, strf_freq_scaleJ = avgvec2strfavg(strf2avgvec(strfTensorJ))
        # pairwisePearsonMatrixFreqScale[iDataset][jDataset]   = spearmanr(strf_freq_scaleI.flatten(),strf_freq_scaleJ.flatten())[0]**2                
        pg_ = pg.corr(strf_freq_scaleI.flatten(), strf_freq_scaleJ.flatten(), method="spearman")
        aa = pg_['CI95%'][0][0]**2
        bb = pg_['CI95%'][0][1]**2
        lower = np.min([aa,bb])
        upper = np.max([aa,bb])
        p = pg_['p-val'][0]
        r = pg_['r'][0]
        power = pg_['power'][0]
        df = pg_['n'][0]-2
        # print(str(iDataset)+' '+str(jDataset)+' '+str(df)+' '+"%.2f" %(r**2)+' '+"%.3f" %p+' ['+"%.3f" %(lower)+';'+"%.3f" %(upper)+'] '+"%.3f" %(power)+' '+"%.3f" %(upper-lower))           
        p_tab.append(p)
        power_tab.append(power)
        CI95range_tab.append(upper-lower)
print('Full Statistics Ranges')
print('p_median='+"%.2f" %(np.median(p_tab))+' p_min='+"%.2f" %(np.min(p_tab)) + ' p_max='+"%.2f" %(np.max(p_tab)))
print('power_median='+"%.2f" %(np.median(power_tab))+' power_min='+"%.2f" %(np.min(power_tab)) + ' power_max='+"%.2f" %(np.max(power_tab)))
print('CI95range_tab_median='+"%.2f" %(np.median(p_tab))+' CI95range_tab_min='+"%.2f" %(np.min(CI95range_tab)) + ' CI95range_tab_max='+"%.2f" %(np.max(CI95range_tab)))
print()

print('Freq/Rate - Mdn='+str(np.median(pairwisePearsonMatrixFreqRate[np.triu_indices(nbDatasets,1)]))   + ' iqr='+str(iqr(pairwisePearsonMatrixFreqRate[np.triu_indices(nbDatasets,1)])))
print('Freq/Scale - Mdn='+str(np.median(pairwisePearsonMatrixFreqScale[np.triu_indices(nbDatasets,1)]))  + ' iqr='+str(iqr(pairwisePearsonMatrixFreqScale[np.triu_indices(nbDatasets,1)])))

print()
print('Mutual Mean pairwise correlations')
print('Full STRF - M='+str(np.mean(pairwisePearsonMatrixFullTensor[np.triu_indices(nbDatasets,1)])) + ' SD='+str(np.std(pairwisePearsonMatrixFullTensor[np.triu_indices(nbDatasets,1)])))
print('Scale/Rate - M='+str(np.mean(pairwisePearsonMatrixScaleRate[np.triu_indices(nbDatasets,1)]))  + ' SD='+str(np.std(pairwisePearsonMatrixScaleRate[np.triu_indices(nbDatasets,1)])))
print('Freq/Rate - M='+str(np.mean(pairwisePearsonMatrixFreqRate[np.triu_indices(nbDatasets,1)]))   + ' SD='+str(np.std(pairwisePearsonMatrixFreqRate[np.triu_indices(nbDatasets,1)])))
print('Freq/Scale - M='+str(np.mean(pairwisePearsonMatrixFreqScale[np.triu_indices(nbDatasets,1)]))  + ' SD='+str(np.std(pairwisePearsonMatrixFreqScale[np.triu_indices(nbDatasets,1)])))

# ###############################
# ##### All Datasets (128x22x11) Eb=311Hz
# sigmasPath = './out_aud_STRF_avgTime/'
# tabCorr = []
# ind = 1;
# for r, d, f in os.walk(sigmasPath):
#  for file in sorted(f):
#   if 'resultsOptims' and '311' in file: # 
#    sigmas = pickle.load(open(os.path.join(sigmasPath, file), 'rb'))
#    sigmasTab.append(sigmas['sigmas'].flatten())
#    corr_value_explained_variance = math.pow(corr(sigmas['representations'],sigmas['sigmas'],sigmas['dissimilarities']),2)
#    tabCorr.append(corr_value_explained_variance)
#    ind+=1
# plt.show()


# # compute pairwise correlations => generalizability
# nbDatasets = sigmasTab.__len__()
# pairwisePearsonMatrixFullTensor = np.zeros((nbDatasets,nbDatasets))
# pairwisePearsonMatrixScaleRate  = np.zeros((nbDatasets,nbDatasets))
# pairwisePearsonMatrixFreqScale  = np.zeros((nbDatasets,nbDatasets))
# pairwisePearsonMatrixFreqRate   = np.zeros((nbDatasets,nbDatasets))

# for iDataset in range(nbDatasets):
#     for jDataset in range(nbDatasets):
#         strfTensorI = np.reshape(sigmasTab[iDataset],(128,11,22))
#         strf_scale_rateI, strf_freq_rateI, strf_freq_scaleI = avgvec2strfavg(strf2avgvec(strfTensorI))
#         strfTensorJ = np.reshape(sigmasTab[jDataset],(128,11,22))
#         strf_scale_rateJ, strf_freq_rateJ, strf_freq_scaleJ = avgvec2strfavg(strf2avgvec(strfTensorJ))

#         pairwisePearsonMatrixFullTensor[iDataset][jDataset]  = spearmanr(strfTensorI.flatten(),strfTensorJ.flatten())[0]**2
#         pairwisePearsonMatrixScaleRate[iDataset][jDataset]   = spearmanr(strf_scale_rateI.flatten(),strf_scale_rateJ.flatten())[0]**2
#         pairwisePearsonMatrixFreqRate[iDataset][jDataset]    = spearmanr(strf_freq_rateI.flatten(),strf_freq_rateJ.flatten())[0]**2
#         pairwisePearsonMatrixFreqScale[iDataset][jDataset]   = spearmanr(strf_freq_scaleI.flatten(),strf_freq_scaleJ.flatten())[0]**2

# print()
# print('##################################')
# print('All Datasets (128x22x11) - Eb=311Hz')
# print('Mutual Median pairwise correlations')
# print('Full STRF - Mdn='+str(np.median(pairwisePearsonMatrixFullTensor[np.triu_indices(nbDatasets,1)])) + ' iqr='+str(iqr(pairwisePearsonMatrixFullTensor[np.triu_indices(nbDatasets,1)])))
# print('Scale/Rate - Mdn='+str(np.median(pairwisePearsonMatrixScaleRate[np.triu_indices(nbDatasets,1)]))  + ' iqr='+str(iqr(pairwisePearsonMatrixScaleRate[np.triu_indices(nbDatasets,1)])))
# print('Freq/Rate - Mdn='+str(np.median(pairwisePearsonMatrixFreqRate[np.triu_indices(nbDatasets,1)]))   + ' iqr='+str(iqr(pairwisePearsonMatrixFreqRate[np.triu_indices(nbDatasets,1)])))
# print('Freq/Scale - Mdn='+str(np.median(pairwisePearsonMatrixFreqScale[np.triu_indices(nbDatasets,1)]))  + ' iqr='+str(iqr(pairwisePearsonMatrixFreqScale[np.triu_indices(nbDatasets,1)])))

# print()
# print('Mutual Mean pairwise correlations')
# print('Full STRF - M='+str(np.mean(pairwisePearsonMatrixFullTensor[np.triu_indices(nbDatasets,1)])) + ' SD='+str(np.std(pairwisePearsonMatrixFullTensor[np.triu_indices(nbDatasets,1)])))
# print('Scale/Rate - M='+str(np.mean(pairwisePearsonMatrixScaleRate[np.triu_indices(nbDatasets,1)]))  + ' SD='+str(np.std(pairwisePearsonMatrixScaleRate[np.triu_indices(nbDatasets,1)])))
# print('Freq/Rate - M='+str(np.mean(pairwisePearsonMatrixFreqRate[np.triu_indices(nbDatasets,1)]))   + ' SD='+str(np.std(pairwisePearsonMatrixFreqRate[np.triu_indices(nbDatasets,1)])))
# print('Freq/Scale - M='+str(np.mean(pairwisePearsonMatrixFreqScale[np.triu_indices(nbDatasets,1)]))  + ' SD='+str(np.std(pairwisePearsonMatrixFreqScale[np.triu_indices(nbDatasets,1)])))


###############################
##### All Datasets Decimated (10x10x10)

# compute pairwise correlations => generalizability
nbDatasets = sigmasTab.__len__()
pairwisePearsonMatrix = np.zeros((nbDatasets,nbDatasets))
pairwiseSpearmanMatrix = np.zeros((nbDatasets,nbDatasets))

print('##################################')
print('All Datasets (10x10x10)')
print('Mutual Median pairwise correlations')
print('Full STRF')
p_tab = []
power_tab = []
CI95range_tab = []
for iDataset in range(nbDatasets):
    for jDataset in range(iDataset+1,nbDatasets):
        strfTensorI = np.reshape(sigmasTab[iDataset],(128,11,22))
        freqVec  = [0, 14, 28, 42, 56, 71, 85, 99, 113, 127]
        rateVec  = [0, 3, 5, 7, 10, 11, 14, 16, 18, 21]
        scaleVec = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]                               
        # indexedTab = strf_avgTime       
        strf_scale_rateI = np.mean(strfTensorI[freqVec,:,:][:,scaleVec,:][:,:,rateVec] ,axis=0)
        strf_freq_rateI  = np.mean(strfTensorI[freqVec,:,:][:,scaleVec,:][:,:,rateVec] ,axis=1)
        strf_freq_scaleI = np.mean(strfTensorI[freqVec,:,:][:,scaleVec,:][:,:,rateVec] ,axis=2)
        # strf_scale_rateI, strf_freq_rateI, strf_freq_scaleI = avgvec2strfavg(strf2avgvec(strfTensorI))
        strfTensorJ = np.reshape(sigmasTab[jDataset],(128,11,22))
        strf_scale_rateJ = np.mean(strfTensorJ[freqVec,:,:][:,scaleVec,:][:,:,rateVec] ,axis=0)
        strf_freq_rateJ  = np.mean(strfTensorJ[freqVec,:,:][:,scaleVec,:][:,:,rateVec] ,axis=1)
        strf_freq_scaleJ = np.mean(strfTensorJ[freqVec,:,:][:,scaleVec,:][:,:,rateVec] ,axis=2)        
        # strf_scale_rateJ, strf_freq_rateJ, strf_freq_scaleJ = avgvec2strfavg(strf2avgvec(strfTensorJ))

        pairwisePearsonMatrixFullTensor[iDataset][jDataset]  = spearmanr(strfTensorI.flatten(),strfTensorJ.flatten())[0]**2
        pg_ = pg.corr(strfTensorI.flatten(), strfTensorJ.flatten(), method="spearman")
        aa = pg_['CI95%'][0][0]**2
        bb = pg_['CI95%'][0][1]**2
        lower = np.min([aa,bb])
        upper = np.max([aa,bb])
        p = pg_['p-val'][0]
        r = pg_['r'][0]
        power = pg_['power'][0]
        df = pg_['n'][0]-2
        # print(str(iDataset)+' '+str(jDataset)+' '+str(df)+' '+"%.2f" %(r**2)+' '+"%.3f" %p+' ['+"%.3f" %(lower)+';'+"%.3f" %(upper)+'] '+"%.3f" %(power)+' '+"%.3f" %(upper-lower))        
        p_tab.append(p)
        power_tab.append(power)
        CI95range_tab.append(upper-lower)
print('Full Statistics Ranges')
print('p_median='+"%.2f" %(np.median(p_tab))+' p_min='+"%.2f" %(np.min(p_tab)) + ' p_max='+"%.2f" %(np.max(p_tab)))
print('power_median='+"%.2f" %(np.median(power_tab))+' power_min='+"%.2f" %(np.min(power_tab)) + ' power_max='+"%.2f" %(np.max(power_tab)))
print('CI95range_tab_median='+"%.2f" %(np.median(p_tab))+' CI95range_tab_min='+"%.2f" %(np.min(CI95range_tab)) + ' CI95range_tab_max='+"%.2f" %(np.max(CI95range_tab)))
print()

print('Scale/Rate')
p_tab = []
power_tab = []
CI95range_tab = []
for iDataset in range(nbDatasets):
    for jDataset in range(iDataset+1,nbDatasets):
        strfTensorI = np.reshape(sigmasTab[iDataset],(128,11,22))
        freqVec  = [0, 14, 28, 42, 56, 71, 85, 99, 113, 127]
        rateVec  = [0, 3, 5, 7, 10, 11, 14, 16, 18, 21]
        scaleVec = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]                               
        # indexedTab = strf_avgTime       
        strf_scale_rateI = np.mean(strfTensorI[freqVec,:,:][:,scaleVec,:][:,:,rateVec] ,axis=0)
        strf_freq_rateI  = np.mean(strfTensorI[freqVec,:,:][:,scaleVec,:][:,:,rateVec] ,axis=1)
        strf_freq_scaleI = np.mean(strfTensorI[freqVec,:,:][:,scaleVec,:][:,:,rateVec] ,axis=2)
        # strf_scale_rateI, strf_freq_rateI, strf_freq_scaleI = avgvec2strfavg(strf2avgvec(strfTensorI))
        strfTensorJ = np.reshape(sigmasTab[jDataset],(128,11,22))
        strf_scale_rateJ = np.mean(strfTensorJ[freqVec,:,:][:,scaleVec,:][:,:,rateVec] ,axis=0)
        strf_freq_rateJ  = np.mean(strfTensorJ[freqVec,:,:][:,scaleVec,:][:,:,rateVec] ,axis=1)
        strf_freq_scaleJ = np.mean(strfTensorJ[freqVec,:,:][:,scaleVec,:][:,:,rateVec] ,axis=2)        
        # strf_scale_rateJ, strf_freq_rateJ, strf_freq_scaleJ = avgvec2strfavg(strf2avgvec(strfTensorJ))

        pairwisePearsonMatrixScaleRate[iDataset][jDataset]   = spearmanr(strf_scale_rateI.flatten(),strf_scale_rateJ.flatten())[0]**2
        pg_ = pg.corr(strf_scale_rateI.flatten(), strf_scale_rateJ.flatten(), method="spearman")
        aa = pg_['CI95%'][0][0]**2
        bb = pg_['CI95%'][0][1]**2
        lower = np.min([aa,bb])
        upper = np.max([aa,bb])
        p = pg_['p-val'][0]
        r = pg_['r'][0]
        power = pg_['power'][0]
        df = pg_['n'][0]-2
        # print(str(iDataset)+' '+str(jDataset)+' '+str(df)+' '+"%.2f" %(r**2)+' '+"%.3f" %p+' ['+"%.3f" %(lower)+';'+"%.3f" %(upper)+'] '+"%.3f" %(power)+' '+"%.3f" %(upper-lower))        
        p_tab.append(p)
        power_tab.append(power)
        CI95range_tab.append(upper-lower)
print('Full Statistics Ranges')
print('p_median='+"%.2f" %(np.median(p_tab))+' p_min='+"%.2f" %(np.min(p_tab)) + ' p_max='+"%.2f" %(np.max(p_tab)))
print('power_median='+"%.2f" %(np.median(power_tab))+' power_min='+"%.2f" %(np.min(power_tab)) + ' power_max='+"%.2f" %(np.max(power_tab)))
print('CI95range_tab_median='+"%.2f" %(np.median(p_tab))+' CI95range_tab_min='+"%.2f" %(np.min(CI95range_tab)) + ' CI95range_tab_max='+"%.2f" %(np.max(CI95range_tab)))
print()


print('Freq/Rate')
p_tab = []
power_tab = []
CI95range_tab = []
for iDataset in range(nbDatasets):
    for jDataset in range(iDataset+1,nbDatasets):
        strfTensorI = np.reshape(sigmasTab[iDataset],(128,11,22))
        freqVec  = [0, 14, 28, 42, 56, 71, 85, 99, 113, 127]
        rateVec  = [0, 3, 5, 7, 10, 11, 14, 16, 18, 21]
        scaleVec = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]                               
        # indexedTab = strf_avgTime       
        strf_scale_rateI = np.mean(strfTensorI[freqVec,:,:][:,scaleVec,:][:,:,rateVec] ,axis=0)
        strf_freq_rateI  = np.mean(strfTensorI[freqVec,:,:][:,scaleVec,:][:,:,rateVec] ,axis=1)
        strf_freq_scaleI = np.mean(strfTensorI[freqVec,:,:][:,scaleVec,:][:,:,rateVec] ,axis=2)
        # strf_scale_rateI, strf_freq_rateI, strf_freq_scaleI = avgvec2strfavg(strf2avgvec(strfTensorI))
        strfTensorJ = np.reshape(sigmasTab[jDataset],(128,11,22))
        strf_scale_rateJ = np.mean(strfTensorJ[freqVec,:,:][:,scaleVec,:][:,:,rateVec] ,axis=0)
        strf_freq_rateJ  = np.mean(strfTensorJ[freqVec,:,:][:,scaleVec,:][:,:,rateVec] ,axis=1)
        strf_freq_scaleJ = np.mean(strfTensorJ[freqVec,:,:][:,scaleVec,:][:,:,rateVec] ,axis=2)        
        # strf_scale_rateJ, strf_freq_rateJ, strf_freq_scaleJ = avgvec2strfavg(strf2avgvec(strfTensorJ))

        pairwisePearsonMatrixFreqRate[iDataset][jDataset]    = spearmanr(strf_freq_rateI.flatten(),strf_freq_rateJ.flatten())[0]**2
        pg_ = pg.corr(strf_freq_rateI.flatten(), strf_freq_rateJ.flatten(), method="spearman")
        aa = pg_['CI95%'][0][0]**2
        bb = pg_['CI95%'][0][1]**2
        lower = np.min([aa,bb])
        upper = np.max([aa,bb])
        p = pg_['p-val'][0]
        r = pg_['r'][0]
        power = pg_['power'][0]
        df = pg_['n'][0]-2
        # print(str(iDataset)+' '+str(jDataset)+' '+str(df)+' '+"%.2f" %(r**2)+' '+"%.3f" %p+' ['+"%.3f" %(lower)+';'+"%.3f" %(upper)+'] '+"%.3f" %(power)+' '+"%.3f" %(upper-lower))        
        p_tab.append(p)
        power_tab.append(power)
        CI95range_tab.append(upper-lower)
print('Full Statistics Ranges')
print('p_median='+"%.2f" %(np.median(p_tab))+' p_min='+"%.2f" %(np.min(p_tab)) + ' p_max='+"%.2f" %(np.max(p_tab)))
print('power_median='+"%.2f" %(np.median(power_tab))+' power_min='+"%.2f" %(np.min(power_tab)) + ' power_max='+"%.2f" %(np.max(power_tab)))
print('CI95range_tab_median='+"%.2f" %(np.median(p_tab))+' CI95range_tab_min='+"%.2f" %(np.min(CI95range_tab)) + ' CI95range_tab_max='+"%.2f" %(np.max(CI95range_tab)))
print()

print('Freq/Scale')
p_tab = []
power_tab = []
CI95range_tab = []
for iDataset in range(nbDatasets):
    for jDataset in range(iDataset+1,nbDatasets):
        strfTensorI = np.reshape(sigmasTab[iDataset],(128,11,22))
        freqVec  = [0, 14, 28, 42, 56, 71, 85, 99, 113, 127]
        rateVec  = [0, 3, 5, 7, 10, 11, 14, 16, 18, 21]
        scaleVec = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]                               
        # indexedTab = strf_avgTime       
        strf_scale_rateI = np.mean(strfTensorI[freqVec,:,:][:,scaleVec,:][:,:,rateVec] ,axis=0)
        strf_freq_rateI  = np.mean(strfTensorI[freqVec,:,:][:,scaleVec,:][:,:,rateVec] ,axis=1)
        strf_freq_scaleI = np.mean(strfTensorI[freqVec,:,:][:,scaleVec,:][:,:,rateVec] ,axis=2)
        # strf_scale_rateI, strf_freq_rateI, strf_freq_scaleI = avgvec2strfavg(strf2avgvec(strfTensorI))
        strfTensorJ = np.reshape(sigmasTab[jDataset],(128,11,22))
        strf_scale_rateJ = np.mean(strfTensorJ[freqVec,:,:][:,scaleVec,:][:,:,rateVec] ,axis=0)
        strf_freq_rateJ  = np.mean(strfTensorJ[freqVec,:,:][:,scaleVec,:][:,:,rateVec] ,axis=1)
        strf_freq_scaleJ = np.mean(strfTensorJ[freqVec,:,:][:,scaleVec,:][:,:,rateVec] ,axis=2)        
        # strf_scale_rateJ, strf_freq_rateJ, strf_freq_scaleJ = avgvec2strfavg(strf2avgvec(strfTensorJ))

        pairwisePearsonMatrixFreqScale[iDataset][jDataset]   = spearmanr(strf_freq_scaleI.flatten(),strf_freq_scaleJ.flatten())[0]**2
        pg_ = pg.corr(strf_freq_scaleI.flatten(), strf_freq_scaleJ.flatten(), method="spearman")
        aa = pg_['CI95%'][0][0]**2
        bb = pg_['CI95%'][0][1]**2
        lower = np.min([aa,bb])
        upper = np.max([aa,bb])
        p = pg_['p-val'][0]
        r = pg_['r'][0]
        power = pg_['power'][0]
        df = pg_['n'][0]-2
        # print(str(iDataset)+' '+str(jDataset)+' '+str(df)+' '+"%.2f" %(r**2)+' '+"%.3f" %p+' ['+"%.3f" %(lower)+';'+"%.3f" %(upper)+'] '+"%.3f" %(power)+' '+"%.3f" %(upper-lower))        
        p_tab.append(p)
        power_tab.append(power)
        CI95range_tab.append(upper-lower)
print('Full Statistics Ranges')
print('p_median='+"%.2f" %(np.median(p_tab))+' p_min='+"%.2f" %(np.min(p_tab)) + ' p_max='+"%.2f" %(np.max(p_tab)))
print('power_median='+"%.2f" %(np.median(power_tab))+' power_min='+"%.2f" %(np.min(power_tab)) + ' power_max='+"%.2f" %(np.max(power_tab)))
print('CI95range_tab_median='+"%.2f" %(np.median(p_tab))+' CI95range_tab_min='+"%.2f" %(np.min(CI95range_tab)) + ' CI95range_tab_max='+"%.2f" %(np.max(CI95range_tab)))


print()
print('##################################')
print('All Datasets (10x10x10)')
print('Mutual Median pairwise correlations')
print('Full STRF - Mdn='+str(np.median(pairwisePearsonMatrixFullTensor[np.triu_indices(nbDatasets,1)])) + ' iqr='+str(iqr(pairwisePearsonMatrixFullTensor[np.triu_indices(nbDatasets,1)])))
print('Scale/Rate - Mdn='+str(np.median(pairwisePearsonMatrixScaleRate[np.triu_indices(nbDatasets,1)]))  + ' iqr='+str(iqr(pairwisePearsonMatrixScaleRate[np.triu_indices(nbDatasets,1)])))
print('Freq/Rate - Mdn='+str(np.median(pairwisePearsonMatrixFreqRate[np.triu_indices(nbDatasets,1)]))   + ' iqr='+str(iqr(pairwisePearsonMatrixFreqRate[np.triu_indices(nbDatasets,1)])))
print('Freq/Scale - Mdn='+str(np.median(pairwisePearsonMatrixFreqScale[np.triu_indices(nbDatasets,1)]))  + ' iqr='+str(iqr(pairwisePearsonMatrixFreqScale[np.triu_indices(nbDatasets,1)])))



