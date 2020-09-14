### Thoret, Caramiaux (c) 2020
###
### This script replicates the results of the section "Clustering of optimized metrics"
###

import os
import numpy as np  
import pickle
import math
import matplotlib.pyplot as plt 
from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr, iqr
import scipy
import pylab
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from scipy.spatial.distance import euclidean
from tslearn.metrics import dtw
# from scipy.stats.stats import kendalltau

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
sigmasPath = './oldResults/out_aud_STRF_avgTime/'
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

pairwiseSpearmanMatrixFullTensor  = np.zeros((nbDatasets,nbDatasets))
pairwiseSpearmanMatrixScaleRate   = np.zeros((nbDatasets,nbDatasets))
pairwiseSpearmanMatrixFreqRate    = np.zeros((nbDatasets,nbDatasets))
pairwiseSpearmanMatrixFreqScale   = np.zeros((nbDatasets,nbDatasets))

manhattan_distance = lambda x, y: np.abs(x - y)

for iDataset in range(nbDatasets):
    print(iDataset)
    for jDataset in range(nbDatasets):
        print(jDataset)
        strfTensorI = np.reshape(sigmasTab[iDataset],(128,11,22))
        strf_scale_rateI, strf_freq_rateI, strf_freq_scaleI = avgvec2strfavg(strf2avgvec(strfTensorI))
        strfTensorJ = np.reshape(sigmasTab[jDataset],(128,11,22))
        strf_scale_rateJ, strf_freq_rateJ, strf_freq_scaleJ = avgvec2strfavg(strf2avgvec(strfTensorJ))

        pairwisePearsonMatrixFullTensor[iDataset][jDataset]  = pearsonr(strfTensorI.flatten(),strfTensorJ.flatten())[0]**2
        pairwisePearsonMatrixScaleRate[iDataset][jDataset]   = pearsonr(strf_scale_rateI.flatten(),strf_scale_rateJ.flatten())[0]**2
        pairwisePearsonMatrixFreqRate[iDataset][jDataset]    = pearsonr(strf_freq_rateI.flatten(),strf_freq_rateJ.flatten())[0]**2
        pairwisePearsonMatrixFreqScale[iDataset][jDataset]   = pearsonr(strf_freq_scaleI.flatten(),strf_freq_scaleJ.flatten())[0]**2

        pairwiseSpearmanMatrixFullTensor[iDataset][jDataset]  = spearmanr(strfTensorI.flatten(),strfTensorJ.flatten())[0]**2
        pairwiseSpearmanMatrixScaleRate[iDataset][jDataset]   = spearmanr(strf_scale_rateI.flatten(),strf_scale_rateJ.flatten())[0]**2
        pairwiseSpearmanMatrixFreqRate[iDataset][jDataset]    = spearmanr(strf_freq_rateI.flatten(),strf_freq_rateJ.flatten())[0]**2
        pairwiseSpearmanMatrixFreqScale[iDataset][jDataset]   = spearmanr(strf_freq_scaleI.flatten(),strf_freq_scaleJ.flatten())[0]**2

        # print(np.asarray(strfTensorJ.flatten()).ravel()) np.asarray(strfTensorI.flatten()),np.asarray(strfTensorJ.flatten())
    
        # # pairwisePearsonMatrixFullTensor[iDataset][jDataset]  = cdist_dtw(strfTensorI.flatten(),strfTensorJ.flatten())       
        # pairwisePearsonMatrixScaleRate[iDataset][jDataset]   = dtw(strf_scale_rateI.flatten(),strf_scale_rateJ.flatten())
        # pairwisePearsonMatrixFreqRate[iDataset][jDataset]    = dtw(strf_freq_rateI.flatten(),strf_freq_rateJ.flatten())
        # pairwisePearsonMatrixFreqScale[iDataset][jDataset]   = dtw(strf_freq_scaleI.flatten(),strf_freq_scaleJ.flatten())

##### 1
fig = pylab.figure(figsize=(8,8))
D = pairwiseSpearmanMatrixFullTensor
ax1 = fig.add_axes([0.17,0.76,0.7,0.22])
Y = sch.linkage(D, method='complete')
Z1 = sch.dendrogram(Y)
ax1.set_xticks([])
ax1.set_yticks([])

axmatrix = fig.add_axes([0.17,0.05,0.7,0.7])
idx1 = Z1['leaves']
D = D[idx1,:]
D = D[:,idx1]
D = np.flipud(D)
im = axmatrix.matshow(D, aspect='auto', origin='lower')
axmatrix.set_xticks([])
axmatrix.set_yticks([])

tsp=np.asarray(['B2010 165','G1977 311','G1978 311','I1993O 262','I1993R 262','I1993W 262', 'McA1995 311','P2012A3 220',
             'P2012DX4 311','P2012GD4 415', 'L2000H 311','L2000C 311','L2000P 311','S2016e2set1 311'
             ,'S2016e2set2 311','S2016e2set3 311','S2016_e3 311'])
tsp = np.flip(tsp[idx1])
axmatrix.set_yticks(range(17))
axmatrix.set_yticklabels(tsp, minor=False)
plt.setp(axmatrix.get_yticklabels(), rotation=30, ha="right",
         rotation_mode="anchor",weight='bold')
axmatrix.yaxis.set_label_position('left')
axmatrix.yaxis.tick_left()
plt.savefig('_spearman_pairwiseSpearmanMatrixFullTensor.eps')
plt.show()



##### 2
fig = pylab.figure(figsize=(8,8))
D = pairwiseSpearmanMatrixScaleRate
ax1 = fig.add_axes([0.17,0.76,0.7,0.22])
Y = sch.linkage(D, method='complete')
Z1 = sch.dendrogram(Y)
ax1.set_xticks([])
ax1.set_yticks([])

axmatrix = fig.add_axes([0.17,0.05,0.7,0.7])
idx1 = Z1['leaves']
D = D[idx1,:]
D = D[:,idx1]
D = np.flipud(D)
im = axmatrix.matshow(D, aspect='auto', origin='lower')
axmatrix.set_xticks([])
axmatrix.set_yticks([])

tsp=np.asarray(['B2010 165','G1977 311','G1978 311','I1993O 262','I1993R 262','I1993W 262', 'McA1995 311','P2012A3 220',
             'P2012DX4 311','P2012GD4 415', 'L2000H 311','L2000C 311','L2000P 311','S2016e2set1 311'
             ,'S2016e2set2 311','S2016e2set3 311','S2016_e3 311'])
tsp = np.flip(tsp[idx1])
axmatrix.set_yticks(range(17))
axmatrix.set_yticklabels(tsp, minor=False)
plt.setp(axmatrix.get_yticklabels(), rotation=30, ha="right",
         rotation_mode="anchor",weight='bold')
axmatrix.yaxis.set_label_position('left')
axmatrix.yaxis.tick_left()
plt.savefig('_spearman_pairwisePearsonMatrixScaleRate.eps')
plt.show()

##### 3
fig = pylab.figure(figsize=(8,8))
D = pairwiseSpearmanMatrixFreqRate
ax1 = fig.add_axes([0.17,0.76,0.7,0.22])
Y = sch.linkage(D, method='complete')
Z1 = sch.dendrogram(Y)
ax1.set_xticks([])
ax1.set_yticks([])

axmatrix = fig.add_axes([0.17,0.05,0.7,0.7])
idx1 = Z1['leaves']
D = D[idx1,:]
D = D[:,idx1]
D = np.flipud(D)
im = axmatrix.matshow(D, aspect='auto', origin='lower')
axmatrix.set_xticks([])
axmatrix.set_yticks([])

tsp=np.asarray(['B2010 165','G1977 311','G1978 311','I1993O 262','I1993R 262','I1993W 262', 'McA1995 311','P2012A3 220',
             'P2012DX4 311','P2012GD4 415', 'L2000H 311','L2000C 311','L2000P 311','S2016e2set1 311'
             ,'S2016e2set2 311','S2016e2set3 311','S2016_e3 311'])
tsp = np.flip(tsp[idx1])
axmatrix.set_yticks(range(17))
axmatrix.set_yticklabels(tsp, minor=False)
plt.setp(axmatrix.get_yticklabels(), rotation=30, ha="right",
         rotation_mode="anchor",weight='bold')
axmatrix.yaxis.set_label_position('left')
axmatrix.yaxis.tick_left()
plt.savefig('_spearman_pairwiseSpearmanMatrixFreqRate.eps')
plt.show()

##### 4
fig = pylab.figure(figsize=(8,8))
D = pairwiseSpearmanMatrixFreqScale
ax1 = fig.add_axes([0.17,0.76,0.7,0.22])
Y = sch.linkage(D, method='complete')
Z1 = sch.dendrogram(Y)
ax1.set_xticks([])
ax1.set_yticks([])

axmatrix = fig.add_axes([0.17,0.05,0.7,0.7])
idx1 = Z1['leaves']
D = D[idx1,:]
D = D[:,idx1]
D = np.flipud(D)
im = axmatrix.matshow(D, aspect='auto', origin='lower')
axmatrix.set_xticks([])
axmatrix.set_yticks([])

tsp=np.asarray(['B2010 165','G1977 311','G1978 311','I1993O 262','I1993R 262','I1993W 262', 'McA1995 311','P2012A3 220',
             'P2012DX4 311','P2012GD4 415', 'L2000H 311','L2000C 311','L2000P 311','S2016e2set1 311'
             ,'S2016e2set2 311','S2016e2set3 311','S2016_e3 311'])
tsp = np.flip(tsp[idx1])
axmatrix.set_yticks(range(17))
axmatrix.set_yticklabels(tsp, minor=False)
plt.setp(axmatrix.get_yticklabels(), rotation=30, ha="right",
         rotation_mode="anchor",weight='bold')
axmatrix.yaxis.set_label_position('left')
axmatrix.yaxis.tick_left()
plt.savefig('_spearman_pairwiseSpearmanMatrixFreqScale.eps')
plt.show()


# plt.subplot(221)
# plt.imshow(pairwisePearsonMatrixFullTensor)
# plt.subplot(222)
# plt.imshow(pairwisePearsonMatrixScaleRate)
# plt.subplot(223)
# plt.imshow(pairwisePearsonMatrixFreqRate)
# plt.subplot(224)
# plt.imshow(pairwisePearsonMatrixFreqScale)

print()
print('##################################')
print('All Datasets (128x22x11)')
print('Mutual Median pairwise correlations')
print('Full STRF - Mdn='+str(np.median(pairwisePearsonMatrixFullTensor[np.triu_indices(nbDatasets,1)])) + ' iqr='+str(iqr(pairwisePearsonMatrixFullTensor[np.triu_indices(nbDatasets,1)])))
print('Scale/Rate - Mdn='+str(np.median(pairwisePearsonMatrixScaleRate[np.triu_indices(nbDatasets,1)]))  + ' iqr='+str(iqr(pairwisePearsonMatrixScaleRate[np.triu_indices(nbDatasets,1)])))
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

#         pairwisePearsonMatrixFullTensor[iDataset][jDataset]  = pearsonr(strfTensorI.flatten(),strfTensorJ.flatten())[0]**2
#         pairwisePearsonMatrixScaleRate[iDataset][jDataset]   = pearsonr(strf_scale_rateI.flatten(),strf_scale_rateJ.flatten())[0]**2
#         pairwisePearsonMatrixFreqRate[iDataset][jDataset]    = pearsonr(strf_freq_rateI.flatten(),strf_freq_rateJ.flatten())[0]**2
#         pairwisePearsonMatrixFreqScale[iDataset][jDataset]   = pearsonr(strf_freq_scaleI.flatten(),strf_freq_scaleJ.flatten())[0]**2

# print(pairwisePearsonMatrixFullTensor.shape)
# plt.imshow(pairwisePearsonMatrixFullTensor)
# plt.show()

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






