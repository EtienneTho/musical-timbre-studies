### Thoret, Caramiaux (c) 2020
###
### This script computes the correlation between the stimuli variability and the optimized metrics
### This scripts replicates the results of the section "Timbre perceptual metrics are experiment-specific"
###

import os
import numpy as np  
import pickle
import math
import matplotlib.pyplot as plt 
from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr, iqr
import pingouin as pg

sigmasPath = './out_aud_STRF_avgTime/'
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

tabCorrVar_full      = []
tabCorrVar_ScaleRate = []
tabCorrVar_FreqRate  = []
tabCorrVar_FreqScale = []

for r, d, f in os.walk(sigmasPath):
    for file in sorted(f):
        if 'resultsOptims' in file:
            sigmas = pickle.load(open(os.path.join(sigmasPath, file), 'rb')) # load sigmas
            sigmasTab.append(sigmas['sigmas'].flatten()) 
            represenationsVariances = np.std(sigmas['representations'],axis=1) # compute variances of representations
            represenationsVariancesTensor = np.reshape(represenationsVariances,(128,11,22))
            strf_scale_rateVar, strf_freq_rateVar, strf_freq_scaleVar = avgvec2strfavg(strf2avgvec(represenationsVariancesTensor)) # compute representations variances projections
 
            sigmas_ = sigmas['sigmas'] # load sigmas
            sigmasTensor = np.reshape(sigmas_,(128,11,22))
            strf_scale_rateSig, strf_freq_rateSig, strf_freq_scaleSig = avgvec2strfavg(strf2avgvec(sigmasTensor)) # compute metrics projections
            tabCorrVar_full.append(pearsonr(sigmas_.flatten(),represenationsVariances.flatten())[0]**2)
            tabCorrVar_ScaleRate.append(pearsonr(strf_scale_rateSig.flatten(),strf_scale_rateVar.flatten())[0]**2)
            tabCorrVar_FreqRate.append(pearsonr(strf_freq_rateSig.flatten(),strf_freq_rateVar.flatten())[0]**2)
            tabCorrVar_FreqScale.append(pearsonr(strf_freq_scaleSig.flatten(),strf_freq_scaleVar.flatten())[0]**2)                        


# print()
# print('Averaged explained variance by metrics with representations variability (std)' )
# print()
# print('Full STRF : ' + str(np.mean(tabCorrVar_full)) + ' (+/-) '+ str(np.std(tabCorrVar_full)))
# print('Scale/Rate : ' + str(np.mean(tabCorrVar_ScaleRate)) + ' (+/-) '+ str(np.std(tabCorrVar_ScaleRate)))
# print('Freq/Rate : ' + str(np.mean(tabCorrVar_FreqRate)) + ' (+/-) '+ str(np.std(tabCorrVar_FreqRate)))
# print('Freq/Scale : ' + str(np.mean(tabCorrVar_FreqScale)) + ' (+/-) '+ str(np.std(tabCorrVar_FreqScale)))

print()
print('Median explained variance by metrics with representations variability (std)' )
print()
print('Full STRF : ' + str(np.median(tabCorrVar_full)) + ' (+/-) '+ str(iqr(tabCorrVar_full)))
print('Scale/Rate : ' + str(np.median(tabCorrVar_ScaleRate)) + ' (+/-) '+ str(iqr(tabCorrVar_ScaleRate)))
print('Freq/Rate : ' + str(np.median(tabCorrVar_FreqRate)) + ' (+/-) '+ str(iqr(tabCorrVar_FreqRate)))
print('Freq/Scale : ' + str(np.median(tabCorrVar_FreqScale)) + ' (+/-) '+ str(iqr(tabCorrVar_FreqScale)))

print()
print('Full STRF analysis')
for r, d, f in os.walk(sigmasPath):
    for file in sorted(f):
        if 'resultsOptims' in file:
            sigmas = pickle.load(open(os.path.join(sigmasPath, file), 'rb')) # load sigmas
            sigmasTab.append(sigmas['sigmas'].flatten()) 
            represenationsVariances = np.std(sigmas['representations'],axis=1) # compute variances of representations
            represenationsVariancesTensor = np.reshape(represenationsVariances,(128,11,22))
            strf_scale_rateVar, strf_freq_rateVar, strf_freq_scaleVar = avgvec2strfavg(strf2avgvec(represenationsVariancesTensor)) # compute representations variances projections
 
            sigmas_ = sigmas['sigmas'] # load sigmas
            sigmasTensor = np.reshape(sigmas_,(128,11,22))
            strf_scale_rateSig, strf_freq_rateSig, strf_freq_scaleSig = avgvec2strfavg(strf2avgvec(sigmasTensor)) # compute metrics projections
            pg_ = pg.corr(sigmas_.flatten(), represenationsVariances.flatten())
            aa = pg_['CI95%'][0][0]**2
            bb = pg_['CI95%'][0][1]**2
            lower = np.min([aa,bb])
            upper = np.max([aa,bb])
            p = pg_['p-val'][0]
            r = pg_['r'][0]
            power = pg_['power'][0]
            df = pg_['n'][0]-2
            # print(file+' r^2='+str(r**2)+' p='+str(p)+' lower='+str(lower)+' upper='+str(upper)+' power='+str(power)+' df='+str(df))
            print(str(df)+' '+"%.2f" %(r**2)+' '+"%.3f" %p+' ['+"%.3f" %(lower)+';'+"%.3f" %(upper)+'] '+"%.3f" %(power))
            


print()
print('Scale/Rate')
for r, d, f in os.walk(sigmasPath):
    for file in sorted(f):
        if 'resultsOptims' in file:
            sigmas = pickle.load(open(os.path.join(sigmasPath, file), 'rb')) # load sigmas
            sigmasTab.append(sigmas['sigmas'].flatten()) 
            represenationsVariances = np.std(sigmas['representations'],axis=1) # compute variances of representations
            represenationsVariancesTensor = np.reshape(represenationsVariances,(128,11,22))
            strf_scale_rateVar, strf_freq_rateVar, strf_freq_scaleVar = avgvec2strfavg(strf2avgvec(represenationsVariancesTensor)) # compute representations variances projections
 
            sigmas_ = sigmas['sigmas'] # load sigmas
            sigmasTensor = np.reshape(sigmas_,(128,11,22))
            strf_scale_rateSig, strf_freq_rateSig, strf_freq_scaleSig = avgvec2strfavg(strf2avgvec(sigmasTensor)) # compute metrics projections
            pg_ = pg.corr(strf_scale_rateSig.flatten(),strf_scale_rateVar.flatten())
            aa = pg_['CI95%'][0][0]**2
            bb = pg_['CI95%'][0][1]**2
            lower = np.min([aa,bb])
            upper = np.max([aa,bb])
            p = pg_['p-val'][0]
            r = pg_['r'][0]
            power = pg_['power'][0]
            df = pg_['n'][0]-2
            # print(file+' r^2='+str(r**2)+' p='+str(p)+' lower='+str(lower)+' upper='+str(upper)+' power='+str(power)+' df='+str(df))
            print(str(df)+' '+"%.2f" %(r**2)+' '+"%.3f" %p+' ['+"%.3f" %(lower)+';'+"%.3f" %(upper)+'] '+"%.3f" %(power))


print()
print('Freq/Rate')
for r, d, f in os.walk(sigmasPath):
    for file in sorted(f):
        if 'resultsOptims' in file:
            sigmas = pickle.load(open(os.path.join(sigmasPath, file), 'rb')) # load sigmas
            sigmasTab.append(sigmas['sigmas'].flatten()) 
            represenationsVariances = np.std(sigmas['representations'],axis=1) # compute variances of representations
            represenationsVariancesTensor = np.reshape(represenationsVariances,(128,11,22))
            strf_scale_rateVar, strf_freq_rateVar, strf_freq_scaleVar = avgvec2strfavg(strf2avgvec(represenationsVariancesTensor)) # compute representations variances projections
 
            sigmas_ = sigmas['sigmas'] # load sigmas
            sigmasTensor = np.reshape(sigmas_,(128,11,22))
            strf_scale_rateSig, strf_freq_rateSig, strf_freq_scaleSig = avgvec2strfavg(strf2avgvec(sigmasTensor)) # compute metrics projections
            pg_ = pg.corr(strf_freq_rateSig.flatten(),strf_freq_rateVar.flatten())
            aa = pg_['CI95%'][0][0]**2
            bb = pg_['CI95%'][0][1]**2
            lower = np.min([aa,bb])
            upper = np.max([aa,bb])
            p = pg_['p-val'][0]
            r = pg_['r'][0]
            power = pg_['power'][0]
            df = pg_['n'][0]-2
            # print(file+' r^2='+str(r**2)+' p='+str(p)+' lower='+str(lower)+' upper='+str(upper)+' power='+str(power)+' df='+str(df))
            print(str(df)+' '+"%.2f" %(r**2)+' '+"%.3f" %p+' ['+"%.3f" %(lower)+';'+"%.3f" %(upper)+'] '+"%.3f" %(power))


print()
print('Freq/Scale')
for r, d, f in os.walk(sigmasPath):
    for file in sorted(f):
        if 'resultsOptims' in file:
            sigmas = pickle.load(open(os.path.join(sigmasPath, file), 'rb')) # load sigmas
            sigmasTab.append(sigmas['sigmas'].flatten()) 
            represenationsVariances = np.std(sigmas['representations'],axis=1) # compute variances of representations
            represenationsVariancesTensor = np.reshape(represenationsVariances,(128,11,22))
            strf_scale_rateVar, strf_freq_rateVar, strf_freq_scaleVar = avgvec2strfavg(strf2avgvec(represenationsVariancesTensor)) # compute representations variances projections
 
            sigmas_ = sigmas['sigmas'] # load sigmas
            sigmasTensor = np.reshape(sigmas_,(128,11,22))
            strf_scale_rateSig, strf_freq_rateSig, strf_freq_scaleSig = avgvec2strfavg(strf2avgvec(sigmasTensor)) # compute metrics projections
            pg_ = pg.corr(strf_freq_scaleSig.flatten(),strf_freq_scaleVar.flatten())
            aa = pg_['CI95%'][0][0]**2
            bb = pg_['CI95%'][0][1]**2
            lower = np.min([aa,bb])
            upper = np.max([aa,bb])
            p = pg_['p-val'][0]
            r = pg_['r'][0]
            power = pg_['power'][0]
            df = pg_['n'][0]-2
            # print(file+' r^2='+str(r**2)+' p='+str(p)+' lower='+str(lower)+' upper='+str(upper)+' power='+str(power)+' df='+str(df))
            print(str(df)+' '+"%.2f" %(r**2)+' '+"%.3f" %p+' ['+"%.3f" %(lower)+';'+"%.3f" %(upper)+'] '+"%.3f" %(power))



