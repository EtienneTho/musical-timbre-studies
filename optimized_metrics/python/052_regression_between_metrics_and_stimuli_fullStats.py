### Thoret, Caramiaux (c) 2020
###
### This script computes the multiple linear regression between the optimized metrics and the stimuli representation
### This scripts replicates the results of the section "Timbre perceptual metrics are experiment-specific" with full statistics
###

import os
import numpy as np  
import pickle
import math
import matplotlib.pyplot as plt 
from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr, iqr
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
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


tab_r2_full      = []
tab_r2_ScaleRate = []
tab_r2_FreqRate  = []
tab_r2_FreqScale = []

tabCorr = []
r_withall_temp = []
p_withall_temp = []
ci95range_withall_temp = []
df_withall_temp = []
power_withall_temp = []


for r, d, f in os.walk(sigmasPath):
    for file in sorted(f):
        if 'resultsOptims' in file:
            print()
            print(file)
            sigmas = pickle.load(open(os.path.join(sigmasPath, file), 'rb')) # load sigmas
            # print(sigmas['representations'].shape)
            # print(sigmas['sigmas'].shape)
            # print(sigmas['sigmas'].shape)
            X = (sigmas['representations'])
            X = np.reshape(X, (128, 11, 22, X.shape[1]))
            X_full = np.reshape(X, (X.shape[0]*X.shape[1]*X.shape[2],X.shape[3]))

            y = (sigmas['sigmas'])
            y = np.reshape(y, (128, 11, 22))
            y_full = np.reshape(y, (y.shape[0]*y.shape[1]*y.shape[2]))

            reg = LinearRegression(normalize=True).fit(X_full, y_full)
            print("full: "+str(reg.score(X_full, y_full)))
            coefFull = reg.coef_
            tab_r2_full.append(reg.score(X_full, y_full))
            model = sm.OLS(y_full, sm.tools.tools.add_constant(X_full)).fit()
            print(model.summary())
            # print('R2: ', model.rsquared)

            # avg w.r.t. freq (scale/rate)
            dimAvg = 0
            X0 = np.mean(X, axis=dimAvg)
            X0 = np.reshape(X0, (X0.shape[0]*X0.shape[1],X0.shape[2]))
            y0 = np.mean(y, axis=dimAvg)
            y0 = np.reshape(y0, (y0.shape[0]*y0.shape[1]))

            reg = LinearRegression(normalize=True).fit(X0, y0)

            print("scale/rate: "+str(reg.score(X0, y0)))
            tab_r2_ScaleRate.append(reg.score(X0, y0))
            coef0 = reg.coef_
            model = sm.OLS(y0, sm.tools.tools.add_constant(X0)).fit()
            print(model.summary())            

            # avg w.r.t. scale (freq/rate)
            dimAvg = 1
            X1 = np.mean(X, axis=dimAvg)
            X1 = np.reshape(X1, (X1.shape[0]*X1.shape[1],X1.shape[2]))
            y1 = np.mean(y, axis=dimAvg)
            y1 = np.reshape(y1, (y1.shape[0]*y1.shape[1]))

            reg = LinearRegression(normalize=True).fit(X1, y1)
            print("freq/rate: "+str(reg.score(X1, y1)))
            tab_r2_FreqRate.append(reg.score(X1, y1))
            coef1 = reg.coef_
            model = sm.OLS(y1, sm.tools.tools.add_constant(X1)).fit()
            print(model.summary())            

            # avg w.r.t. scale (freq/scale)
            dimAvg = 2
            X2 = np.mean(X, axis=dimAvg)
            X2 = np.reshape(X2, (X.shape[0]*X2.shape[1],X2.shape[2]))
            y2 = np.mean(y, axis=dimAvg)
            y2 = np.reshape(y2, (y2.shape[0]*y2.shape[1]))

            reg = LinearRegression(normalize=True).fit(X2, y2)
            print("freq/scale: "+str(reg.score(X2, y2)))
            tab_r2_FreqScale.append(reg.score(X2, y2))
            coef2 = reg.coef_
            model = sm.OLS(y2, sm.tools.tools.add_constant(X2)).fit()
            print(model.summary())            

            # corr btw weights
            print("scale/rate vs. freq/rate: "+str(math.pow(spearmanr(coef0, coef1)[0],2)))
            print("scale/rate vs. freq/scale: "+str(math.pow(spearmanr(coef0, coef2)[0],2)))
            print("freq/rate vs. freq/scale: "+str(math.pow(spearmanr(coef1, coef2)[0],2)))
            print("full vs. scale/rate: "+str(math.pow(spearmanr(coefFull, coef0)[0],2)))
            print("full vs. freq/rate: "+str(math.pow(spearmanr(coefFull, coef1)[0],2)))
            print("full vs. freq/scale: "+str(math.pow(spearmanr(coefFull, coef2)[0],2)))


            tabCorr.append(math.pow(spearmanr(coef0, coef1)[0],2))
            tabCorr.append(math.pow(spearmanr(coef0, coef2)[0],2))
            tabCorr.append(math.pow(spearmanr(coef1, coef2)[0],2))
            tabCorr.append(math.pow(spearmanr(coefFull, coef0)[0],2))
            tabCorr.append(math.pow(spearmanr(coefFull, coef1)[0],2))
            tabCorr.append(math.pow(spearmanr(coefFull, coef2)[0],2))
            ##
            pg_ = pg.corr(coef0, coef1, method='spearman')
            aa = pg_['CI95%'][0][0]**2
            bb = pg_['CI95%'][0][1]**2
            lower_ = np.min([aa,bb])
            upper_ = np.max([aa,bb])
            p_ = pg_['p-val'][0]
            r_ = pg_['r'][0]
            power_ = pg_['power'][0]
            df_ = pg_['n'][0]-2
            r_withall_temp.append(r_)
            p_withall_temp.append(p_)
            temp___ = (upper_-lower_)
            ci95range_withall_temp.append(temp___)
            power_withall_temp.append(power_)
            df_withall_temp.append(df_)
            ##
            pg_ = pg.corr(coef0, coef2, method='spearman')
            aa = pg_['CI95%'][0][0]**2
            bb = pg_['CI95%'][0][1]**2
            lower_ = np.min([aa,bb])
            upper_ = np.max([aa,bb])
            p_ = pg_['p-val'][0]
            r_ = pg_['r'][0]
            power_ = pg_['power'][0]
            df_ = pg_['n'][0]-2
            r_withall_temp.append(r_)
            p_withall_temp.append(p_)
            temp___ = (upper_-lower_)
            ci95range_withall_temp.append(temp___)
            power_withall_temp.append(power_)
            df_withall_temp.append(df_)
            ##
            pg_ = pg.corr(coef1, coef2, method='spearman')
            aa = pg_['CI95%'][0][0]**2
            bb = pg_['CI95%'][0][1]**2
            lower_ = np.min([aa,bb])
            upper_ = np.max([aa,bb])
            p_ = pg_['p-val'][0]
            r_ = pg_['r'][0]
            power_ = pg_['power'][0]
            df_ = pg_['n'][0]-2
            r_withall_temp.append(r_)
            p_withall_temp.append(p_)
            temp___ = (upper_-lower_)
            ci95range_withall_temp.append(temp___)
            power_withall_temp.append(power_)
            df_withall_temp.append(df_)
            ##
            pg_ = pg.corr(coefFull, coef0, method='spearman')
            aa = pg_['CI95%'][0][0]**2
            bb = pg_['CI95%'][0][1]**2
            lower_ = np.min([aa,bb])
            upper_ = np.max([aa,bb])
            p_ = pg_['p-val'][0]
            r_ = pg_['r'][0]
            power_ = pg_['power'][0]
            df_ = pg_['n'][0]-2
            r_withall_temp.append(r_)
            p_withall_temp.append(p_)
            temp___ = (upper_-lower_)
            ci95range_withall_temp.append(temp___)
            power_withall_temp.append(power_)
            df_withall_temp.append(df_)
            ##
            pg_ = pg.corr(coefFull, coef1, method='spearman')
            aa = pg_['CI95%'][0][0]**2
            bb = pg_['CI95%'][0][1]**2
            lower_ = np.min([aa,bb])
            upper_ = np.max([aa,bb])
            p_ = pg_['p-val'][0]
            r_ = pg_['r'][0]
            power_ = pg_['power'][0]
            df_ = pg_['n'][0]-2
            r_withall_temp.append(r_)
            p_withall_temp.append(p_)
            temp___ = (upper_-lower_)
            ci95range_withall_temp.append(temp___)
            power_withall_temp.append(power_)
            df_withall_temp.append(df_)
            ##
            pg_ = pg.corr(coefFull, coef2, method='spearman')
            aa = pg_['CI95%'][0][0]**2
            bb = pg_['CI95%'][0][1]**2
            lower_ = np.min([aa,bb])
            upper_ = np.max([aa,bb])
            p_ = pg_['p-val'][0]
            r_ = pg_['r'][0]
            power_ = pg_['power'][0]
            df_ = pg_['n'][0]-2
            r_withall_temp.append(r_)
            p_withall_temp.append(p_)
            temp___ = (upper_-lower_)
            ci95range_withall_temp.append(temp___)
            power_withall_temp.append(power_)
            df_withall_temp.append(df_)




tab_r2_full      = np.asarray(tab_r2_full)
tab_r2_ScaleRate = np.asarray(tab_r2_ScaleRate)
tab_r2_FreqRate  = np.asarray(tab_r2_FreqRate)
tab_r2_FreqScale = np.asarray(tab_r2_FreqScale)

r_withall_temp = np.asarray(r_withall_temp)
p_withall_temp = np.asarray(p_withall_temp)
ci95range_withall_temp = np.asarray(ci95range_withall_temp)
df_withall_temp = np.asarray(df_withall_temp)
power_withall_temp = np.asarray(power_withall_temp)

print()
print("r2 full stmf: Mdn="+str(np.median(tab_r2_full))+" IQR="+str(iqr(tab_r2_full)))
print("r2 scale rate: Mdn="+str(np.median(tab_r2_ScaleRate))+" IQR="+str(iqr(tab_r2_ScaleRate)))
print("r2 freq rate: Mdn="+str(np.median(tab_r2_FreqRate))+" IQR="+str(iqr(tab_r2_FreqRate)))
print("r2 freq scale: Mdn="+str(np.median(tab_r2_FreqScale))+" IQR="+str(iqr(tab_r2_FreqScale)))

tabCorr = np.asarray(tabCorr)
tabCorr = np.reshape(tabCorr,(17,6))

r_withall_temp = np.reshape(r_withall_temp,(17,6))
p_withall_temp = np.reshape(p_withall_temp,(17,6))
ci95range_withall_temp = np.reshape(ci95range_withall_temp,(17,6))
df_withall_temp = np.reshape(df_withall_temp,(17,6))
power_withall_temp = np.reshape(power_withall_temp,(17,6))

print("correlations btw weight: Mdn="+str(np.median(tabCorr,axis=0))+" IQR="+str(iqr(tabCorr,axis=0)))


print(str(np.median(np.asarray(df_withall_temp),axis=0)))

print(str(np.median(np.asarray(p_withall_temp),axis=0)))
print(str(np.min(np.asarray(p_withall_temp),axis=0)))
print(str(np.max(np.asarray(p_withall_temp),axis=0)))

print(str(np.median(np.asarray(ci95range_withall_temp),axis=0)))
print(str(np.min(np.asarray(ci95range_withall_temp),axis=0)))
print(str(np.max(np.asarray(ci95range_withall_temp),axis=0)))

print(str(np.median(np.asarray(power_withall_temp),axis=0)))
print(str(np.min(np.asarray(power_withall_temp),axis=0)))
print(str(np.max(np.asarray(power_withall_temp),axis=0)))



