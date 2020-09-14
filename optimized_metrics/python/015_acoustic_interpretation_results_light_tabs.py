### Thoret, Caramiaux (c) 2020
###
### This script computes the analysis of the cross-validation
### This script replicates the results of the section "Optimized metrics simulating human dissimilarity ratings" with full statistics
###

import os
import numpy as np  
import pickle
import math
import matplotlib.pyplot as plt 
from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr, iqr
import pingouin as pg

# from scipy.stats.stats import kendalltau

sigmas_tab = []

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


###############################

def cross_correlation(data_as_array):
    correlations = []    
    r_temp = []
    p_temp = []
    ci95range_temp = []
    power_temp = []
    df_temp = []    
    for ki1 in range(len(data_as_array)):
        for ki2 in range(len(data_as_array)):
            if ki2 > ki1: 
                correlations.append(pearsonr(data_as_array[ki1], data_as_array[ki2])[0] ** 2)
                pg_ = pg.corr(data_as_array[ki1], data_as_array[ki2])
                aa = pg_['CI95%'][0][0]**2
                bb = pg_['CI95%'][0][1]**2
                lower_ = np.min([aa,bb])
                upper_ = np.max([aa,bb])
                p_ = pg_['p-val'][0]
                r_ = pg_['r'][0]
                power_ = pg_['power'][0]
                df_ = pg_['n'][0]-2
                r_temp.append(r_)
                p_temp.append(p_)
                temp___ = (upper_-lower_)
                ci95range_temp.append(temp___)
                power_temp.append(power_)
                df_temp.append(df_)                
    return correlations, r_temp, p_temp, ci95range_temp, power_temp, df_temp


# def get_crossval_sigmas_from_folder(
#     timbre_spaces = ['Barthet2010','Grey1977','Grey1978','Iverson1993_Onset',
#                  'Iverson1993_Remainder','Iverson1993_Whole', 'McAdams1995','Patil2012_A3',
#                  'Patil2012_DX4','Patil2012_GD4','Siedenburg2016_e3', 'Lakatos2000_Harm',
#                  'Lakatos2000_Comb','Lakatos2000_Perc','Siedenburg2016_e2set1','Siedenburg2016_e2set2',
#                  'Siedenburg2016_e2set3'], 
#     representation = ['auditory_strf'], 
#     folder='out_crossval_storingsigs', 
#     averaging='avg_time_avg_freq', 
#     early_stopping=True):
#     sigmas = []
#     for tsp in timbre_spaces:
#         for root, dirs, files in os.walk(os.path.join(folder, tsp.lower())):
#             if representation[0] in root: 
#                 if 'fold' not in root.split('/')[-1]:
#                     for f in files:
#                         if averaging + '_' + tsp in f:
#                             # print(os.path.join(root, f))
#                             optim = pickle.load(open(os.path.join(root, f), 'rb'))
#                             sigmas_ = []
#                             for fold in range(len(optim['correlations'])):
#                                 if early_stopping:
#                                     m_i = np.argmin(optim['correlations_testing'][fold])
#                                 else:
#                                     m_i = -1
#                                 sigmas_.append(optim['sigmas'][fold][m_i])
#                             sigmas_ = np.array(sigmas_)
#                             print('{} - corr={:.3f} (std={:.3f})'.format(tsp, np.mean(cross_correlation(sigmas_)), np.std(cross_correlation(sigmas_))))
#                             sigmas.append(np.mean(sigmas_, axis=0))
#     return sigmas


def get_crossval_sigmas_from_folder(
    timbre_spaces = ['Barthet2010','Grey1977','Grey1978','Iverson1993_Onset',
                 'Iverson1993_Remainder','Iverson1993_Whole', 'McAdams1995','Patil2012_A3',
                 'Patil2012_DX4','Patil2012_GD4', 'Lakatos2000_Harm',
                 'Lakatos2000_Comb','Lakatos2000_Perc','Siedenburg2016_e2set1','Siedenburg2016_e2set2',
                 'Siedenburg2016_e2set3','Siedenburg2016_e3'],
    representation = ['auditory_strf'],
    folder='results_light',
    averaging='avg_time_avg_freq',
    early_stopping=True,
    folder_old='all'):
    sigmas = []
    correlation_testing = []
    correlation_training = []
    correlation_with_all = []
    cross_corr = []
    for tsp in timbre_spaces:
        for root, dirs, files in os.walk(os.path.join(folder, tsp.lower())):
            for f in files:
                if averaging + '_' + tsp.lower() in f:
                    results = pickle.load(open(os.path.join(root, f), 'rb'))
                    # print(files)
                    metricOnAll = pickle.load(open(os.path.join(root, 'all.pkl'), 'rb'))
                    metricOnAll = metricOnAll['sigmas'].flatten()

                    sigmas_ = [results['sigmas'][fold] for fold in range(len(results['correlations']))]
                    sigmas_ = np.array(sigmas_)
                    corr_ = [pearsonr(results['sigmas'][fold].flatten(),metricOnAll)[0] for fold in range(len(results['correlations']))]
                    r_withall_temp = []
                    p_withall_temp = []
                    ci95range_withall_temp = []
                    df_withall_temp = []
                    power_withall_temp = []

                    for fold in range(len(results['correlations'])):
                        pearsonr(results['sigmas'][fold].flatten(),metricOnAll)[0]
                        pg_ = pg.corr(results['sigmas'][fold].flatten(), metricOnAll)
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


                    file_ = 'resultsOptims_strf'+f.split('results')[1].split('.pkl')[0]+'.pkl'
                    # resultsOptims_strf_avg_time_Grey1978_F311.pkl

                    sigmas__ = pickle.load(open(os.path.join('./out_aud_STRF_crossval/', file_), 'rb'))
                    r_training_temp = []
                    p_training_temp = []
                    ci95range_training_temp = []
                    power_training_temp = []
                    df_training_temp = []
                    
                    r_testing_temp = []
                    p_testing_temp = []
                    ci95range_testing_temp = []
                    power_testing_temp = []
                    df_testing_temp = []

                    pg_training = []
                    pg_testing = []

                    for fold in range(len(results['correlations'])):
                        # training data
                        train_idx = [i for i in range(sigmas__['representations'].shape[1]) if i != fold]                        
                        input_data_training  = sigmas__['representations'][:,train_idx]
                        target_data_training = sigmas__['dissimilarities'][train_idx, :]
                        target_data_training = target_data_training[:, train_idx]                        
                        r_, p_, lower_, upper_, power_, df_ = corr(input_data_training,results['sigmas'][fold],target_data_training)
                        r_training_temp.append(r_)
                        p_training_temp.append(p_)
                        temp___ = (upper_-lower_)
                        ci95range_training_temp.append(temp___)
                        power_training_temp.append(power_)
                        df_training_temp.append(df_)

                        # pg_training.append(corr(input_data_training,results['sigmas'][fold],target_data_training))


                        # testing data
                        test_idx = [fold]
                        ninstrus = sigmas__['representations'].shape[1]
                        input_data_testing = sigmas__['representations'][:, test_idx[0]]
                        target_data_testing = np.zeros((ninstrus-1, 1))

                        cpt_i = 0
                        for i in range(ninstrus):
                            if i > fold:
                                target_data_testing[cpt_i] = sigmas__['dissimilarities'][fold, i]
                                cpt_i += 1
                            elif i < fold:
                                target_data_testing[cpt_i] = sigmas__['dissimilarities'][i, fold]
                                cpt_i += 1                        
                        
                        # print(input_data_testing)
                        # print(target_data_testing)
                        # pg_testing.append(corr(input_data_testing,results['sigmas'][fold],target_data_testing))

                        test_input  = input_data_testing
                        test_target = target_data_testing

                        mean_target_test = np.mean(test_target)
                        std_target_test  = np.std(test_target)
                        kernel_test = np.zeros((ninstrus-1, 1))
                        # print(input_data_training.shape)
                        for i in range(len(kernel_test)):
                            # print(i)
                            kernel_test[i, 0] = -np.sum(
                                np.power(
                                    np.divide(test_input - input_data_training[:,i], 
                                        (results['sigmas'][fold] + np.finfo(float).eps)), 2))

                        # pg_testing.append(pearsonr(kernel_test[:,0], test_target[:,0])[0])

                        pg_ = pg.corr(np.asarray(kernel_test[:,0]), np.asarray(test_target[:,0]))
                        aa = pg_['CI95%'][0][0]**2
                        bb = pg_['CI95%'][0][1]**2
                        lower_ = np.min([aa,bb])
                        upper_ = np.max([aa,bb])
                        p_ = pg_['p-val'][0]
                        r_ = pg_['r'][0]
                        power_ = pg_['power'][0]
                        df_ = pg_['n'][0]-2
                        r_testing_temp.append(r_)
                        p_testing_temp.append(p_)
                        temp___ = (upper_-lower_)
                        ci95range_testing_temp.append(temp___)
                        power_testing_temp.append(power_)
                        df_testing_temp.append(df_)

                        # pg_testing.append(corr(input_data_testing,results['sigmas'][fold],target_data_testing))

                    # print(np.median(np.asarray(pg_training)**2))
                    # print(iqr(np.asarray(pg_training)**2))
                    # print(np.median(np.asarray(pg_testing)**2))
                    # print(iqr(np.asarray(pg_testing)**2)) 
                    # print(iqr(np.asarray(results['correlations_testing'])**2))
             
                    # print('{} - corr={:.3f} (std={:.3f})'.format(
                    #     tsp, 
                    #     np.mean(cross_correlation(sigmas_)), 
                    #     np.std(cross_correlation(sigmas_))))
                    # sigmas.append(np.mean(sigmas_, axis=0))           

                    print("%i" %(np.median(np.asarray(df_training_temp)))+' '

                        # +"%.2f" %np.median(np.asarray(r_training_temp)**2)+' '
                        # +"%.2f" %np.min(np.asarray(r_training_temp)**2)+' '
                        # +"%.2f" %np.max(np.asarray(r_training_temp)**2)+' '

                        +"%.3f" %np.median(np.asarray(p_training_temp))+' '
                        +"%.3f" %np.min(np.asarray(p_training_temp))+' '
                        +"%.3f" %np.max(np.asarray(p_training_temp))+' '

                        +"%.3f" %np.median(np.asarray(ci95range_training_temp))+' '
                        +"%.3f" %np.min(np.asarray(ci95range_training_temp))+' '
                        +"%.3f" %np.max(np.asarray(ci95range_training_temp))+' '

                        +"%.3f" %np.median(np.asarray(power_training_temp))+' '
                        +"%.3f" %np.min(np.asarray(power_training_temp))+' '
                        +"%.3f" %np.max(np.asarray(power_training_temp)))         

                    print("%i" %(np.median(np.asarray(df_testing_temp)))+' '

                        # +"%.2f" %np.median(np.asarray(r_training_temp)**2)+' '
                        # +"%.2f" %np.min(np.asarray(r_training_temp)**2)+' '
                        # +"%.2f" %np.max(np.asarray(r_training_temp)**2)+' '

                        +"%.3f" %np.median(np.asarray(p_testing_temp))+' '
                        +"%.3f" %np.min(np.asarray(p_testing_temp))+' '
                        +"%.3f" %np.max(np.asarray(p_testing_temp))+' '

                        +"%.3f" %np.median(np.asarray(ci95range_testing_temp))+' '
                        +"%.3f" %np.min(np.asarray(ci95range_testing_temp))+' '
                        +"%.3f" %np.max(np.asarray(ci95range_testing_temp))+' '

                        +"%.3f" %np.median(np.asarray(power_testing_temp))+' '
                        +"%.3f" %np.min(np.asarray(power_testing_temp))+' '
                        +"%.3f" %np.max(np.asarray(power_testing_temp))) 

                    _, r_temp_within, p_temp_within, ci95range_temp_within, power_temp_within, df_temp_within = cross_correlation(sigmas_)
                    print("%i" %(np.median(np.asarray(df_testing_temp)))+' '

                        # +"%.2f" %np.median(np.asarray(r_training_temp)**2)+' '
                        # +"%.2f" %np.min(np.asarray(r_training_temp)**2)+' '
                        # +"%.2f" %np.max(np.asarray(r_training_temp)**2)+' '

                        +"%.3f" %np.median(np.asarray(p_temp_within))+' '
                        +"%.3f" %np.min(np.asarray(p_temp_within))+' '
                        +"%.3f" %np.max(np.asarray(p_temp_within))+' '

                        +"%.3f" %np.median(np.asarray(ci95range_temp_within))+' '
                        +"%.3f" %np.min(np.asarray(ci95range_temp_within))+' '
                        +"%.3f" %np.max(np.asarray(ci95range_temp_within))+' '

                        +"%.3f" %np.median(np.asarray(power_temp_within))+' '
                        +"%.3f" %np.min(np.asarray(power_temp_within))+' '
                        +"%.3f" %np.max(np.asarray(power_temp_within))) 


                    print("%i" %(np.median(np.asarray(df_withall_temp)))+' '

                        # +"%.2f" %np.median(np.asarray(r_training_temp)**2)+' '
                        # +"%.2f" %np.min(np.asarray(r_training_temp)**2)+' '
                        # +"%.2f" %np.max(np.asarray(r_training_temp)**2)+' '

                        +"%.3f" %np.median(np.asarray(p_withall_temp))+' '
                        +"%.3f" %np.min(np.asarray(p_withall_temp))+' '
                        +"%.3f" %np.max(np.asarray(p_withall_temp))+' '

                        +"%.3f" %np.median(np.asarray(ci95range_withall_temp))+' '
                        +"%.3f" %np.min(np.asarray(ci95range_withall_temp))+' '
                        +"%.3f" %np.max(np.asarray(ci95range_withall_temp))+' '

                        +"%.3f" %np.median(np.asarray(power_withall_temp))+' '
                        +"%.3f" %np.min(np.asarray(power_withall_temp))+' '
                        +"%.3f" %np.max(np.asarray(power_withall_temp))) 


                    # print('{} correlations_training - corr: Mdn={:.2f} (IQR={:.3f})'.format(
                    #     tsp, 
                    #     np.median(np.asarray(results['correlations'])**2), 
                    #     iqr(np.asarray(results['correlations'])**2)))
                    # correlation_training.append(np.median(np.asarray(results['correlations'])**2))    


                    # print('{} correlations_testing - corr: Mdn={:.2f} (IQR={:.3f})'.format(
                    #     tsp, 
                    #     np.median(np.asarray(results['correlations_testing'])**2), 
                    #     iqr(np.asarray(results['correlations_testing'])**2)))
                    # correlation_testing.append(np.median(np.asarray(results['correlations_testing'])**2))


                    # print('{} within_corr - corr: Mdn={:.2f} (IQR={:.3f})'.format(
                    #     tsp, 
                    #     np.median(np.asarray(cross_correlation(sigmas_))), 
                    #     iqr(np.asarray(cross_correlation(sigmas_)))))
                    # cross_corr.append(np.median(np.asarray(cross_correlation(sigmas_))))                    
                    # sigmas.append(np.mean(sigmas_, axis=0))

                    # print('{} correlation with all - corr: Mdn={:.2f} (IQR={:.3f})'.format(
                    #     tsp, 
                    #     np.median(np.asarray(corr_)**2), 
                    #     iqr(np.asarray(corr_)**2)))
                    # print()
                    # correlation_with_all.append(np.median(np.asarray(corr_)**2))                    
                    # sigmas.append(np.mean(sigmas_, axis=0))                    

    print(correlation_training)
    print(correlation_testing)
    print('Cross val correlation: '+str(pearsonr(correlation_training,correlation_testing)[0]**2)+' '+str(pearsonr(correlation_training,correlation_testing)[1]))
    plt.scatter(np.asarray(correlation_training),np.asarray(correlation_testing))
    plt.show()
    print(np.median(correlation_training))
    print(iqr(correlation_training))
    print(np.median(correlation_testing))
    print(iqr(correlation_testing))    
    print(np.median(cross_corr))
    print(iqr(cross_corr))    
    print(np.median(correlation_with_all))
    print(iqr(correlation_with_all))




    mds_data = [0.9368,0.6845,0.5935,0.2046,0.2371,0.5662,0.6901,0.8612,0.5610,0.4604,0.3068,0.7646,0.3152,0.6535,0.7005,0.7070,0.3616]
    plt.scatter(np.asarray(mds_data),np.asarray(correlation_testing))
    plt.show() 
    print(pearsonr(mds_data,correlation_testing))

    return sigmas



###############################
##### All Datasets (128x22x11)

# CAS 1:
# sigmasPath = './out_aud_STRF_avgTime/'
# tabCorr = []
# ind = 1;
# for r, d, f in os.walk(sigmasPath):
#  for file in sorted(f):
#   if 'resultsOptims' in file: # 
#    sigmas = pickle.load(open(os.path.join(sigmasPath, file), 'rb'))
#    sigmas_tab.append(sigmas['sigmas'].flatten())
#    corr_value_explained_variance = math.pow(corr(sigmas['representations'],sigmas['sigmas'],sigmas['dissimilarities']),2)
#    tabCorr.append(corr_value_explained_variance)
#    ind+=1
# plt.show()



# CAS 2:
folder = './results-light/'
folder_old = './oldResults/out_aud_STRF_avgTime/'
tabCorr = []
ind = 1;
sigmas = get_crossval_sigmas_from_folder(folder=folder, averaging='avg_time', folder_old=folder_old)
for ri in range(len(sigmas)): # 
   # sigmas = pickle.load(open(os.path.join(sigmasPath, file), 'rb'))
   sigmas_tab.append(sigmas[ri].flatten())
   # corr_value_explained_variance = math.pow(corr(sigmas['representations'],sigmas['sigmas'],sigmas['dissimilarities']),2)
   # tabCorr.append(corr_value_explained_variance)
   ind+=1
plt.show()


# compute pairwise correlations => generalizability
num_datasets = len(sigmas_tab)
pairwisePearsonMatrixFullTensor = np.zeros((num_datasets,num_datasets))
pairwisePearsonMatrixScaleRate  = np.zeros((num_datasets,num_datasets))
pairwisePearsonMatrixFreqScale  = np.zeros((num_datasets,num_datasets))
pairwisePearsonMatrixFreqRate   = np.zeros((num_datasets,num_datasets))

for i_dataset in range(num_datasets):
    for j_dataset in range(num_datasets):
        strfTensorI = np.reshape(sigmas_tab[i_dataset],(128,11,22))
        strf_scale_rateI, strf_freq_rateI, strf_freq_scaleI = avgvec2strfavg(strf2avgvec(strfTensorI))
        strfTensorJ = np.reshape(sigmas_tab[j_dataset],(128,11,22))
        strf_scale_rateJ, strf_freq_rateJ, strf_freq_scaleJ = avgvec2strfavg(strf2avgvec(strfTensorJ))

        pairwisePearsonMatrixFullTensor[i_dataset][j_dataset]  = pearsonr(strfTensorI.flatten(),strfTensorJ.flatten())[0]**2
        pairwisePearsonMatrixScaleRate[i_dataset][j_dataset]   = pearsonr(strf_scale_rateI.flatten(),strf_scale_rateJ.flatten())[0]**2
        pairwisePearsonMatrixFreqRate[i_dataset][j_dataset]    = pearsonr(strf_freq_rateI.flatten(),strf_freq_rateJ.flatten())[0]**2
        pairwisePearsonMatrixFreqScale[i_dataset][j_dataset]   = pearsonr(strf_freq_scaleI.flatten(),strf_freq_scaleJ.flatten())[0]**2

print()
print('##################################')
print('All Datasets (128x22x11)')
print('Mutual Median pairwise correlations')
print('Full STRF - M='+str(np.median(pairwisePearsonMatrixFullTensor[np.triu_indices(num_datasets,1)])) + ' SD='+str(np.std(pairwisePearsonMatrixFullTensor[np.triu_indices(num_datasets,1)])))
print('Scale/Rate - M='+str(np.median(pairwisePearsonMatrixScaleRate[np.triu_indices(num_datasets,1)]))  + ' SD='+str(np.std(pairwisePearsonMatrixScaleRate[np.triu_indices(num_datasets,1)])))
print('Freq/Rate - M='+str(np.median(pairwisePearsonMatrixFreqRate[np.triu_indices(num_datasets,1)]))   + ' SD='+str(np.std(pairwisePearsonMatrixFreqRate[np.triu_indices(num_datasets,1)])))
print('Freq/Scale - M='+str(np.median(pairwisePearsonMatrixFreqScale[np.triu_indices(num_datasets,1)]))  + ' SD='+str(np.std(pairwisePearsonMatrixFreqScale[np.triu_indices(num_datasets,1)])))

print()
print('Mutual Mean pairwise correlations')
print('Full STRF - M='+str(np.mean(pairwisePearsonMatrixFullTensor[np.triu_indices(num_datasets,1)])) + ' SD='+str(np.std(pairwisePearsonMatrixFullTensor[np.triu_indices(num_datasets,1)])))
print('Scale/Rate - M='+str(np.mean(pairwisePearsonMatrixScaleRate[np.triu_indices(num_datasets,1)]))  + ' SD='+str(np.std(pairwisePearsonMatrixScaleRate[np.triu_indices(num_datasets,1)])))
print('Freq/Rate - M='+str(np.mean(pairwisePearsonMatrixFreqRate[np.triu_indices(num_datasets,1)]))   + ' SD='+str(np.std(pairwisePearsonMatrixFreqRate[np.triu_indices(num_datasets,1)])))
print('Freq/Scale - M='+str(np.mean(pairwisePearsonMatrixFreqScale[np.triu_indices(num_datasets,1)]))  + ' SD='+str(np.std(pairwisePearsonMatrixFreqScale[np.triu_indices(num_datasets,1)])))

###############################
##### All Datasets (128x22x11) Eb=311Hz
sigmasPath = './out_aud_STRF_avgTime/'
tabCorr = []
ind = 1;
for r, d, f in os.walk(sigmasPath):
 for file in sorted(f):
  if 'resultsOptims' and '311' in file: # 
   print(file)
   sigmas = pickle.load(open(os.path.join(sigmasPath, file), 'rb'))
   sigmas_tab.append(sigmas['sigmas'].flatten())
   corr_value_explained_variance = math.pow(corr(sigmas['representations'],sigmas['sigmas'],sigmas['dissimilarities']),2)
   tabCorr.append(corr_value_explained_variance)
   ind+=1
plt.show()


# compute pairwise correlations => generalizability
num_datasets = sigmas_tab.__len__()
pairwisePearsonMatrixFullTensor = np.zeros((num_datasets,num_datasets))
pairwisePearsonMatrixScaleRate  = np.zeros((num_datasets,num_datasets))
pairwisePearsonMatrixFreqScale  = np.zeros((num_datasets,num_datasets))
pairwisePearsonMatrixFreqRate   = np.zeros((num_datasets,num_datasets))

for i_dataset in range(num_datasets):
    for j_dataset in range(num_datasets):
        strfTensorI = np.reshape(sigmas_tab[i_dataset],(128,11,22))
        strf_scale_rateI, strf_freq_rateI, strf_freq_scaleI = avgvec2strfavg(strf2avgvec(strfTensorI))
        strfTensorJ = np.reshape(sigmas_tab[j_dataset],(128,11,22))
        strf_scale_rateJ, strf_freq_rateJ, strf_freq_scaleJ = avgvec2strfavg(strf2avgvec(strfTensorJ))

        pairwisePearsonMatrixFullTensor[i_dataset][j_dataset]  = pearsonr(strfTensorI.flatten(),strfTensorJ.flatten())[0]**2
        pairwisePearsonMatrixScaleRate[i_dataset][j_dataset]   = pearsonr(strf_scale_rateI.flatten(),strf_scale_rateJ.flatten())[0]**2
        pairwisePearsonMatrixFreqRate[i_dataset][j_dataset]    = pearsonr(strf_freq_rateI.flatten(),strf_freq_rateJ.flatten())[0]**2
        pairwisePearsonMatrixFreqScale[i_dataset][j_dataset]   = pearsonr(strf_freq_scaleI.flatten(),strf_freq_scaleJ.flatten())[0]**2

print()
print('##################################')
print('All Datasets (128x22x11) - Eb=311Hz')
print('Mutual Median pairwise correlations')
print('Full STRF - M='+str(np.median(pairwisePearsonMatrixFullTensor[np.triu_indices(num_datasets,1)])) + ' SD='+str(np.std(pairwisePearsonMatrixFullTensor[np.triu_indices(num_datasets,1)])))
print('Scale/Rate - M='+str(np.median(pairwisePearsonMatrixScaleRate[np.triu_indices(num_datasets,1)]))  + ' SD='+str(np.std(pairwisePearsonMatrixScaleRate[np.triu_indices(num_datasets,1)])))
print('Freq/Rate - M='+str(np.median(pairwisePearsonMatrixFreqRate[np.triu_indices(num_datasets,1)]))   + ' SD='+str(np.std(pairwisePearsonMatrixFreqRate[np.triu_indices(num_datasets,1)])))
print('Freq/Scale - M='+str(np.median(pairwisePearsonMatrixFreqScale[np.triu_indices(num_datasets,1)]))  + ' SD='+str(np.std(pairwisePearsonMatrixFreqScale[np.triu_indices(num_datasets,1)])))

print()
print('Mutual Mean pairwise correlations')
print('Full STRF - M='+str(np.mean(pairwisePearsonMatrixFullTensor[np.triu_indices(num_datasets,1)])) + ' SD='+str(np.std(pairwisePearsonMatrixFullTensor[np.triu_indices(num_datasets,1)])))
print('Scale/Rate - M='+str(np.mean(pairwisePearsonMatrixScaleRate[np.triu_indices(num_datasets,1)]))  + ' SD='+str(np.std(pairwisePearsonMatrixScaleRate[np.triu_indices(num_datasets,1)])))
print('Freq/Rate - M='+str(np.mean(pairwisePearsonMatrixFreqRate[np.triu_indices(num_datasets,1)]))   + ' SD='+str(np.std(pairwisePearsonMatrixFreqRate[np.triu_indices(num_datasets,1)])))
print('Freq/Scale - M='+str(np.mean(pairwisePearsonMatrixFreqScale[np.triu_indices(num_datasets,1)]))  + ' SD='+str(np.std(pairwisePearsonMatrixFreqScale[np.triu_indices(num_datasets,1)])))


###############################
##### All Datasets Decimated (11x11x11)

# compute pairwise correlations => generalizability
num_datasets = sigmas_tab.__len__()
pairwisePearsonMatrix = np.zeros((num_datasets,num_datasets))
pairwiseSpearmanMatrix = np.zeros((num_datasets,num_datasets))

for i_dataset in range(num_datasets):
    for j_dataset in range(num_datasets):
        strfTensorI = np.reshape(sigmas_tab[i_dataset],(128,11,22))
        strf_scale_rateI = np.mean(strfTensorI[0:128:12,0:11,0:22:2],axis=0)
        strf_freq_rateI  = np.mean(strfTensorI[0:128:12,0:11,0:22:2],axis=1)
        strf_freq_scaleI = np.mean(strfTensorI[0:128:12,0:11,0:22:2],axis=2)
        # strf_scale_rateI, strf_freq_rateI, strf_freq_scaleI = avgvec2strfavg(strf2avgvec(strfTensorI))
        strfTensorJ = np.reshape(sigmas_tab[j_dataset],(128,11,22))
        strf_scale_rateJ = np.mean(strfTensorJ[0:128:12,0:11,0:22:2],axis=0)
        strf_freq_rateJ  = np.mean(strfTensorJ[0:128:12,0:11,0:22:2],axis=1)
        strf_freq_scaleJ = np.mean(strfTensorJ[0:128:12,0:11,0:22:2],axis=2)        
        # strf_scale_rateJ, strf_freq_rateJ, strf_freq_scaleJ = avgvec2strfavg(strf2avgvec(strfTensorJ))

        pairwisePearsonMatrixFullTensor[i_dataset][j_dataset]  = pearsonr(strfTensorI.flatten(),strfTensorJ.flatten())[0]**2
        pairwisePearsonMatrixScaleRate[i_dataset][j_dataset]   = pearsonr(strf_scale_rateI.flatten(),strf_scale_rateJ.flatten())[0]**2
        pairwisePearsonMatrixFreqRate[i_dataset][j_dataset]    = pearsonr(strf_freq_rateI.flatten(),strf_freq_rateJ.flatten())[0]**2
        pairwisePearsonMatrixFreqScale[i_dataset][j_dataset]   = pearsonr(strf_freq_scaleI.flatten(),strf_freq_scaleJ.flatten())[0]**2

print()
print('##################################')
print('All Datasets (11x11x11)')
print('Mutual Median pairwise correlations')
print('Full STRF - M='+str(np.median(pairwisePearsonMatrixFullTensor[np.triu_indices(num_datasets,1)])) + ' SD='+str(np.std(pairwisePearsonMatrixFullTensor[np.triu_indices(num_datasets,1)])))
print('Scale/Rate - M='+str(np.median(pairwisePearsonMatrixScaleRate[np.triu_indices(num_datasets,1)]))  + ' SD='+str(np.std(pairwisePearsonMatrixScaleRate[np.triu_indices(num_datasets,1)])))
print('Freq/Rate - M='+str(np.median(pairwisePearsonMatrixFreqRate[np.triu_indices(num_datasets,1)]))   + ' SD='+str(np.std(pairwisePearsonMatrixFreqRate[np.triu_indices(num_datasets,1)])))
print('Freq/Scale - M='+str(np.median(pairwisePearsonMatrixFreqScale[np.triu_indices(num_datasets,1)]))  + ' SD='+str(np.std(pairwisePearsonMatrixFreqScale[np.triu_indices(num_datasets,1)])))

print()
print('Mutual Mean pairwise correlations')
print('Full STRF - M='+str(np.mean(pairwisePearsonMatrixFullTensor[np.triu_indices(num_datasets,1)])) + ' SD='+str(np.std(pairwisePearsonMatrixFullTensor[np.triu_indices(num_datasets,1)])))
print('Scale/Rate - M='+str(np.mean(pairwisePearsonMatrixScaleRate[np.triu_indices(num_datasets,1)]))  + ' SD='+str(np.std(pairwisePearsonMatrixScaleRate[np.triu_indices(num_datasets,1)])))
print('Freq/Rate - M='+str(np.mean(pairwisePearsonMatrixFreqRate[np.triu_indices(num_datasets,1)]))   + ' SD='+str(np.std(pairwisePearsonMatrixFreqRate[np.triu_indices(num_datasets,1)])))
print('Freq/Scale - M='+str(np.mean(pairwisePearsonMatrixFreqScale[np.triu_indices(num_datasets,1)]))  + ' SD='+str(np.std(pairwisePearsonMatrixFreqScale[np.triu_indices(num_datasets,1)])))
