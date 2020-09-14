### Thoret, Caramiaux (c) 2020
###
### This script optimizes the weigths of a Gaussian Kernel on acoustic representation in order to fit with the human dissimilarity ratings
### The results are logged in the "log_foldername" folder (line 268), must be changed for each optimization to not erase former results
### The representation to consider is defined by rs (line 305) and can take the following values 'auditory_spectrum', 
### 'fourier_spectrum', 'auditory_strf', 'fourier_strf', 'auditory_spectrogram', 'fourier_spectrogram', 'auditory_mps', 'fourier_mps'
###
### and by "reduce_strf" value (line 269) for the averaging of the STMF representation and can take the following values: 
### 'avg_time': averaged over time, 'avg_time_avg_freq': averaged over time and frequency (scale-rate),
### 'avg_time_avg_scale': averaged over time and frequency (freq-rate), 'avg_time_avg_rate': averaged over time and frequency (freq-scale),
###
### This script replicates the results of the section "Optimized metrics simulating human dissimilarity ratings" for the Full STMF, scale-rate, 
### freq-rate, freq-scale representations
###

import subprocess
import numpy as np
import os
import time
import pickle
import matplotlib.pylab as plt
from lib import pca
from lib import load
from lib import training
import random
from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr

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

def run_one_tsp_rs(tsp, rs, args):
    rslog_foldername = args['optim_args']['log_foldername'] # logfolder
    rs = rslog_foldername.split('/')[-1].split('-')[0] 
    dissimil_mat = load.timbrespace_dismatrix(tsp, load.database()) # load dissimilarity matrix
    aud_repres = load.timbrespace_features(
        tsp, representations=[rs], audio_args=args['audio_args'])[rs] # load representations    
    tab_red = []
    rs_type = rs.split('_')[-1]
    mapping = []
    variances = []
    reduce_strf = args['reduce_strf']
    if rs_type == 'strf': # parse the type of input representation
        if reduce_strf == 'avg_time':
            for i in range(len(aud_repres)):
                strf_avgTime = np.mean(np.abs(aud_repres[i]), axis=0)                
                tab_red.append(strf_avgTime.flatten())
            tab_red = np.transpose(np.asarray(tab_red))
        elif reduce_strf == 'avg_time_avg_scale_rate_freq':
            for i in range(len(aud_repres)):
                tab_red.append(strf2avgvec(aud_repres[i]).flatten())
            tab_red = np.transpose(np.asarray(tab_red))      
        elif reduce_strf == 'avg_time_avg_freq':
            for i in range(len(aud_repres)):
                strf_scale_rate, strf_freq_rate, strf_freq_scale = avgvec2strfavg(strf2avgvec(aud_repres[i]))
                tab_red.append(strf_scale_rate.flatten())                
            tab_red = np.transpose(np.asarray(tab_red))
        elif reduce_strf == 'avg_time_avg_scale':
            for i in range(len(aud_repres)):
                strf_scale_rate, strf_freq_rate, strf_freq_scale = avgvec2strfavg(strf2avgvec(aud_repres[i]))
                tab_red.append(strf_freq_rate.flatten())
            tab_red = np.transpose(np.asarray(tab_red))
        elif reduce_strf == 'avg_time_avg_rate':
            for i in range(len(aud_repres)):
                strf_scale_rate, strf_freq_rate, strf_freq_scale = avgvec2strfavg(strf2avgvec(aud_repres[i]))
                tab_red.append(strf_freq_scale.flatten())                
            tab_red = np.transpose(np.asarray(tab_red))                                           
        else:
            for i in range(len(aud_repres)):
                tab_red.append(aud_repres[i].flatten())
            tab_red = np.transpose(np.asarray(tab_red))

    print(args['audio_args'])
    print('  data dimension:', tab_red.shape)
    print('* normalizing')
    print(tab_red)
    print(np.mean(np.max(np.abs(tab_red), axis=0)).shape)                                                
    tab_red = tab_red / np.mean(np.max(np.abs(tab_red), axis=0))
    correlations, sigmas__ = training.kernel_optim_lbfgs_log(tab_red, dissimil_mat,
                                               **args['optim_args'])
    pickle.dump({'sigmas': sigmas__,
                 'representations': tab_red,
                 'dissimilarities': dissimil_mat,
                 'audio_args': args['audio_args']
    }, open(os.path.join(args['log_foldername']+'resultsOptims_'+rs_type+'_'+reduce_strf+'_'+tsp+'.pkl'), 'wb'))

def run_one_tsp_rs_crossval(tsp, rs, args):
    rslog_foldername = args['optim_args']['log_foldername']
    rs = rslog_foldername.split('/')[-1].split('-')[0]
    dissimil_mat = load.timbrespace_dismatrix(tsp, load.database())
    aud_repres = load.timbrespace_features(tsp, representations=[rs], audio_args=args['audio_args'])[rs]
    tab_red = []
    rs_type = rs.split('_')[-1]
    mapping = []
    variances = []
    reduce_strf = args['reduce_strf']
    if rs_type == 'strf':
        if reduce_strf == 'pca':
            n_components = 1
            for i in range(len(aud_repres)):
                strf_reduced, mapping_, variances = pca.pca_patil(
                    np.absolute(aud_repres[i]),
                    aud_repres[i].shape[1],
                    n_components=n_components)
                strf_reduced = strf_reduced.flatten()
                tab_red.append(strf_reduced)
                mapping.append(mapping_)
            tab_red = np.transpose(np.asarray(tab_red))
        elif reduce_strf == 'avg_time':
            for i in range(len(aud_repres)):
                tab_red.append(np.absolute(np.mean(aud_repres[i], axis=0)).flatten())
            tab_red = np.transpose(np.asarray(tab_red))
        else:
            for i in range(len(aud_repres)):
                tab_red.append(aud_repres[i].flatten())
            tab_red = np.transpose(np.asarray(tab_red))
    elif rs_type == 'spectrogram' or rs_type == 'mps':
        for i in range(len(aud_repres)):
            tab_red.append(aud_repres[i].flatten())
        tab_red = np.transpose(np.asarray(tab_red))
    elif rs_type == 'spectrum':
        for i in range(len(aud_repres)):
            tab_red.append(aud_repres[i])
        tab_red = np.transpose(np.asarray(tab_red))
    pickle.dump({
        'data_repres': aud_repres,
        'data_proj': tab_red,
        'mapping': mapping,
        'variances': variances,
        'dissimilarities': dissimil_mat,
        'audio_args': args['audio_args']
    }, open(os.path.join(rslog_foldername, 'dataset.pkl'), 'wb'))
    
    print('  data dimension:', tab_red.shape)
    print('* normalizing')
    tab_red = tab_red / np.mean(np.max(np.abs(tab_red), axis=0))
    ninstrus = tab_red.shape[1]
    correlations_training = []
    sigmas_training = []
    correlations_testing = []
    correlations_testing_pearson = []
    correlations_testing_spearman = []
    sigmas_testing = []
    distances_testing = []
    target_testing = []
    
    print('* cross-validation tests')
    org_fname = args['optim_args']['log_foldername']
    for fold in range(ninstrus):
        # training data
        train_idx = [i for i in range(tab_red.shape[1]) if i != fold]
        sorted_train_idx = sorted(train_idx)
        input_data_training = tab_red[:,train_idx]
        target_data_training = dissimil_mat[train_idx, :]
        target_data_training = target_data_training[:, train_idx]
        # testing data
        test_idx = [fold]
        input_data_testing = tab_red[:, test_idx[0]]
        target_data_testing = np.zeros((ninstrus - 1, 1))
        cpt_i = 0
        for i in range(ninstrus):
            if i > test_idx[0]:
                target_data_testing[cpt_i] = dissimil_mat[test_idx[0], i]
                cpt_i += 1
            elif i < test_idx[0]:
                target_data_testing[cpt_i] = dissimil_mat[i, test_idx[0]]
                cpt_i += 1
        target_testing.append(target_data_testing.reshape(1,-1))
        mean_target_test = np.mean(target_data_testing)
        std_target_test = np.std(target_data_testing)

        # optimisation on fold i
        print('* Fold {} - {} {}'.format(fold+1, train_idx, test_idx))
        args['optim_args']['allow_resume'] = False
        args['optim_args']['log_foldername'] = org_fname + '/fold{}'.format(fold+1)
        subprocess.call(['mkdir', '-p', args['optim_args']['log_foldername']])
        correlations, sigmas = training.kernel_optim_lbfgs_log(input_data=input_data_training, 
                                                               target_data=target_data_training, 
                                                               test_data=(input_data_testing, target_data_testing),
                                                               **args['optim_args'])
        correlations_training.append(correlations)
        sigmas_training.append(sigmas)
        
        distances = np.zeros((ninstrus - 1, 1))       
        for i in range(ninstrus-1):
            distances[i, 0] = -np.sum(np.power(np.divide(tab_red[:, test_idx[0]] - tab_red[:, sorted_train_idx[i]],
                (sigmas + np.finfo(float).eps)), 2))
        distances_testing.append(distances.reshape(1,-1))
        mean_distances = np.mean(distances)
        stddev_distances = np.std(distances)
        Jn = np.sum(np.multiply(distances - mean_distances, target_data_testing - mean_target_test))
        Jd = std_target_test * stddev_distances * (ninstrus-1)
        
        correlations_testing.append(Jn/Jd)
        sigmas_testing.append(sigmas)
        pearsr = pearsonr(distances, target_data_testing)
        correlations_testing_pearson.append(pearsr)
        spear_rho, spear_p_val  = spearmanr(distances, target_data_testing)
        spear = [spear_rho, spear_p_val]
        correlations_testing_spearman.append(spear)

        print('\tFold={} test_corr={:.3f} (PE {} | SP {}) (train_corr={:.3f})'.format(fold+1, Jn/Jd, pearsr, spear, correlations[-1]))
        pickle.dump({
            'correlations_training': correlations_training,
            'sigmas_training': sigmas_training,
            'correlations_testing': correlations_testing,
            'sigmas_testing': sigmas_testing,
            'correlations_testing_pearson': correlations_testing_pearson,
            'correlations_testing_spearman': correlations_testing_spearman,
            'distances_testing': distances_testing,
            'target_data_testing': target_testing
            }, open(os.path.join(rslog_foldername, 'crossval_intrats_res.pkl'), 'wb'))

    mean_correlations_training = [correlations_training[i][-1] for i in range(len(correlations_training))]
    mean_correlations_training = np.mean(mean_correlations_training)

    distances_testing = np.squeeze(np.array(distances_testing)).reshape(-1,1)
    target_testing = np.squeeze(np.array(target_testing)).reshape(-1,1)

    print('\tAll Folds: mean_training_corr={} | corr_testing_pears={} | corr_testing_spear={}'.format(
        mean_correlations_training, 
        pearsonr(distances_testing[:,0], target_testing[:,0]),
        spearmanr(distances_testing[:,0], target_testing[:,0])))


def run_optimization(args={}):
    orig_fn = args['log_foldername']
    for i, tsp in enumerate(args['timbre_spaces']):
        print('Processing', tsp)
        log_foldername = os.path.join(orig_fn, tsp.lower())
        subprocess.call(['mkdir', '-p', log_foldername])
        for rs in args['audio_representations']:
            rslog_foldername = log_foldername + '/' + rs + '-' + time.strftime(
                '%y%m%d@%H%M%S')
            subprocess.call(['mkdir', '-p', rslog_foldername])
            args['optim_args'].update({'log_foldername': rslog_foldername})
            if not args['test_args']['snd_crossval']:
                run_one_tsp_rs(tsp, rs, args)
            else:
                run_one_tsp_rs_crossval(tsp, rs, args)


def run(tsp=None, rs=None):
    valid_tsps = list(sorted(load.database().keys())) if tsp == None else tsp
    valid_rs = ['auditory_spectrum', 'fourier_spectrum', 'auditory_strf',
            'fourier_strf', 'auditory_spectrogram', 'fourier_spectrogram',
            'auditory_mps', 'fourier_mps'
        ] if rs == None else rs # valid representation name
    
    args = {
        'timbre_spaces': valid_tsps,
        'audio_representations': valid_rs,
        'log_foldername': './out_aud_STRF_avgTime/', # output folder
        'reduce_strf': 'avg_time',
        # type of averaging
        # avg_time : STRF averaged over time, 
        # avg_time_avg_scale_rate_freqavg_time_avg_freq : STRF averaged over time and freq => scale/rate projection
        # avg_time_avg_scale : STRF averaged over time and scale => freq/rate projection
        # avg_time_avg_rate : STRF averaged over time and rate => freq/scale projection
        'test_args': {
            'snd_crossval': False
        },
        'audio_args': {
            'resampling_fs': 16000, # resampling frequency
            'duration': -1, # duration of the cut in seconds, -1 : no cut
            'duration_cut_decay': 0.05 # duration of the fade out
        },
        'optim_args': {
            'cost': 'correlation', # type of cost function
            'loss': 'exp_sum', # loss type
            'method': 'L-BFGS-B', # optimization method
            'init_sig_mean': 1.0, # initial sigma mean
            'init_sig_var': 0.01, # initial sigma variance
            'num_loops': 300, # number of maximum iterations
            'logging': True
        },
    }    
    start_time = time.time()
    run_optimization(args)
    print(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

if __name__ == '__main__':
    
    run(tsp=['Barthet2010','Grey1977','Grey1978','Iverson1993_Onset','Iverson1993_Remainder','Iverson1993_Whole', 'McAdams1995','Patil2012_A3',
             'Patil2012_DX4','Patil2012_GD4','Siedenburg2016_e3', 'Lakatos2000_Harm','Lakatos2000_Comb','Lakatos2000_Perc','Siedenburg2016_e2set1'
             ,'Siedenburg2016_e2set2','Siedenburg2016_e2set3'],# run optimizations on the datasets listed in tsp
        rs=['auditory_strf']) # representations to be used