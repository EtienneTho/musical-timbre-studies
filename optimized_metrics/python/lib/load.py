# Copyright (c) Baptiste Caramiaux, Etienne Thoret
# All rights reserved

import os
import pickle
import numpy as np
from lib import utils
from lib import auditory
from lib import fourier
import subprocess


def database(root_path='../../ext/python/'):
    timbrespace_db = {}
    for root, dirs, files in os.walk(os.path.join(root_path, 'sounds')):
        for name in dirs:
            timbrespace_db[name] = {}
            timbrespace_db[name]['path'] = os.path.join(root, name)
    return timbrespace_db


def get_representation_func(representation):
    repres_dict = {
        'fourier_strf': fourier.strf,
        'fourier_mps': fourier.mps,
        'fourier_spectrogram': fourier.spectrogram,
        'fourier_spectrum': fourier.spectrum,
        'auditory_strf': auditory.strf,
        'auditory_mps': auditory.mps,
        'auditory_spectrogram': auditory.spectrogram,
        'auditory_spectrum': auditory.spectrum
    }
    return repres_dict[representation]

def timbrespace_features_low_storage(timbrespace='Iverson93Whole',
                         representations=['auditory_strf'],
                         audio_args={},
                         verbose = True):
    if (verbose):
        print('* get {} for {}'.format(representations,timbrespace))

    # if the database with the timbrespaces' names is not loaded yet, load it
    # if (timbrespace_db == None):
    timbrespace_db = database()

    # check if the timbrespace name is actually in the db
    valid_timbrespace_name = timbrespace in timbrespace_db.keys()
    if not valid_timbrespace_name:
        raise ValueError('{} is a wrong timbre space name'.format(timbrespace))

    # if the set of strfs for that timbrespace has not been computed and stored yet, do it
    # otherwise load it (the set of strfs is stored in a 'pickle' file)
    # load strf parameters
    # strf_params = utils.load_strf_params()
    # in case there is more than one type of representation space to be computed,
    # build a dictionnary with one entry for each space
    timbrespace_features = {}
    for rs in representations:
        timbrespace_features[rs] = []

    # load each sound in this timbrespace and compute the STRF of that sound
    filename_dict = {}
    audio_lengths = []
    for rs in representations:
        filename_dict[rs] = []
        for root, dirs, files in os.walk(timbrespace_db[timbrespace]['path']):
            for name in files:
                if name.split('.')[-1] in ['aiff', 'wav']:
                    filename_dict[rs].append(os.path.join(root, name))
                    audio, fs = utils.audio_data(os.path.join(root, name))
                    audio_lengths.append(len(audio))
    min_aud_len = np.min(audio_lengths)
    for rs in representations:
        for fn in sorted(filename_dict[rs]):
            # if (verbose):
            #     print('  |_ {}'.format(fn))
            audio, fs = utils.audio_data(fn)
            # audio = audio[:min_aud_len]
            print(fs)
            repres = get_representation_func(rs)(audio, fs, **audio_args)
            timbrespace_features[rs].append(np.mean(np.abs(repres), axis=0))

    if (verbose):
        print('  |_ num. of sounds: {}'.format(
            len(timbrespace_features[representations[0]])))
        print('  |_ spaces: {}'.format(list(timbrespace_features.keys())))
    return timbrespace_features


def timbrespace_features(timbrespace='Iverson93Whole',
                         representations=['auditory_strf'],
                         audio_args={},
                         verbose = True):
    if (verbose):
        print('* get {} for {}'.format(representations,timbrespace))

    # if the database with the timbrespaces' names is not loaded yet, load it
    # if (timbrespace_db == None):
    timbrespace_db = database()

    # check if the timbrespace name is actually in the db
    valid_timbrespace_name = timbrespace in timbrespace_db.keys()
    if not valid_timbrespace_name:
        raise ValueError('{} is a wrong timbre space name'.format(timbrespace))

    # if the set of strfs for that timbrespace has not been computed and stored yet, do it
    # otherwise load it (the set of strfs is stored in a 'pickle' file)
    # load strf parameters
    # strf_params = utils.load_strf_params()
    # in case there is more than one type of representation space to be computed,
    # build a dictionnary with one entry for each space
    timbrespace_features = {}
    for rs in representations:
        timbrespace_features[rs] = []

    # load each sound in this timbrespace and compute the STRF of that sound
    filename_dict = {}
    audio_lengths = []
    for rs in representations:
        filename_dict[rs] = []
        for root, dirs, files in os.walk(timbrespace_db[timbrespace]['path']):
            for name in files:
                if name.split('.')[-1] in ['aiff', 'wav']:
                    filename_dict[rs].append(os.path.join(root, name))
                    audio, fs = utils.audio_data(os.path.join(root, name))
                    audio_lengths.append(len(audio))
    min_aud_len = np.min(audio_lengths)
    for rs in representations:
        for fn in sorted(filename_dict[rs]):
            # if (verbose):
            #     print('  |_ {}'.format(fn))
            audio, fs = utils.audio_data(fn)
            # audio = audio[:min_aud_len]
            print(fs)
            repres = get_representation_func(rs)(audio, fs, **audio_args)
            print(repres.shape)
            timbrespace_features[rs].append(repres)

    if (verbose):
        print('  |_ num. of sounds: {}'.format(
            len(timbrespace_features[representations[0]])))
        print('  |_ spaces: {}'.format(list(timbrespace_features.keys())))
    return timbrespace_features


def load_timbrespace_windowed_features(timbrespace,
                                       representations=['strf'],
                                       win_length=1.0,
                                       hop_length=0.1,
                                       timbrespace_db=None,
                                       verbose=True):
    if (verbose):
        print('* get STRF for {}'.format(timbrespace))

    # if the database with the timbrespaces' names is not loaded yet, load it
    if (timbrespace_db == None):
        timbrespace_db = database()

    # check if the timbrespace name is actually in the db
    valid_timbrespace_name = timbrespace in timbrespace_db.keys()
    if not valid_timbrespace_name:
        raise ValueError('{} is a wrong timbre space name'.format(timbrespace))

    # load strf parameters
    strf_params = utils.load_strf_params()
    # in case there is more than one type of representation space to be computed,
    # build a dictionnary with one entry for each space
    timbrespace_features = {}
    for rs in representations:
        timbrespace_features[rs] = []

    for rs in representations:
        for root, dirs, files in os.walk(timbrespace_db[timbrespace]['path']):
            for name in files:
                if name.split('.')[-1] in ['aiff', 'wav']:
                    audio, fs = utils.audio_data(os.path.join(root, name))
                    strf_params.update({'fs': fs})
                    # windowing parameters
                    win_length_n = int(win_length * fs)
                    hop_length_n = int(hop_length * fs)
                    num_frames = int(
                        (len(audio) - win_length_n) / hop_length_n)
                    windowed_features = []
                    for wn in range(num_frames):
                        start_n = wn * win_length_n
                        end_n = wn * win_length_n + hop_length_n
                        if (rs == 'strf'):
                            windowed_features.append(
                                features.strf(audio[start_n:end_n], **
                                              strf_params))
                    timbrespace_features[rs].append(windowed_features)

    if (verbose):
        print('  |_ num. of sounds: {}'.format(
            len(timbrespace_features[representations[0]])))
        print('  |_ spaces: {}'.format(list(timbrespace_features.keys())))
    return timbrespace_features


def timbrespace_dismatrix(timbrespace, timbrespace_db=None, verbose=False):
    if (verbose): print('* get dissimiarity matrix for {}'.format(timbrespace))
    if (timbrespace_db == None):
        timbrespace_db = database()
    valid_timbrespace_name = timbrespace in timbrespace_db.keys()
    if not valid_timbrespace_name:
        raise ValueError('{} is a wrong timbre space name'.format(timbrespace))
    root_path = os.path.join(
        *timbrespace_db[timbrespace]['path'].split('/')[:-2])
    if os.path.isfile(
            os.path.join(root_path, 'data', timbrespace +
                         '_dissimilarity_matrix.txt')):
        return np.loadtxt(
            os.path.join(root_path, 'data', timbrespace +
                         '_dissimilarity_matrix.txt'))


def load_dismatrices(root_path='../dataSoundsDissim/'):
    timbrespace_names = load_timbrespace_names(root_path)
    dismatrices = []
    for i, el in enumerate(timbrespace_names):
        if os.path.isfile(
                os.path.join(root_path, 'data', el['name'] +
                             '_dissimilarity_matrix.txt')):
            dismatrices.append({
                'name': el['name'], \
                'matrix': np.loadtxt(
                            os.path.join(root_path, 'data', el['name'] +
                                 '_dissimilarity_matrix.txt'))
            })
    print(len(dismatrices), dismatrices[0]['name'], dismatrices[0]['matrix'])


def delete_outpkls(path='/Volumes/LaCie/outputs'):
    timbrespace_db = [
        'grey1977',
        'grey1978',
        'iverson1993onset',
        'iverson1993whole',
        'lakatoscomb',
        'lakatosharm',
        'lakatosperc',
        'mcadams1995',
        'patil2012_a3',
        'patil2012_dx4',
        'patil2012_gd4'
    ]
    for i, tsp in enumerate(sorted(timbrespace_db)):
        folder = os.path.join(path,tsp.lower())
        for root, dirs, files in os.walk(folder):
            for f in files:
                # print(root)
                if f.split('=')[0] == 'optim_process_l':
                    if int(f.split('=')[1].split('.')[0])%20==0:
                        print('keep', f, root)
                    else:
                        print('delete', os.path.join(root,f))
                        subprocess.call(['rm', '-r', os.path.join(root,f)])

if __name__ == "__main__":
    # load_dismatrices()
    # print(load_timbrespace_names())
    delete_outpkls(path='/Volumes/LaCie/outputs')