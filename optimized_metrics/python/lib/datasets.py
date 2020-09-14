# Copyright (c) Baptiste Caramiaux, Etienne Thoret
# All rights reserved

import os
import pickle
import numpy as np
import utils
import features


def load_timbrespace_database(root_path='../ext/'):
    timbrespace_db = {}
    for root, dirs, files in os.walk(os.path.join(root_path, 'sounds')):
        for name in dirs:
            timbrespace_db[name] = {}
            timbrespace_db[name]['path'] = os.path.join(root, name)
    return timbrespace_db


def load_timbrespace_features(timbrespace='Iverson93Whole',
                              representations=['strf'],
                              window=None,
                              timbrespace_db=None,
                              verbose=True):
    if (verbose):
        print('* get STRF for {}'.format(timbrespace))

    # if the database with the timbrespaces' names is not loaded yet, load it
    if (timbrespace_db == None):
        timbrespace_db = load_timbrespace_database()

    # check if the timbrespace name is actually in the db
    valid_timbrespace_name = timbrespace in timbrespace_db.keys()
    if not valid_timbrespace_name:
        raise ValueError('{} is a wrong timbre space name'.format(timbrespace))

    # if the set of strfs for that timbrespace has not been computed and stored yet, do it
    # otherwise load it (the set of strfs is stored in a 'pickle' file)
    # load strf parameters
    strf_params = utils.load_strf_params()
    # in case there is more than one type of representation space to be computed,
    # build a dictionnary with one entry for each space
    timbrespace_features = {}
    for rs in representations:
        timbrespace_features[rs] = []

    # load each sound in this timbrespace and compute the STRF of that sound
    for rs in representations:
        for root, dirs, files in os.walk(timbrespace_db[timbrespace]['path']):
            for name in files:
                if name.split('.')[-1] in ['aiff', 'wav']:
                    if (verbose):
                        print('  |_ {}'.format(name))
                    audio, fs = utils.audio_data(os.path.join(root, name))
                    strf_params.update({'fs': fs})
                    # compute each space in the list 'representations'
                    if (rs == 'strf'):
                        if window == None:
                            timbrespace_features[rs].append(
                                features.strf(audio, **strf_params))
                        else:
                            # windowing parameters
                            win_length_n = int(window['win_length'] * fs)
                            hop_length_n = int(window['hop_length'] * fs)
                            num_frames = max(
                                int((
                                    len(audio) - win_length_n) / hop_length_n),
                                1)
                            windowed_features = []
                            for wn in range(num_frames):
                                start_n = wn * hop_length_n
                                end_n = start_n + win_length_n
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
        timbrespace_db = load_timbrespace_database()

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


def load_timbrespace_dismatrix(timbrespace, timbrespace_db=None, verbose=True):
    if (verbose): print('* get dissimiarity matrix for {}'.format(timbrespace))
    if (timbrespace_db == None):
        timbrespace_db = load_timbrespace_database()
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


if __name__ == "__main__":
    load_dismatrices()
    # print(load_timbrespace_names())