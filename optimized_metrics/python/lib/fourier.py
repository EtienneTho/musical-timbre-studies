'''
Copyright (c) Baptiste Caramiaux, Etienne Thoret
All rights reserved

'''
import numpy as np
import math
from scipy import signal
from lib import utils
from lib import features  #import spectrum2scaletime, scaletime2scalerate, scalerate2cortical, waveform2auditoryspectrogram

# def load_params():
#     params = {}
#     # params['window_size'] = 743
#     # params['frame_step'] = 185
#     params['window_size'] = 256
#     params['frame_step'] = 128
#     params['duration'] = 0.25
#     params['duration_cut_decay'] = 0.05
#     params['resampling_fs'] = 16000
#     return params


def gaussianWdw2d(mu_x, sigma_x, mu_y, sigma_y, x, y):
    # %window = exp(-(x - mu_x).^2 / 2 / sigma_x / sigma_x) .* exp(-(y - mu_y).^2 / 2 / sigma_y / sigma_y) ;
    m1 = np.exp(-np.power((x - mu_x), 2) / 2 / sigma_x / sigma_x).reshape(
        -1, 1)
    m2 = np.exp(-np.power(y - mu_y, 2) / 2 / sigma_y / sigma_y).reshape(1, -1)
    window = np.dot(m1, m2)
    # print(window.shape)
    return window


def spectrogram(wavtemp,
                audio_fs=44100,
                window_size=1024,
                frame_step=128,
                duration=0.25,
                duration_cut_decay=0.05,
                resampling_fs=16000,
                offset=0):
    wavtemp = np.r_[wavtemp, np.zeros(resampling_fs)]
    # if wavtemp.shape[0] > math.floor(duration * audio_fs):
    #     wavtemp = wavtemp[:int(duration * audio_fs)]
    #     wavtemp[wavtemp.shape[0] - int(
    #         audio_fs * duration_cut_decay):] = wavtemp[wavtemp.shape[0] - int(
    #             audio_fs * duration_cut_decay):] * utils.raised_cosine(
    #                 np.arange(int(audio_fs * duration_cut_decay)), 0,
    #                 int(audio_fs * duration_cut_decay))
    if wavtemp.shape[0] > math.floor(duration * audio_fs):
        offset_n = int(offset * audio_fs)
        duration_n = int(duration * audio_fs)
        duration_decay_n = int(duration_cut_decay * audio_fs)
        wavtemp = wavtemp[offset_n:offset_n + duration_n]
        if offset_n==0:
            wavtemp[wavtemp.shape[0] - duration_decay_n:] = wavtemp[
                wavtemp.shape[0] - duration_decay_n:] * utils.raised_cosine(
                    np.arange(duration_decay_n), 0, duration_decay_n)
        else:
            wavtemp[wavtemp.shape[0] - duration_decay_n:] = wavtemp[
                wavtemp.shape[0] - duration_decay_n:] * utils.raised_cosine(
                    np.arange(duration_decay_n), 0, duration_decay_n)
            wavtemp[:duration_decay_n] = wavtemp[:duration_decay_n] * utils.raised_cosine(
                    np.arange(duration_decay_n), duration_decay_n, duration_decay_n)
    wavtemp = (wavtemp / 1.01) / (np.max(wavtemp) + np.finfo(float).eps)
    wavtemp = signal.resample(wavtemp,
                              int(wavtemp.shape[0] / audio_fs * resampling_fs))

    spectrogram_ = features.complexSpectrogram(wavtemp, window_size,
                                               frame_step)
    spectrogram_ = np.abs(spectrogram_[:int(spectrogram_.shape[0] / 2), :])
    return spectrogram_


def spectrum(wavtemp,
             audio_fs=44100,
             window_size=1024,
             frame_step=128,
             duration=0.25,
             duration_cut_decay=0.05,
             resampling_fs=16000,
             offset=0):
    spectrogram_ = spectrogram(wavtemp, audio_fs, window_size, frame_step,
                               duration, duration_cut_decay, resampling_fs, offset)
    spectrum_ = np.mean(spectrogram_, axis=1)
    return spectrum_


def mps(wavtemp,
        audio_fs=44100,
        window_size=256,
        frame_step=128,
        duration=0.25,
        duration_cut_decay=0.05,
        resampling_fs=16000,
        offset=0):
    spectrogram_ = np.transpose(
        spectrogram(wavtemp, audio_fs, window_size, frame_step, duration,
                    duration_cut_decay, resampling_fs, offset))
    N = spectrogram_.shape[0]
    M = spectrogram_.shape[1]
    N1 = 2**utils.nextpow2(spectrogram_.shape[0])
    N2 = 2 * N1
    M1 = 2**utils.nextpow2(spectrogram_.shape[1])
    M2 = 2 * M1
    Y = np.zeros((N2, M2), dtype=np.complex_)
    for n in range(N):
        R1 = np.fft.fft(spectrogram_[n, :], M2)
        Y[n, :] = R1[:M2]
    for m in range(M2):
        R1 = np.fft.fft(Y[:N, m], N2)
        Y[:, m] = R1
    mps_ = np.absolute(Y[:, :int(Y.shape[1] / 2)])
    return mps_


def strf(wavtemp,
         audio_fs=44100,
         window_size=256,
         frame_step=128,
         duration=0.25,
         duration_cut_decay=0.05,
         resampling_fs=16000,
         offset=0):
    spectrogram_ = np.transpose(
        spectrogram(wavtemp, audio_fs, window_size, frame_step, duration,
                    duration_cut_decay, resampling_fs, offset))
    N = spectrogram_.shape[0]
    M = spectrogram_.shape[1]
    N1 = 2**utils.nextpow2(spectrogram_.shape[0])
    N2 = 2 * N1
    M1 = 2**utils.nextpow2(spectrogram_.shape[1])
    M2 = 2 * M1
    Y = np.zeros((N2, M2), dtype=np.complex_)
    for n in range(N):
        R1 = np.fft.fft(spectrogram_[n, :], M2)
        Y[n, :] = R1[:M2]
    for m in range(M2):
        R1 = np.fft.fft(Y[:N, m], N2)
        Y[:, m] = R1
    mps_ = np.absolute(Y[:, :int(Y.shape[1] / 2)])
    maxRate = resampling_fs / frame_step / 2  #; % max rate values
    maxScale = window_size / (resampling_fs * 1e-3) / 2  #; % max scale value
    ratesVector = np.linspace(-maxRate + 5, maxRate - 5, num=22)
    deltaRates = ratesVector[1] - ratesVector[0]
    scalesVector = np.linspace(0, maxScale - 5, num=11)
    deltaScales = scalesVector[2] - scalesVector[1]

    overlapRate = .75
    overlapScale = .75
    stdRate = deltaRates / 2 * (overlapRate + 1)
    stdScale = deltaScales / 2 * (overlapScale + 1)

    maxRatePoints = int(len(mps_) / 2)
    maxScalePoints = mps_.shape[1]
    stdRatePoints = maxRatePoints * stdRate / maxRate
    stdScalePoints = maxScalePoints * stdScale / maxScale
    # strf_ = np.zeros((N, M, len(ratesVector), len(scalesVector)))
    strf_ = np.zeros((N, M, len(scalesVector), len(ratesVector)))
    for iRate in range(len(ratesVector)):
        rateCenter = ratesVector[iRate]
        # %rate center in point
        if rateCenter <= 0:
            rateCenterPoint = maxRatePoints * (
                2 - np.abs(rateCenter) / maxRate)
        else:
            rateCenterPoint = maxRatePoints * np.abs(rateCenter) / maxRate

        for iScale in range(len(scalesVector)):
            scaleCenter = scalesVector[iScale]
            # %scale center in point
            scaleCenterPoint = maxScalePoints * np.abs(scaleCenter) / maxScale
            filterPoint = gaussianWdw2d(rateCenterPoint, stdRatePoints,
                                        scaleCenterPoint, stdScalePoints,
                                        np.linspace(
                                            1,
                                            2 * maxRatePoints,
                                            num=2 * maxRatePoints),
                                        np.linspace(
                                            1,
                                            maxScalePoints,
                                            num=maxScalePoints))
            MPS_filtered = mps_ * filterPoint
            MPS_quadrantPoint = np.c_[MPS_filtered, np.fliplr(MPS_filtered)]
            stftRec = np.fft.ifft(np.transpose(np.fft.ifft(MPS_quadrantPoint)))
            ll = len(stftRec)
            stftRec = np.transpose(
                np.r_[stftRec[:M, :N], stftRec[ll - M:ll, :N]])
            # !! taking real values
            strf_[:, :, iScale, iRate] = np.abs(
                stftRec[:, :int(stftRec.shape[1] / 2)])

    return strf_
