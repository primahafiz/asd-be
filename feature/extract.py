import numpy as np
from .gammatone import calcGammatone
from .logmel import calcLogMel
from .utils import downSampleSpectrogram
from constant import *
from librosa import load

def extractGammatoneFromArray(x_t:np.ndarray):
    gammatoneSpectrogram = calcGammatone(x_t)
    resultGammatoneSpectrogram = downSampleSpectrogram(gammatoneSpectrogram,SCALING_GAMMATONE)

    return resultGammatoneSpectrogram

def extractGammatoneFromPath(pathSound:str):
    sound,_ = load(pathSound,sr=SAMPLE_RATE)

    return extractGammatoneFromArray(sound)

def extractLogMelFromArray(x_t:np.ndarray):
    logMelSpectrogram = calcLogMel(x_t)
    resultLogMelSpectrogram = downSampleSpectrogram(logMelSpectrogram)

    return resultLogMelSpectrogram

def extractLogMelFromPath(pathSound:str):
    sound,_ = load(pathSound,sr=SAMPLE_RATE)

    return extractLogMelFromArray(sound)