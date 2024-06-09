import numpy as np
from librosa.feature import melspectrogram
from librosa import power_to_db
from constant import *

def calcLogMel(x_t: np.ndarray):
    melSpectrogram = melspectrogram(
        y=x_t, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=NUM_BANDS
    )
    logMelSpectrogram = power_to_db(melSpectrogram)

    return logMelSpectrogram