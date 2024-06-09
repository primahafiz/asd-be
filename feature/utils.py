import numpy as np
from constant import *
import matplotlib.pyplot as plt
import copy
import time
from PIL import Image
import io
from base64 import b64encode

def downSampleSpectrogram(filterResult:np.ndarray, scaling:int = 1):

    NUM_PER_FRAME = filterResult.shape[1] // TOTAL_FRAMES
    downSampledFeature = np.array([[0 for j in range(TOTAL_FRAMES)] for i in range(filterResult.shape[0])])
    
    for i in range(filterResult.shape[0]):
        for j in range(TOTAL_FRAMES):
            downSampledFeature[i][j] = scaling * np.mean(filterResult[i,j*NUM_PER_FRAME:(j+1)*NUM_PER_FRAME])

    downSampledFeature = downSampledFeature.transpose()
    
    dimCol = FRAME_SIZE
    divi = FRAME_SIZE // downSampledFeature.shape[1]
    dimRow = downSampledFeature.shape[0]//divi
    
    featuresGammatone = np.array([[0 for j in range(dimCol)] for i in range(dimRow)],dtype='float32')
    
    for i in range(len(downSampledFeature)):
        for j in range(len(downSampledFeature[0])):
            num = (i * len(downSampledFeature[0])) + j
            row = num // dimCol
            col =  num % dimCol
            if row >= dimRow:
                continue
            featuresGammatone[row][col] = downSampledFeature[i][j]

    
    return featuresGammatone

def convertSpectrogramToOriginalForm(spectrogram:np.ndarray):
    resSpectrogram = np.array([[0 for _ in range(NUM_BANDS)] for _ in range(TOTAL_FRAMES)],dtype='float32')
    frameForEachRow = FRAME_SIZE // NUM_BANDS

    for i in range(TOTAL_FRAMES):
        row = i // frameForEachRow
        col = i % frameForEachRow
        resSpectrogram[:][i] = spectrogram[row][col*NUM_BANDS:col*NUM_BANDS+NUM_BANDS]

    return resSpectrogram

def convertSpectrogramToVisualizedForm(spectrogram:np.ndarray):
    resSpectrogram = convertSpectrogramToOriginalForm(spectrogram)

    return np.flipud(resSpectrogram.transpose())

def visualizeSpectrogram(spectrogram:np.ndarray):
    _, ax = plt.subplots(1, 1, tight_layout=True)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    ax.imshow(spectrogram,interpolation='none', cmap='magma')

def getImageSpectrogram(spectrogram:np.ndarray):
    _, ax = plt.subplots(1,1,tight_layout=True)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    ax.imshow(spectrogram,interpolation='none', cmap='magma')

    newPath = PATH_TMP + str(int(time.time())) + 'spectrogram.png'
    plt.savefig(newPath, bbox_inches='tight')

    bytesArr = []
    with open(newPath, 'rb') as f:
        bytesArr = f.read()

    return b64encode(bytesArr)

def expandSpectrogram(spectrogram:np.ndarray):
    res = copy.deepcopy(spectrogram.tolist())
    for i in range(1,5):
        for j in range(len(spectrogram)-1):
            res.append(np.concatenate((spectrogram[j,i*64:],spectrogram[j+1,:i*64])).tolist())
    return np.array(res)

def getFeatureType(val:str):
    for f in FeatureType:
        if f.value == val:
            return f
        
def getMachineType(val:str):
    for m in MachineType:
        if m.value == val:
            return m