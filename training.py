import random
import math
import os
from os.path import isfile,join
import yaml
import copy
from constant import *
from feature import *
from math import radians
import torch
from torch import nn
import random
from torch.nn.modules.fold import Unfold
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

from train import *

# Init constants
yamlConfig = None
with open('train.yaml','r') as stream:
    yamlConfig = yaml.safe_load(stream)
PATH_INPUT_SOUND_NORMAL = yamlConfig['data']['normal']
PATH_INPUT_SOUND_ABNORMAL = yamlConfig['data']['abnormal']
SEED = yamlConfig['general']['seed']
VERBOSE_PERIOD = yamlConfig['general']['verbose_period_extract']

# Dataset Class
class Datasets(Dataset):
    def __init__(self, data, transform=None):
        self.data = data

    def __getitem__(self, index):
        return transforms.ToTensor(self.data[index])

    def __len__(self):
        return len(self.data)


def extractListPathWithNoOverlap(pathSounds:list[str], featureType: str, verboseTypeData: str):
    res = []
    for i in range(len(pathSounds)):
        pathSound = pathSounds[i]
        currentSoundData = None
        if featureType == FeatureType.LOGMEL.value:
            currentSoundData = extractLogMelFromPath(pathSound)
        elif featureType == FeatureType.GAMMATONE.value:
            currentSoundData = extractGammatoneFromPath(pathSound)
        cur = copy.deepcopy(currentSoundData.tolist())
        res.append(cur)
        if i % VERBOSE_PERIOD == 0:
            print('Extracting', verboseTypeData, i+1)

    return res

def extractListPathWithOverlap(pathSounds:list[str], featureType: str, verboseTypeData: str):
    res = []
    for i in range(len(pathSounds)):
        pathSound = pathSounds[i]
        currentSoundData = None
        if featureType == FeatureType.LOGMEL.value:
            currentSoundData = extractLogMelFromPath(pathSound)
        elif featureType == FeatureType.GAMMATONE.value:
            currentSoundData = extractGammatoneFromPath(pathSound)
        cur = copy.deepcopy(currentSoundData.tolist())
        for j in range(1,5):
            for k in range(len(currentSoundData)-1):
                cur.append(np.concatenate((currentSoundData[k,j*64:],currentSoundData[k+1,:j*64])).tolist())
        res.append(cur)
        if i % VERBOSE_PERIOD == 0:
            print('Extracting', verboseTypeData, i+1)
    
    return res


def train(modelType,featureType):
    listFileSoundNormal = []
    for i in range(len(PATH_INPUT_SOUND_NORMAL)):
        listFiles = [f for f in os.listdir(PATH_INPUT_SOUND_NORMAL[i]) if isfile(join(PATH_INPUT_SOUND_NORMAL[i],f)) and f[-4:] == '.wav']
        for j in range(len(listFiles)):
            listFileSoundNormal.append(f"{PATH_INPUT_SOUND_NORMAL[i]}/{listFiles[j]}")


    listFileSoundAbnormal = []
    for i in range(len(PATH_INPUT_SOUND_ABNORMAL)):
        listFiles = [f for f in os.listdir(PATH_INPUT_SOUND_ABNORMAL[i]) if isfile(join(PATH_INPUT_SOUND_ABNORMAL[i],f)) and f[-4:] == '.wav']
        for j in range(len(listFiles)):
            listFileSoundAbnormal.append(f"{PATH_INPUT_SOUND_ABNORMAL[i]}/{listFiles[j]}")

    listFileSoundNormal.sort()
    listFileSoundAbnormal.sort()

    random.Random(SEED).shuffle(listFileSoundNormal)
    random.Random(SEED).shuffle(listFileSoundAbnormal)

    totalDataNormal = len(listFileSoundNormal)
    totalDataAbnormal = len(listFileSoundAbnormal)

    print('Total Data: ')
    print(totalDataNormal,totalDataAbnormal)

    offsetDataTrain = 0
    maxIdxDataTrain = totalDataNormal - totalDataAbnormal

    offsetDataNormalTest = totalDataNormal - totalDataAbnormal
    maxIdxDataNormalTest = totalDataNormal

    offsetDataAbnormalTest = 0
    maxIdxDataAbnormalTest = totalDataAbnormal

    minDataTest = min(maxIdxDataNormalTest-offsetDataNormalTest,totalDataAbnormal)

    maxIdxDataAbnormalTest = minDataTest
    maxIdxDataNormalTest = offsetDataNormalTest + minDataTest

    offsetDataNormalVal = offsetDataNormalTest
    maxIdxDataNormalVal = offsetDataNormalTest + (maxIdxDataNormalTest - offsetDataNormalTest)//2

    offsetDataNormalTest = maxIdxDataNormalVal

    offsetDataAbnormalVal = 0
    maxIdxDataAbnormalVal = maxIdxDataAbnormalTest//2

    offsetDataAbnormalTest = maxIdxDataAbnormalVal

    print('Data Allocation: ')
    print(offsetDataTrain,maxIdxDataTrain,offsetDataNormalTest,maxIdxDataNormalTest,offsetDataAbnormalTest,maxIdxDataAbnormalTest)

    trainFiles = []
    for i in range(offsetDataTrain,maxIdxDataTrain):
        trainFiles.append(listFileSoundNormal[i])

    testFiles = []
    for i in range(offsetDataNormalTest,maxIdxDataNormalTest):
        testFiles.append(listFileSoundNormal[i])

    for i in range(offsetDataAbnormalTest,maxIdxDataAbnormalTest):
        testFiles.append(listFileSoundAbnormal[i])
        
    valFiles = []

    for i in range(offsetDataNormalVal,maxIdxDataNormalVal):
        valFiles.append(listFileSoundNormal[i])

    for i in range(offsetDataAbnormalVal,maxIdxDataAbnormalVal):
        valFiles.append(listFileSoundAbnormal[i])

    trainLabels = [0 for i in range(len(trainFiles))]
    testLabels = [0 for i in range(maxIdxDataNormalTest - offsetDataNormalTest)] + [1 for i in range(maxIdxDataAbnormalTest - offsetDataAbnormalTest)]
    valLabels = [0 for i in range(maxIdxDataNormalVal - offsetDataNormalVal)] + [1 for i in range(maxIdxDataAbnormalVal - offsetDataAbnormalVal)]


    dataset = {
        "train_files": trainFiles,
        "test_files": testFiles,
        "val_files": valFiles,
        "train_labels": trainLabels,
        "test_labels": testLabels,
        "val_labels": valLabels,
    }

    extractedDataSoundTrain = []
    extractedDataSoundTest = []
    extractedDataSoundVal = []

    # Overlapping frames for IDNN based model
    if modelType == ModelType.AEIDNN.value or modelType == ModelType.UNETIDNN.value:
        extractedDataSoundTrain = extractListPathWithOverlap(dataset["train_files"],featureType,'Train')
        extractedDataSoundTest = extractListPathWithOverlap(dataset["test_files"],featureType, 'Test')
        extractedDataSoundVal = extractListPathWithOverlap(dataset["val_files"],featureType, 'Validation')
    elif modelType == ModelType.AE.value or modelType == ModelType.UNET.value:
        extractedDataSoundTrain = extractListPathWithNoOverlap(dataset["train_files"],featureType,'Train')
        extractedDataSoundTest = extractListPathWithNoOverlap(dataset["test_files"],featureType, 'Test')
        extractedDataSoundVal = extractListPathWithNoOverlap(dataset["val_files"],featureType, 'Validation')

    trainLoader = DataLoader(dataset=np.array(extractedDataSoundTrain,dtype='float32'),
                                              batch_size=64,
                                              shuffle=True)
    valLoader = DataLoader(dataset=np.array(extractedDataSoundVal,dtype='float32'),
                                              batch_size=1,
                                              shuffle=False)
    testLoader = DataLoader(dataset=np.array(extractedDataSoundTest,dtype='float32'),
                                              batch_size=1,
                                              shuffle=False)
    
    anomalyScore = None
    if modelType == ModelType.AE.value:
        anomalyScore = trainAE(trainLoader,valLoader,testLoader,dataset,yamlConfig['model'])
    elif modelType == ModelType.UNET.value:
        anomalyScore = trainUNet(trainLoader,valLoader,testLoader,dataset,yamlConfig['model'])
    elif modelType == ModelType.AEIDNN.value:
        anomalyScore = trainIdnn(trainLoader,valLoader,testLoader,dataset,yamlConfig['model'])
    elif modelType == ModelType.UNETIDNN.value:
        anomalyScore = trainUNetIDNN(trainLoader,valLoader,testLoader,dataset,yamlConfig['model'])
    
    # AUC Score Evaluation
    fpr, tpr, _ = metrics.roc_curve(dataset['test_labels'], anomalyScore)
    roc_auc = metrics.auc(fpr, tpr)

    print("AUC Score =",roc_auc)

    # method I: plt
    import matplotlib.pyplot as plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()