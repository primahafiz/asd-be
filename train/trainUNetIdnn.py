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
from scipy.stats import norm

from model import UNetIdnn
from constant import *

def trainUNetIDNN(trainLoader: DataLoader, valLoader: DataLoader, testLoader: DataLoader, dataset: dict, modelConfigs: dict):
    model = UNetIdnn(NUM_LAYER_UNETIDNN_GAMMATONE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=modelConfigs['lr'])
    
    if modelConfigs['cuda']:
        model.cuda()

    # Training process
    model.train()
    for epoch in range(modelConfigs['epoch']):
        avg = 0
        for extractedFeature in trainLoader:
            if dataset['val_labels'] == 1:
                continue
            if modelConfigs['cuda']:
                extractedFeature = extractedFeature.cuda()
            recon, target = model(extractedFeature)
            loss = criterion(recon, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg += loss.item()
            
        avg /= len(dataset['train_files'])
        print("EPOCH",epoch + 1,"LOSS =",avg)
    
    # Save model
    torch.save(model.state_dict(),f'{modelConfigs["save_path"]}/model.pkl')
    print('Model saved in',f'{modelConfigs["save_path"]}/model.pkl')

    # Evaluation for validation data
    model.eval()
    reconErrorVal = []
    for extractedFeature in valLoader:
        if modelConfigs['cuda']:
            extractedFeature = extractedFeature.cuda()
        recon, target = model(extractedFeature)
        loss = criterion(recon, target)
        reconErrorVal.append(loss.item())
    
    reconErrorVal = np.array(reconErrorVal)
    avgNorm = np.mean(reconErrorVal)
    stdNorm = np.std(reconErrorVal)

    print(f'Anomaly Score stats: mean = {avgNorm}, std = {stdNorm}')

    # Evaluation for testing data with normalization postprocessing
    reconErrorTest = []
    for extractedFeature in testLoader:
        if modelConfigs['cuda']:
            extractedFeature = extractedFeature.cuda()
        recon, target = model(extractedFeature)
        loss = criterion(recon, target)
        reconErrorTest.append(norm(avgNorm,stdNorm).cdf(loss.item()))

    # return normalized anomaly score
    return reconErrorTest