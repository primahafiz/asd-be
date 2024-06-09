from .type import *

# Audio file
SAMPLE_RATE = 16000

# Spectrogram
F_MIN = 60.0
F_MAX = 8000.0
NUM_BANDS = 64
FRAME_SIZE = 320
TOTAL_FRAMES = 310

# Gammatone
SCALING_GAMMATONE = 100000

# Log Mel
N_FFT = 1024
HOP_LENGTH = 512

# UNetIDNN
NUM_LAYER_UNETIDNN_GAMMATONE = 6
NUM_LAYER_UNETIDNN_LOGMEL = 3

UNET_IDNN_STATS = {}

for f in FeatureType:
    UNET_IDNN_STATS[f.value] = {}
    for m in MachineType:
        UNET_IDNN_STATS[f.value][m.value] = {}
        for g in range(0,8,2):
            UNET_IDNN_STATS[f.value][m.value][g] = {'mean':0,'std':0}

UNET_IDNN_STATS[FeatureType.GAMMATONE.value][MachineType.VALVE.value][0] = {'mean':320.1384732521187,'std':94.17234341108507}
UNET_IDNN_STATS[FeatureType.GAMMATONE.value][MachineType.VALVE.value][2] = {'mean':395.96305185953776,'std':92.52358849201727}
UNET_IDNN_STATS[FeatureType.GAMMATONE.value][MachineType.VALVE.value][4] = {'mean':428.59117228190104,'std':421.3530403126358}
UNET_IDNN_STATS[FeatureType.GAMMATONE.value][MachineType.VALVE.value][6] = {'mean':452.3229726155599,'std':144.18799129346291}

UNET_IDNN_STATS[FeatureType.GAMMATONE.value][MachineType.SLIDER.value][0] = {'mean':437.48609890026995,'std':154.49557077966782}
UNET_IDNN_STATS[FeatureType.GAMMATONE.value][MachineType.SLIDER.value][2] = {'mean':590.7560025552161,'std':149.19999090253478}
UNET_IDNN_STATS[FeatureType.GAMMATONE.value][MachineType.SLIDER.value][4] = {'mean':1024.1824882593048,'std':1024.1824882593048}
UNET_IDNN_STATS[FeatureType.GAMMATONE.value][MachineType.SLIDER.value][6] = {'mean':1163.0921020507812,'std':245.39268309626834}

UNET_IDNN_STATS[FeatureType.GAMMATONE.value][MachineType.FAN.value][0] = {'mean':714.4152077360106,'std':159.82165630540823}
UNET_IDNN_STATS[FeatureType.GAMMATONE.value][MachineType.FAN.value][2] = {'mean':1102.621286743846,'std':85.26187933385148}
UNET_IDNN_STATS[FeatureType.GAMMATONE.value][MachineType.FAN.value][4] = {'mean':1016.6967120992726 ,'std':105.55936238788759}
UNET_IDNN_STATS[FeatureType.GAMMATONE.value][MachineType.FAN.value][6] = {'mean':627.749743991428,'std':116.23654804008561}

UNET_IDNN_STATS[FeatureType.GAMMATONE.value][MachineType.PUMP.value][0] = {'mean':951.870527240592 ,'std':239.97757551958415}
UNET_IDNN_STATS[FeatureType.GAMMATONE.value][MachineType.PUMP.value][2] = {'mean':644.7364274458452 ,'std':216.13348517217617}
UNET_IDNN_STATS[FeatureType.GAMMATONE.value][MachineType.PUMP.value][4] = {'mean':572.7998529052734,'std':117.16903048410492}
UNET_IDNN_STATS[FeatureType.GAMMATONE.value][MachineType.PUMP.value][6] = {'mean':657.8441042432598,'std':224.00959825005597}

UNET_IDNN_PATH = {}
for f in FeatureType:
    UNET_IDNN_PATH[f.value] = {}
    for m in MachineType:
        UNET_IDNN_PATH[f.value][m.value] = {}
        for g in range(0,8,2):
            UNET_IDNN_PATH[f.value][m.value][g] = f'./saved-model/{f.value}/{ModelType.UNETIDNN.value}/{f.value}-{ModelType.UNETIDNN.value}-{m.value}-{g}.pkl'