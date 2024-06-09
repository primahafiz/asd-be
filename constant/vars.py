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

# IDNN

IDNN_STATS = {}

for f in FeatureType:
    IDNN_STATS[f.value] = {}
    for m in MachineType:
        IDNN_STATS[f.value][m.value] = {}
        for g in range(0,8,2):
            IDNN_STATS[f.value][m.value][g] = {'mean':0,'std':0}

IDNN_STATS[FeatureType.GAMMATONE.value][MachineType.VALVE.value][0] = {'mean':381.89355856685313  ,'std':123.33687027981664}
IDNN_STATS[FeatureType.GAMMATONE.value][MachineType.VALVE.value][2] = {'mean':390.16126912434896 ,'std':78.28741634822008}
IDNN_STATS[FeatureType.GAMMATONE.value][MachineType.VALVE.value][4] = {'mean':462.56020634969076 ,'std':412.6959290324409}
IDNN_STATS[FeatureType.GAMMATONE.value][MachineType.VALVE.value][6] = {'mean':460.1874262491862 ,'std':119.70603934573582}
IDNN_STATS[FeatureType.GAMMATONE.value][MachineType.SLIDER.value][0] = {'mean':377.6050481046184 ,'std':98.36678064252752}
IDNN_STATS[FeatureType.GAMMATONE.value][MachineType.SLIDER.value][2] = {'mean':547.3517219715549 ,'std':103.70613660490116}
IDNN_STATS[FeatureType.GAMMATONE.value][MachineType.SLIDER.value][4] = {'mean':724.0422486723139 ,'std':159.42451629901882}
IDNN_STATS[FeatureType.GAMMATONE.value][MachineType.SLIDER.value][6] = {'mean':884.8889402909713 ,'std':222.98256105938265}
IDNN_STATS[FeatureType.GAMMATONE.value][MachineType.FAN.value][0] = {'mean':669.447391453635 ,'std':147.49811170645802}
IDNN_STATS[FeatureType.GAMMATONE.value][MachineType.FAN.value][2] = {'mean':1124.8422128688023 ,'std':102.27787308400933}
IDNN_STATS[FeatureType.GAMMATONE.value][MachineType.FAN.value][4] = {'mean':1046.039431166375 ,'std':113.21602502273076}
IDNN_STATS[FeatureType.GAMMATONE.value][MachineType.FAN.value][6] = {'mean':582.4351272583008 ,'std':131.41885983509104}
IDNN_STATS[FeatureType.GAMMATONE.value][MachineType.PUMP.value][0] = {'mean':971.5251671159771 ,'std':243.17199715558726}
IDNN_STATS[FeatureType.GAMMATONE.value][MachineType.PUMP.value][2] = {'mean':740.017079301314 ,'std':241.5424529330203}
IDNN_STATS[FeatureType.GAMMATONE.value][MachineType.PUMP.value][4] = {'mean':562.7903314208984 ,'std':106.5052749702296}
IDNN_STATS[FeatureType.GAMMATONE.value][MachineType.PUMP.value][6] = {'mean':732.5788604137945 ,'std':225.02054280057348}

IDNN_PATH = {}
for f in FeatureType:
    IDNN_PATH[f.value] = {}
    for m in MachineType:
        IDNN_PATH[f.value][m.value] = {}
        for g in range(0,8,2):
            IDNN_PATH[f.value][m.value][g] = f'./saved-model/{f.value}/{ModelType.AEIDNN.value}/{f.value}-{ModelType.AEIDNN.value}-{m.value}-{g}.pkl'