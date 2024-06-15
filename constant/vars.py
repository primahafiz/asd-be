from .type import *

# Audio file
SAMPLE_RATE = 16000
PATH_TMP = './data/tmp/'

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

IDNN_STATS[FeatureType.LOGMEL.value][MachineType.VALVE.value][0] = {'mean':10.859665709026789  ,'std':3.81141161892206}
IDNN_STATS[FeatureType.LOGMEL.value][MachineType.VALVE.value][2] = {'mean':12.426837841669718   ,'std':5.525098692469147}
IDNN_STATS[FeatureType.LOGMEL.value][MachineType.VALVE.value][4] = {'mean':13.039303382237753  ,'std':7.687970100234219}
IDNN_STATS[FeatureType.LOGMEL.value][MachineType.VALVE.value][6] = {'mean':10.580167977015178   ,'std':3.95782499340724}
IDNN_STATS[FeatureType.LOGMEL.value][MachineType.SLIDER.value][0] = {'mean':11.46715271338988   ,'std':4.883634890339186}
IDNN_STATS[FeatureType.LOGMEL.value][MachineType.SLIDER.value][2] = {'mean':12.498674299483909   ,'std':3.767104219082907}
IDNN_STATS[FeatureType.LOGMEL.value][MachineType.SLIDER.value][4] = {'mean':13.339369891734606  ,'std':3.9372256232921847}
IDNN_STATS[FeatureType.LOGMEL.value][MachineType.SLIDER.value][6] = {'mean':12.764654777266763  ,'std':4.864927864881787}
IDNN_STATS[FeatureType.LOGMEL.value][MachineType.FAN.value][0] = {'mean':7.3483967781066895  ,'std':1.7598609543727035}
IDNN_STATS[FeatureType.LOGMEL.value][MachineType.FAN.value][2] = {'mean':6.1018389936265995  ,'std':0.6824203777717801}
IDNN_STATS[FeatureType.LOGMEL.value][MachineType.FAN.value][4] = {'mean':5.770696489290259  ,'std':0.6815690250573724}
IDNN_STATS[FeatureType.LOGMEL.value][MachineType.FAN.value][6] = {'mean':6.179108450147841  ,'std':1.5188924649335098}
IDNN_STATS[FeatureType.LOGMEL.value][MachineType.PUMP.value][0] = {'mean':12.008172075513382  ,'std':1.418887325562146}
IDNN_STATS[FeatureType.LOGMEL.value][MachineType.PUMP.value][2] = {'mean':11.14604400287975  ,'std':1.7266334595194077}
IDNN_STATS[FeatureType.LOGMEL.value][MachineType.PUMP.value][4] = {'mean':7.953440008163452  ,'std':2.3088967849205333}
IDNN_STATS[FeatureType.LOGMEL.value][MachineType.PUMP.value][6] = {'mean':9.738179038552676   ,'std':2.45307160203343}

IDNN_PATH = {}
for f in FeatureType:
    IDNN_PATH[f.value] = {}
    for m in MachineType:
        IDNN_PATH[f.value][m.value] = {}
        for g in range(0,8,2):
            IDNN_PATH[f.value][m.value][g] = f'./saved-model/{f.value}/{ModelType.AEIDNN.value}/{f.value}-{ModelType.AEIDNN.value}-{m.value}-{g}.pkl'

# UNet

UNET_STATS = {}

for f in FeatureType:
    UNET_STATS[f.value] = {}
    for m in MachineType:
        UNET_STATS[f.value][m.value] = {}
        for g in range(0,8,2):
            UNET_STATS[f.value][m.value][g] = {'mean':0,'std':0}

UNET_STATS[FeatureType.GAMMATONE.value][MachineType.VALVE.value][0] = {'mean':282.95537735243977 ,'std':80.5409081401243}
UNET_STATS[FeatureType.GAMMATONE.value][MachineType.VALVE.value][2] = {'mean':440.65931599934896 ,'std':101.15301976515619}
UNET_STATS[FeatureType.GAMMATONE.value][MachineType.VALVE.value][4] = {'mean':430.11817474365233 ,'std':336.36434644813215}
UNET_STATS[FeatureType.GAMMATONE.value][MachineType.VALVE.value][6] = {'mean':456.9370844523112 ,'std':122.03355670864042}
UNET_STATS[FeatureType.GAMMATONE.value][MachineType.SLIDER.value][0] = {'mean':539.4229527162702 ,'std':169.19005722996633}
UNET_STATS[FeatureType.GAMMATONE.value][MachineType.SLIDER.value][2] = {'mean':887.7867817125822 ,'std':183.0222255527423}
UNET_STATS[FeatureType.GAMMATONE.value][MachineType.SLIDER.value][4] = {'mean':1169.0739602078213 ,'std':223.60926961398232}
UNET_STATS[FeatureType.GAMMATONE.value][MachineType.SLIDER.value][6] = {'mean':1450.0045179887252 ,'std':318.67743307404305}
UNET_STATS[FeatureType.GAMMATONE.value][MachineType.FAN.value][0] = {'mean':872.1722748855065 ,'std':150.9096141079823}
UNET_STATS[FeatureType.GAMMATONE.value][MachineType.FAN.value][2] = {'mean':1033.5365266426982 ,'std':90.76322511898142}
UNET_STATS[FeatureType.GAMMATONE.value][MachineType.FAN.value][4] = {'mean':986.7419612489898,'std':104.08510445864748}
UNET_STATS[FeatureType.GAMMATONE.value][MachineType.FAN.value][6] = {'mean':666.1144410875108,'std':129.76474089729692}
UNET_STATS[FeatureType.GAMMATONE.value][MachineType.PUMP.value][0] = {'mean':804.8636435925121,'std':193.4580255314632}
UNET_STATS[FeatureType.GAMMATONE.value][MachineType.PUMP.value][2] = {'mean':581.6889881480823 ,'std':204.2897792729988}
UNET_STATS[FeatureType.GAMMATONE.value][MachineType.PUMP.value][4] = {'mean':573.8050891113281,'std':118.22501887502781}
UNET_STATS[FeatureType.GAMMATONE.value][MachineType.PUMP.value][6] = {'mean':673.4198153626685 ,'std':673.4198153626685 }

UNET_STATS[FeatureType.LOGMEL.value][MachineType.VALVE.value][0] = {'mean':6.546152947312694,'std':1.5782103638046736}
UNET_STATS[FeatureType.LOGMEL.value][MachineType.VALVE.value][2] = {'mean':8.258123254776,'std':1.9625969409708295}
UNET_STATS[FeatureType.LOGMEL.value][MachineType.VALVE.value][4] = {'mean':7.399481081962586 ,'std':1.9170007153446356}
UNET_STATS[FeatureType.LOGMEL.value][MachineType.VALVE.value][6] = {'mean':6.964283474286398,'std':1.2519733585158328}
UNET_STATS[FeatureType.LOGMEL.value][MachineType.SLIDER.value][0] = {'mean':7.96979637896077 ,'std':2.5991325798298144}
UNET_STATS[FeatureType.LOGMEL.value][MachineType.SLIDER.value][2] = {'mean':6.985688883559148 ,'std':1.4428771067391046}
UNET_STATS[FeatureType.LOGMEL.value][MachineType.SLIDER.value][4] = {'mean':12.496844055947292 ,'std':2.9071935630233607}
UNET_STATS[FeatureType.LOGMEL.value][MachineType.SLIDER.value][6] = {'mean':10.264550783417441 ,'std':2.9635733593878135}
UNET_STATS[FeatureType.LOGMEL.value][MachineType.FAN.value][0] = {'mean':6.461556037658541,'std':1.284913454558021}
UNET_STATS[FeatureType.LOGMEL.value][MachineType.FAN.value][2] = {'mean':4.224566636804762,'std':0.2574060807027397}
UNET_STATS[FeatureType.LOGMEL.value][MachineType.FAN.value][4] = {'mean':4.374068834315771,'std':0.3829215224240193}
UNET_STATS[FeatureType.LOGMEL.value][MachineType.FAN.value][6] = {'mean':4.7514107240570915 ,'std':0.8484633825427739}
UNET_STATS[FeatureType.LOGMEL.value][MachineType.PUMP.value][0] = {'mean':8.2439872244714 ,'std':0.7795831509369724}
UNET_STATS[FeatureType.LOGMEL.value][MachineType.PUMP.value][2] = {'mean':7.710004312341863 ,'std':0.9449543845911139}
UNET_STATS[FeatureType.LOGMEL.value][MachineType.PUMP.value][4] = {'mean':5.041432113647461 ,'std':0.7289398107304101}
UNET_STATS[FeatureType.LOGMEL.value][MachineType.PUMP.value][6] = {'mean':7.176342094645781 ,'std':1.683630987014549}

UNET_PATH = {}
for f in FeatureType:
    UNET_PATH[f.value] = {}
    for m in MachineType:
        UNET_PATH[f.value][m.value] = {}
        for g in range(0,8,2):
            UNET_PATH[f.value][m.value][g] = f'./saved-model/{f.value}/{ModelType.UNET.value}/{f.value}-{ModelType.UNET.value}-{m.value}-{g}.pkl'


# Autoencoder

AE_STATS = {}

for f in FeatureType:
    AE_STATS[f.value] = {}
    for m in MachineType:
        AE_STATS[f.value][m.value] = {}
        for g in range(0,8,2):
            AE_STATS[f.value][m.value][g] = {'mean':0,'std':0}

AE_STATS[FeatureType.GAMMATONE.value][MachineType.VALVE.value][0] = {'mean':532.0957145044359  ,'std':213.4420848365593}
AE_STATS[FeatureType.GAMMATONE.value][MachineType.VALVE.value][2] = {'mean':865.272553507487 ,'std':216.37682212258528}
AE_STATS[FeatureType.GAMMATONE.value][MachineType.VALVE.value][4] = {'mean':784.0978480021159  ,'std':585.1499609461677}
AE_STATS[FeatureType.GAMMATONE.value][MachineType.VALVE.value][6] = {'mean':839.1335174560547  ,'std':267.3274290728484}
AE_STATS[FeatureType.GAMMATONE.value][MachineType.SLIDER.value][0] = {'mean':983.8712902283401  ,'std':282.4645353283643}
AE_STATS[FeatureType.GAMMATONE.value][MachineType.SLIDER.value][2] = {'mean':1430.5257352671229  ,'std':261.16565639852126}
AE_STATS[FeatureType.GAMMATONE.value][MachineType.SLIDER.value][4] = {'mean':1470.3954460915554  ,'std':243.2094831286745}
AE_STATS[FeatureType.GAMMATONE.value][MachineType.SLIDER.value][6] = {'mean':2006.4040249911222  ,'std':468.92405062786594}
AE_STATS[FeatureType.GAMMATONE.value][MachineType.FAN.value][0] = {'mean':1333.8214057208281  ,'std':199.0023403715269}
AE_STATS[FeatureType.GAMMATONE.value][MachineType.FAN.value][2] = {'mean':1192.0123008003447  ,'std':126.1412617545919}
AE_STATS[FeatureType.GAMMATONE.value][MachineType.FAN.value][4] = {'mean':1213.6945022056843  ,'std':127.14687652171513}
AE_STATS[FeatureType.GAMMATONE.value][MachineType.FAN.value][6] = {'mean':1043.7298214382595  ,'std':183.90272206766187}
AE_STATS[FeatureType.GAMMATONE.value][MachineType.PUMP.value][0] = {'mean':1120.5898480482504  ,'std':278.5968057095847}
AE_STATS[FeatureType.GAMMATONE.value][MachineType.PUMP.value][2] = {'mean':838.6603049538352 ,'std':279.33555090111264}
AE_STATS[FeatureType.GAMMATONE.value][MachineType.PUMP.value][4] = {'mean':999.3321008300782  ,'std':403.0262429558882}
AE_STATS[FeatureType.GAMMATONE.value][MachineType.PUMP.value][6] = {'mean':925.4919493432138  ,'std':262.12964649361993}

AE_STATS[FeatureType.LOGMEL.value][MachineType.VALVE.value][0] = {'mean':12.27461813263974  ,'std':3.8949891505234997}
AE_STATS[FeatureType.LOGMEL.value][MachineType.VALVE.value][2] = {'mean':15.11864849726359   ,'std':4.042797145771021}
AE_STATS[FeatureType.LOGMEL.value][MachineType.VALVE.value][4] = {'mean':11.61592926979065   ,'std':5.409055245713924}
AE_STATS[FeatureType.LOGMEL.value][MachineType.VALVE.value][6] = {'mean':10.918512479464214   ,'std':3.5045246944806814}
AE_STATS[FeatureType.LOGMEL.value][MachineType.SLIDER.value][0] = {'mean':12.943723260686639   ,'std':4.797178744446813}
AE_STATS[FeatureType.LOGMEL.value][MachineType.SLIDER.value][2] = {'mean':4.797178744446813  ,'std':3.3427124655136504}
AE_STATS[FeatureType.LOGMEL.value][MachineType.SLIDER.value][4] = {'mean':16.029850348997652   ,'std':3.5098851281599313}
AE_STATS[FeatureType.LOGMEL.value][MachineType.SLIDER.value][6] = {'mean':15.749767325141214   ,'std':4.543140358894665}
AE_STATS[FeatureType.LOGMEL.value][MachineType.FAN.value][0] = {'mean':6.946994891894862   ,'std':1.6033529005671578}
AE_STATS[FeatureType.LOGMEL.value][MachineType.FAN.value][2] = {'mean':5.745640866583286   ,'std':0.5514983517538955}
AE_STATS[FeatureType.LOGMEL.value][MachineType.FAN.value][4] = {'mean':5.770193176708002   ,'std':0.6802899279213345}
AE_STATS[FeatureType.LOGMEL.value][MachineType.FAN.value][6] = {'mean':5.615332105424669   ,'std':1.0852488276574115}
AE_STATS[FeatureType.LOGMEL.value][MachineType.PUMP.value][0] = {'mean':11.712823437972807   ,'std':1.3230869376546968}
AE_STATS[FeatureType.LOGMEL.value][MachineType.PUMP.value][2] = {'mean':11.235299197110264   ,'std':1.7742726001840472}
AE_STATS[FeatureType.LOGMEL.value][MachineType.PUMP.value][4] = {'mean':7.939149293899536   ,'std':2.095285349397584}
AE_STATS[FeatureType.LOGMEL.value][MachineType.PUMP.value][6] = {'mean':9.459083005493762   ,'std':2.3850163516430936}

AE_PATH = {}
for f in FeatureType:
    AE_PATH[f.value] = {}
    for m in MachineType:
        AE_PATH[f.value][m.value] = {}
        for g in range(0,8,2):
            AE_PATH[f.value][m.value][g] = f'./saved-model/{f.value}/{ModelType.AE.value}/{f.value}-{ModelType.AE.value}-{m.value}-{g}.pkl'