from constant import *
from model import *
import torch
from feature import *
from torchvision.transforms import ToTensor

def inferenceUNetIDNN(featureType: FeatureType, machineType: MachineType, id: int, audioFilePath: str):
    model = None
    if featureType.value == FeatureType.GAMMATONE.value:
        model = UNetIdnn(NUM_LAYER_UNETIDNN_GAMMATONE)
    elif featureType.value == FeatureType.LOGMEL.value:
        model = UNetIdnn(NUM_LAYER_UNETIDNN_LOGMEL)
        
    model.load_state_dict(torch.load(UNET_IDNN_PATH[featureType.value][machineType.value][id],map_location=torch.device('cpu')))

    model.eval()

    extractedFeature = np.array([],dtype='float32')

    if featureType.value == FeatureType.GAMMATONE.value:
        extractedFeature = expandSpectrogram(extractGammatoneFromPath(audioFilePath))
    elif featureType.value == FeatureType.LOGMEL.value:
        extractedFeature = expandSpectrogram(extractLogMelFromPath(audioFilePath))
        
    extractedFeature = np.array([extractedFeature],dtype='float32')

    transform = ToTensor()
    extractedFeature = transform(extractedFeature)
    extractedFeature = extractedFeature.transpose(0,1).transpose(1,2)
    
    model.setLossStats(UNET_IDNN_STATS[featureType.value][machineType.value][id]['mean'],UNET_IDNN_STATS[featureType.value][machineType.value][id]['std'])
    loss = model.calcLoss(extractedFeature,nn.MSELoss())

    return loss.item()

def inferenceIDNN(featureType: FeatureType, machineType: MachineType, id: int, audioFilePath: str):
    model = Idnn()
    model.load_state_dict(torch.load(IDNN_PATH[featureType.value][machineType.value][id],map_location=torch.device('cpu')))

    model.eval()

    extractedFeature = np.array([],dtype='float32')

    if featureType.value == FeatureType.GAMMATONE.value:
        extractedFeature = expandSpectrogram(extractGammatoneFromPath(audioFilePath))
    elif featureType.value == FeatureType.LOGMEL.value:
        extractedFeature = expandSpectrogram(extractLogMelFromPath(audioFilePath))
        
    extractedFeature = np.array([extractedFeature],dtype='float32')

    transform = ToTensor()
    extractedFeature = transform(extractedFeature)
    extractedFeature = extractedFeature.transpose(0,1).transpose(1,2)
    
    model.setLossStats(IDNN_STATS[featureType.value][machineType.value][id]['mean'],IDNN_STATS[featureType.value][machineType.value][id]['std'])
    loss = model.calcLoss(extractedFeature,nn.MSELoss())

    return loss.item()

def inferenceUNet(featureType: FeatureType, machineType: MachineType, id: int, audioFilePath: str):
    model = UNet()
    model.load_state_dict(torch.load(UNET_PATH[featureType.value][machineType.value][id],map_location=torch.device('cpu')))

    model.eval()

    extractedFeature = np.array([],dtype='float32')

    if featureType.value == FeatureType.GAMMATONE.value:
        extractedFeature = expandSpectrogram(extractGammatoneFromPath(audioFilePath))
    elif featureType.value == FeatureType.LOGMEL.value:
        extractedFeature = expandSpectrogram(extractLogMelFromPath(audioFilePath))
        
    extractedFeature = np.array([extractedFeature],dtype='float32')

    transform = ToTensor()
    extractedFeature = transform(extractedFeature)
    extractedFeature = extractedFeature.transpose(0,1).transpose(1,2)
    
    model.setLossStats(UNET_STATS[featureType.value][machineType.value][id]['mean'],UNET_STATS[featureType.value][machineType.value][id]['std'])
    loss = model.calcLoss(extractedFeature,nn.MSELoss())

    return loss.item()

def inferenceAE(featureType: FeatureType, machineType: MachineType, id: int, audioFilePath: str):
    model = Autoencoder()
    model.load_state_dict(torch.load(AE_PATH[featureType.value][machineType.value][id],map_location=torch.device('cpu')))

    model.eval()

    extractedFeature = np.array([],dtype='float32')

    if featureType.value == FeatureType.GAMMATONE.value:
        extractedFeature = expandSpectrogram(extractGammatoneFromPath(audioFilePath))
    elif featureType.value == FeatureType.LOGMEL.value:
        extractedFeature = expandSpectrogram(extractLogMelFromPath(audioFilePath))

    extractedFeature = np.array([extractedFeature],dtype='float32')

    transform = ToTensor()
    extractedFeature = transform(extractedFeature)
    extractedFeature = extractedFeature.transpose(0,1).transpose(1,2)
    
    model.setLossStats(AE_STATS[featureType.value][machineType.value][id]['mean'],AE_STATS[featureType.value][machineType.value][id]['std'])
    loss = model.calcLoss(extractedFeature,nn.MSELoss())

    return loss.item()