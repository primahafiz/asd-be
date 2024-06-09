from constant import *
from model import *
import torch
from feature import *
from torchvision.transforms import ToTensor

def inferenceUNetIDNN(featureType: FeatureType, machineType: MachineType, id: int):
    model = UNetIdnn(NUM_LAYER_UNETIDNN_GAMMATONE)
    model.load_state_dict(torch.load(UNET_IDNN_PATH[featureType.value][machineType.value][id],map_location=torch.device('cpu')))

    model.eval()

    extractedFeature = expandSpectrogram(extractGammatoneFromPath('./data/sample/valve0abn_00000001.wav'))
    extractedFeature = np.array([extractedFeature],dtype='float32')

    transform = ToTensor()
    extractedFeature = transform(extractedFeature)
    extractedFeature = extractedFeature.transpose(0,1).transpose(1,2)
    
    model.setLossStats(UNET_IDNN_STATS[featureType.value][machineType.value][id]['mean'],UNET_IDNN_STATS[featureType.value][machineType.value][id]['std'])
    loss = model.calcLoss(extractedFeature,nn.MSELoss())

    return loss.item()