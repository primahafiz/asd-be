from enum import Enum

class FeatureType(Enum):
    LOGMEL = 'logmel'
    GAMMATONE = 'gammatone'

class ModelType(Enum):
    UNETIDNN = 'unetidnn'
    AEIDNN = 'idnn'
    UNET = 'unet'
    AE = 'autoencoder'

class MachineType(Enum):
    SLIDER = 'slider'
    FAN = 'fan'
    PUMP = 'pump'
    VALVE = 'valve'
