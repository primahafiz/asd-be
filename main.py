from feature import *
from inference import *

# visualizeSpectrogram(convertSpectrogramToVisualizedForm(extractLogMelFromPath('./data/sample/slider6_00000029.wav')))

loss = inferenceUNetIDNN(FeatureType.GAMMATONE,MachineType.VALVE,0)

print(loss)