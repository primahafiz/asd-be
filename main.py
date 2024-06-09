from feature import *
from inference import *

# visualizeSpectrogram(convertSpectrogramToVisualizedForm(extractLogMelFromPath('./data/sample/slider6_00000029.wav')))

loss = inferenceIDNN(FeatureType.GAMMATONE,MachineType.SLIDER,6)

print(loss)