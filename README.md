# Anomalous Sound Detection System
Aims to identify machine's condition based on its sound to classify a machine as normal or anomalous (broken)

## System Specification

### Proposed
- Feature : Time Domain Gammatone Spectrogram
- Model : AE-IDNN & UNet-IDNN

### Baseline
- Feature : Log Mel Spectrogram
- Model : Autoencoder & UNet

## How To Run
1. Install requirements </br>
```pip install requirements.txt```
2. Run command in ```scripts``` folder