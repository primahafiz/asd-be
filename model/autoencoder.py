import torch
import torch.nn as nn
import torch
from torch import nn
from scipy.stats import norm

class Autoencoder(nn.Module):
        def __init__(self):
            self.input_dim = 320
            super().__init__()        
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim,64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64,64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64,8),
                nn.BatchNorm1d(8),
                nn.ReLU(),
            )

            self.decoder = nn.Sequential(
                nn.Linear(8,64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64,64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64,self.input_dim)
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded
    
        def setLossStats(self,avgLoss,stdLoss):
            self.avg_loss = avgLoss
            self.std_loss = stdLoss
            
        def calcLoss(self,x,criterion):
            x = x.reshape(-1,320)
            recon = self.forward(x)
            loss = criterion(recon, x)
            normalizedLoss = norm(self.avg_loss,self.std_loss).cdf(loss.item())
            return normalizedLoss
    