import torch
import torch.nn as nn
import torch
from torch import nn
from scipy.stats import norm

class UNet(nn.Module):
        def __init__(self):
            self.input_dim = 320
            super().__init__()        
            

            self.encoder1 = nn.Sequential(
                nn.Linear(self.input_dim,128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
            )
            self.encoder2 = nn.Sequential(
                nn.Linear(128,128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
            )
            self.encoder3 = nn.Sequential(
                nn.Linear(128,128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
            )
            self.encoder4 = nn.Sequential(
                nn.Linear(128,128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
            )
            self.encoder5 = nn.Sequential(
                nn.Linear(128,8),
                nn.BatchNorm1d(8),
                nn.ReLU(),
            )

            self.decoder1 = nn.Sequential(
                nn.Linear(8,128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
            )
            self.decoder2 = nn.Sequential(
                nn.Linear(128,128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
            )
            self.decoder3 = nn.Sequential(
                nn.Linear(128,128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
            )
            self.decoder4 = nn.Sequential(
                nn.Linear(128,128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
            )
            self.decoder5 = nn.Sequential(
                nn.Linear(128,self.input_dim)
            )

        def forward(self, x):
            e1 = self.encoder1(x)
            e2 = self.encoder2(e1)
            e3 = self.encoder3(e2)
            e4 = self.encoder4(e3)
            e5 = self.encoder5(e4)

            d1 = self.decoder1(e5)
            d2 = self.decoder2((e4+d1)/2)
            d3 = self.decoder3((e3+d2)/2)
            d4 = self.decoder4((e2+d3)/2)
            decoded = self.decoder5((e1+d4)/2)
        
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