import torch
import torch.nn as nn
import torch
from torch import nn
from scipy.stats import norm

class Idnn(nn.Module):
    def __init__(self):
        super(Idnn, self).__init__()
        self.input_dim = 64 * 4
        self.mel_bins = 64
        self.avg_loss = 0
        self.std_loss = 0
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, self.mel_bins)
        )

    def forward(self, x):
        context, middle_frame = self.preprocess(x)
        z = self.encoder(context)
        output = self.decoder(z)
        return output, middle_frame.reshape(-1,64)
    
    def preprocess(self, random_frames):
        left = random_frames[:, :, 0:64*2]

        middle_frame = random_frames[:, :, 64*2:64*3]

        right = random_frames[:, :, 64*3:]

        context = torch.cat((left, right), dim=-1)

        context = context.reshape(-1,context.shape[2])
        
        return context, middle_frame

    
    def setLossStats(self,avgLoss,stdLoss):
        self.avg_loss = avgLoss
        self.std_loss = stdLoss
        
    def calcLoss(self,x,criterion):
        recon,target = self.forward(x)
        loss = criterion(recon, target)
        normalizedLoss = norm(self.avg_loss,self.std_loss).cdf(loss.item())
        return normalizedLoss
