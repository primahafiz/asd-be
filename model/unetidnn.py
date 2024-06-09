import torch
import torch.nn as nn
import torch
from torch import nn
from scipy.stats import norm

class UNetIdnn(nn.Module):
        def __init__(self,numLayer):
            super(UNetIdnn, self).__init__()
            self.input_dim = 64 * 4
            self.mel_bins = 64
            self.num_layer = numLayer
            self.avg_loss = 0
            self.std_loss = 0
            
            encoderLayer = [32,16]
            if self.num_layer > 2:
                encoderLayer = [64,32] + encoderLayer
            if self.num_layer > 3:
                encoderLayer = [64 for i in range(2*(self.num_layer-3))] + encoderLayer
            encoderLayer = [self.input_dim,encoderLayer[0]] + encoderLayer
            
            decoderLayer = [16,32] + [64,64]
            if self.num_layer > 2:
                decoderLayer = decoderLayer + [128 if i%2==0 else 64 for i in range(2*(self.num_layer-2))]
                
            self.encoder = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(encoderLayer[2*i],encoderLayer[2*i+1]),
                    nn.BatchNorm1d(encoderLayer[2*i+1]),
                    nn.ReLU()
                )
                for i in range(len(encoderLayer)//2)
            ])
            
            self.decoder = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(decoderLayer[2*i],decoderLayer[2*i+1]),
                    nn.BatchNorm1d(decoderLayer[2*i+1]),
                    nn.ReLU()
                )
                for i in range(len(decoderLayer)//2-1)
            ])
            
            self.decoder.append(
                nn.Sequential(
                    nn.Linear(decoderLayer[len(decoderLayer)-2],decoderLayer[len(decoderLayer)-1])
                )
            )

            
        def preprocess_input(self, x):
            left = x[:, :, 0:64*2]
            middle = x[:, :, 64*2:64*3]
            right = x[:, :, 64*3:]
            context = torch.cat((left, right), dim=-1)
            return context.transpose(1, 2), middle

        def forward(self, x):
            context, middle_frame = self.preprocess(x)
            
            e = [0 for i in range(len(self.encoder))]
            d = [0 for i in range(len(self.decoder))]
            
            for i in range(len(e)):
                if i == 0:
                    e[i] = self.encoder[i](context)
                else:
                    e[i] = self.encoder[i](e[i-1])
                    
            for i in range(len(d)):
                if i == 0:
                    d[i] = self.decoder[i](e[len(e)-i-1])
                else:
                    d[i] = self.decoder[i](torch.cat((e[len(e)-i-1],d[i-1]),dim=1))
            
            return d[len(d)-1], middle_frame.reshape(-1,64)

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