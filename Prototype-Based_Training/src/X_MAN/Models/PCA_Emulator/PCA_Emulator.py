import torch
import torch.nn as nn

class PCA_Emulator(nn.Module):
    
    def __init__(self, origNumComp : int, newNumComp : int):
        super(PCA_Emulator, self).__init__()
        
        self.transform = nn.Sequential(nn.Linear(origNumComp, newNumComp, bias=True), nn.LeakyReLU(1))

    def forward(self, x):
        out = self.transform(x)
        
        return out