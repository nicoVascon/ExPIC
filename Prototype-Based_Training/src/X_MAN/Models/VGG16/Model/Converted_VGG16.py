import torch
import torch.nn as nn
from enum import Enum

import X_MAN.Models.VGG16.Model as Model
from X_MAN.Models.PCA_Emulator.PCA_Emulator import PCA_Emulator

class Feature_Extraction_Layers(str, Enum):
    LAST_CONV = "last_conv"
    LAST_FC = "last_fc"
    
    def available_layers():
        layers = (member.value for member in Feature_Extraction_Layers)
        return list(layers)

class Converted_VGG16(nn.Module):
    
    def __init__(self, weights_path : str = None, fe_layer : str = "last_fc", numClasses : int = None, 
                    pca_emulator : bool = False, pca_emul_weights_path : str = None, 
                    pca_origNumComp : int = None, pca_newNumComp : int = None):
        super(Converted_VGG16, self).__init__()
        model_path = Model.__path__[0] + "/VGG16_Keras_to_PyTorch.pt"
        self.net = torch.load(model_path)
        if numClasses is not None:
            self.classifier = nn.Sequential(nn.Linear(4096, numClasses), nn.Sigmoid(), nn.Softmax(dim=1))
        elif fe_layer == "last_conv":
            self.net = self.net[:34]
        
        if pca_emulator:
            if pca_origNumComp is None or pca_newNumComp is None:
                raise Exception("Error!!! When 'pca_emulator' is True 'pca_origNumComp' and 'pca_newNumComp' must be passeed too.")
            self.pca_emulator = PCA_Emulator(origNumComp = pca_origNumComp, newNumComp = pca_newNumComp)
            if pca_emul_weights_path is not None:
                avail_devices_count = torch.cuda.device_count()
                if avail_devices_count < 2:
                    device = torch.device("cuda:0")
                    self.pca_emulator.load_state_dict(torch.load(pca_emul_weights_path, map_location=device))
                else:
                    self.pca_emulator.load_state_dict(torch.load(pca_emul_weights_path))

        if weights_path is not None:            
            self.load_state_dict(torch.load(weights_path))
        # self.fe_layer = fe_layer

    def forward(self, x):
        # newModel[32:](newModel[25:32](newModel[18:25](newModel[11:18](newModel[6:11](newModel[0:6](test_input)[0])[0])[0])[0])[0])
        out = self.net[0:6](x)[0]
        out = self.net[6:11](out)[0]
        out = self.net[11:18](out)[0]
        out = self.net[18:25](out)[0]
        out = self.net[25:32](out)[0]
        out = self.net[32:](out)
        if getattr(self, "classifier", None) is not None:
            out = self.classifier(out)
        if getattr(self, "pca_emulator", None) is not None:
            out = self.pca_emulator(out)
        return out
     