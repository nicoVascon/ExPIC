import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow import keras

import torch
import copy
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

import numpy as np

VGG16_keras = VGG16(weights="imagenet", include_top= True, classes=1000)
layer_name = 'fc2'
VGG16_keras = keras.Model(inputs=VGG16_keras.input,outputs=VGG16_keras.get_layer(layer_name).output)
VGG16_keras.summary()

save_model_path = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/Prototype-Based_Training/VGG16_Keras_to_PyTorch.pt"

import onnx
import tf2onnx
from onnx2pytorch import ConvertModel

spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)

output_path = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/Prototype-Based_Training/VGG16_Keras_to_Onnx.onnx"

onnx_model, _ = tf2onnx.convert.from_keras(VGG16_keras, input_signature=spec, opset=13, output_path=output_path)
pytorch_model = ConvertModel(onnx_model)
# torch.save(pytorch_model.state_dict(), save_model_path)
torch.save(pytorch_model, save_model_path)
