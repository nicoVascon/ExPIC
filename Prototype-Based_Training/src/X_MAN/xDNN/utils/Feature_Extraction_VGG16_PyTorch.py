#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 22:43:03 2020

@author: eduardosoares
"""
"""
Modified on Sun April  30 17:20:25 2023

@author: Nicolas Vasconcellos
"""

import re
import torch
import torchvision.models as models
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import os
import pandas as pd

import sys
sys.path.insert(0, '/nfs/home/nvasconcellos.it/softLinkTests/X-MAN/Prototype-Based_Training/src')
from X_MAN.Models.VGG16.Model.Converted_VGG16 import Converted_VGG16

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

def Extract_Features(model_dir, fe_layer = "last_fc", config_dic : dict = {}, **kwargs):
  
  if kwargs.get("model") is not None:
    model = kwargs.get("model")
  else:
    weights_path = model_dir + "/checkpoints/best_model.pt"
    if not os.path.exists(weights_path):
      weights_path = None
    
    model = Converted_VGG16(weights_path,fe_layer = fe_layer,
                              # numClasses = 2, # Temp Code
                              pca_emulator=config_dic["PCA"]["PCA_Emulator"],
                              pca_emul_weights_path=config_dic["PCA"]["Weights"],
                              pca_origNumComp=config_dic["PCA"]["Original_Num_Comp"],
                              pca_newNumComp=config_dic["PCA"]["New_Num_Comp"])
        
    # Temp Code #############
    # if fe_layer == "last_conv":
    #   model.classifier = None
    #   model.net = model.net[:34]
    # elif fe_layer == "last_fc":
    #   model.classifier = None
    #########################
    
    model.eval()
    if config_dic.get("Hardware") is not None and config_dic["Hardware"].get("GPU") is not None:
      device_indx = config_dic["Hardware"]["GPU"]
      avail_devices_count = torch.cuda.device_count()
      actual_device_indx = device_indx if device_indx < avail_devices_count else avail_devices_count - 1
      torch_device = "cuda:" + str(actual_device_indx)
      device = torch.device(torch_device if torch.cuda.is_available() else "cpu")
    else:
      device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
  
  device=next(model.parameters()).device
  
  if config_dic.get("Dataset") is not None and config_dic["Dataset"].get("Mask_dir") is not None:
    mask_dir = config_dic["Dataset"]["Mask_dir"]
  else:
    mask_dir = None

  # Load PCA V Matrix 
  if config_dic.get("PCA") is not None and config_dic["PCA"].get("V_Matrix") is not None:
    V_matrix = torch.load(config_dic["PCA"]["V_Matrix"]).to(torch.float32)
  else: 
    V_matrix = None  
  V_matrix = None  

  # V_matrix = V_matrix[:, :config_dic["PCA"]["New_Num_Comp"]]
  
  def extractor(img_path, net, mask = None):
      img = image.load_img(img_path, target_size=(224, 224))
      x = image.img_to_array(img)
      x = np.expand_dims(x, axis=0)
      x = preprocess_input(x)
      if mask is not None:
        print("Masked!!")
        mask = image.img_to_array(mask)
        x = x*(mask/255.0)
      
      x = torch.from_numpy(x.copy())
      x = x.to(device)

      # y = net(x[:,:3, :, :]).cpu()
      y = net(x).cpu()

      if V_matrix is not None:
        y = torch.matmul(y, V_matrix) # PCA Reduction
      
      y = torch.squeeze(y)
      y = y.data.numpy()

      return y

  fv_dic = {}

  # datas_dir = '/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/ArxivDataset_VGG16_80Train_20Test/ArxiveDataset_VGG16_FromMatLabSplit/'
  # datas_dir = '/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/dataset_NeuralNet_Journal/Split_Train72.2_Test27.8/'
  datas_dir = config_dic["Dataset"]["Directory"] + '/'

  for set_dir in os.listdir(datas_dir):
    
    data_dir = datas_dir + set_dir + '/'
    
    contents = os.listdir(data_dir)
    contents.sort()
    classes = [each for each in contents if os.path.isdir(data_dir + each)]
    #Each folder becomes a different class

    print(contents)
    print(classes)

    images = []
    batch = []
    labels = []

    j =0

    for each in classes: #Loop for the folders
      print("Starting {} images".format(each))
      class_path = data_dir + each
      files = os.listdir(class_path)
      
      for ii, file in enumerate(files, 1):

        img = os.path.join(class_path, file)
        if mask_dir is not None:
          mask_class_path = mask_dir + "/" + set_dir + "/" + each + "/lung masks"
          mask_path = os.path.join(mask_class_path, file)
          mask = image.load_img(mask_path, target_size=(224, 224))
        else:
          mask = None
        features = extractor(img, model, mask=mask) # Extract features using the VGG-16 structure
        batch.append(features)
        images.append(file)
        labels.append(str(j))
        print("finish {}".format(ii))
      j = j + 1    #Class iterator
    np_batch = np.array(batch)
    np_labels = np.array(labels)
    np_images = np.array(images)

    np_labels_T = np_labels.reshape(-1,1)
    np_images_T = np_images.reshape(-1,1)
    print(np_labels_T)


    np_images_labels = np.hstack((np_images_T,np_labels_T))
    print(np_images_labels)
    
    fv_dic[set_dir] = (np_batch, np_images_labels)

  # #Slpit the data into training and test sets
  # from sklearn.model_selection import train_test_split

  # X_train, X_test, y_train, y_test = train_test_split(
  # np_batch, np_images_labels, test_size=0.3, random_state=0)

  # X_train, y_train = fv_dic['train']
  # X_test, y_test = fv_dic['val']

  # print(X_test.shape)
  # print(y_test.shape)
  
  for set_name in fv_dic.keys():
    print(set_name + " X Shape: " + str(fv_dic[set_name][0].shape))
    print(set_name + " y Shape: " + str(fv_dic[set_name][1].shape))

  #Convert data to Pandas in order to save as .csv

  # data_df_X_train = pd.DataFrame(X_train)
  # data_df_y_train = pd.DataFrame(y_train)
  # data_df_X_test = pd.DataFrame(X_test)
  # data_df_y_test = pd.DataFrame(y_test)
  
  
  ##### Uncomment the following lines to save the data as .csv #####
  
  data_df_dict = {}
  for set_name in fv_dic.keys():
    data_df_dict[set_name] = {"X": pd.DataFrame(fv_dic[set_name][0]), "y": pd.DataFrame(fv_dic[set_name][1])}

  # print(data_df_X_train)
  # print(data_df_dict[set_name]["X"]) 

  if fe_layer == "last_conv":    
    output_dir = model_dir + '/Feature_Vectors_FromFlatten/'
    # output_dir = model_dir + '/Feature_Vectors_FromFlatten_PCA402c_xDNNEvolving/'
  else:
    output_dir = model_dir + '/Feature_Vectors/'
    # output_dir = model_dir + '/Feature_Vectors_PCA289c_xDNNOffline/'
  # output_dir = '/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/ArxivDataset_VGG16_80Train_20Test/Feature_Vectors/'

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  # Save file as .csv
  # data_df_X_train.to_csv(output_dir + 'data_df_X_train.csv',header=False,index=False)
  # data_df_y_train.to_csv(output_dir + 'data_df_y_train.csv',header=False,index=False)
  # data_df_X_test.to_csv(output_dir + 'data_df_X_test.csv',header=False,index=False)
  # data_df_y_test.to_csv(output_dir + 'data_df_y_test.csv',header=False,index=False)
  for set_name in data_df_dict.keys():
    data_df_dict[set_name]["X"].to_csv(output_dir + "data_df_X_" + set_name + ".csv",header=False,index=False)
    data_df_dict[set_name]["y"].to_csv(output_dir + "data_df_y_" + set_name + ".csv",header=False,index=False)
  
  ##################################################################

  print(fv_dic["train"][0])
  print(fv_dic["train"][0].shape)
  
  return (fv_dic["train"][0], pd.DataFrame(fv_dic["train"][1]), fv_dic["val"][0], pd.DataFrame(fv_dic["val"][1]))
  # return (fv_dic["train"][0], pd.DataFrame(fv_dic["train"][1]), fv_dic["test"][0], pd.DataFrame(fv_dic["test"][1]))

def run(trainings_dir, **kwargs):
  
  if kwargs.get('fe_layer') == "last_conv":
    fe_layer = "last_conv"
  elif kwargs.get('fe_layer') == "last_fc" or kwargs.get('fe_layer') is None:
    fe_layer = "last_fc"
  else:
    raise Exception("Error!!!! fe_layer must be either 'last_conv' or 'last_fc'")    
  
  if kwargs.get("config_dic") is not None:
    config_dic = kwargs.get("config_dic")
  else:
    config_dic = {}
  
  if kwargs.get('multipleTrainings') is True:    
    trainings_names = os.listdir(trainings_dir)
    trainings_names.sort()
    for training_name in trainings_names:
      print(training_name)
      model_dir = trainings_dir + training_name + '/'
      return Extract_Features(model_dir, fe_layer=fe_layer, config_dic = config_dic)
  else:
    return Extract_Features(trainings_dir, fe_layer=fe_layer, config_dic = config_dic)

# run("/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/NeuralNetJournal_Dataset_Exp/Prototype-Based_Training/From_xDNN_Offline/Optuna/Prototype_Based_Training")

# Extract_Features("/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/NeuralNetJournal_Dataset_Exp/Prototype-Based_Training/From_xDNN_Offline/VGG16_BaseLine",
#                   fe_layer = "last_conv")

# import json

# # config_file_path = "/nfs/home/nvasconcellos.it/softLinkTests/X-MAN/Prototype-Based_Training/src/X_MAN/FineTuning/config/COVID-QU-Ex_Dataset/parameters_Offline_PCA_PyTorch_GPU0.json"
# config_file_path = "/nfs/home/nvasconcellos.it/softLinkTests/X-MAN/Prototype-Based_Training/src/X_MAN/FineTuning/config/DDSM_Dataset/parameters_Offline_PCA_PyTorch_GPU1_LowEpochs.json"
# config_dic = json.load(open(config_file_path))
# config_dic["Hardware"]["GPU"] = 1
# config_dic["PCA"]["V_Matrix"] = None


# # Extract_Features("/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/NeuralNetJournal_Dataset_Exp/Prototype-Based_Training/From_xDNN_Offline/Tests_by_Parts/Only_xDNN_Offline",
# Extract_Features("/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/Benchmark_Datasets/CBIS_DDSM_Dataset/Prototype-Based_Training/Test_2/xDNN_Offline",
#                   fe_layer = "last_conv", 
#                   config_dic = config_dic,
#                   )