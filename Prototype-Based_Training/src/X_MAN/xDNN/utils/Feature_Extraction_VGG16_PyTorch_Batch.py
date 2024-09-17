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

from re import T
from tqdm import tqdm
import torch
import torchvision.models as models
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import sys
sys.path.insert(0, '/nfs/home/nvasconcellos.it/softLinkTests/X-MAN/Prototype-Based_Training/src')
from X_MAN.Models.VGG16.Model.Converted_VGG16 import Converted_VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

import copy
from X_MAN.FineTuning.utils.CMMD_preprocessing import preprocess_image

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, mask_paths=None, transform=None, **kwargs):
        self.image_paths = image_paths
        self.labels = labels
        self.mask_paths = mask_paths
        self.transform = transform
        if kwargs.get("preprocess") is not None:
            self.preprocess = kwargs.get("preprocess")
            self.CLAEH_clip_limit = kwargs.get("CLAHE_clip_limit")
        else:
            self.preprocess = False

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        if not self.preprocess:
          img = image.load_img(img_path, target_size=(224, 224))
          x = image.img_to_array(img)
        else:
          img = image.load_img(img_path)
          x = image.img_to_array(img)
          print("CLAHE_clip_limit: " + str(self.CLAEH_clip_limit))
          x = preprocess_image(x.astype(np.int32), new_height=224, CLAHE_clip_limit=self.CLAEH_clip_limit).astype(x.dtype)
          print("Preprocessed Image Shape: " + str(x.shape))
           
        if self.mask_paths is not None:
            mask_path = self.mask_paths[idx]
            mask = image.load_img(mask_path, target_size=(224, 224))
            mask = image.img_to_array(mask)
            x = x * (mask / 255.0)
        
        x = preprocess_input(x)
        label = self.labels[idx]
        
        return torch.from_numpy(x.copy()), label

def Extract_Features(model_dir, fe_layer="last_fc", config_dic: dict = {}, **kwargs):
    
    # if config_dic.get("Hyperparameters") is not None and config_dic["Hyperparameters"].get("batch_size") is not None:
    #   batch_size = config_dic["Hyperparameters"]["batch_size"]
    # else:
    #   batch_size = 32
    batch_size = 2
    print("Batch Size: " + str(batch_size))
    
    if kwargs.get("model") is not None:
        model = kwargs.get("model")
    else:
        weights_path = model_dir + "/checkpoints/best_model.pt"
        if not os.path.exists(weights_path):
            weights_path = None
        
        model = Converted_VGG16(weights_path, fe_layer=fe_layer,
                                pca_emulator=config_dic["PCA"]["PCA_Emulator"],
                                pca_emul_weights_path=config_dic["PCA"]["Weights"],
                                pca_origNumComp=config_dic["PCA"]["Original_Num_Comp"],
                                pca_newNumComp=config_dic["PCA"]["New_Num_Comp"])
        
        model.eval()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)
  
    device = next(model.parameters()).device

    if config_dic.get("Dataset") is not None and config_dic["Dataset"].get("Mask_dir") is not None:
        mask_dir = config_dic["Dataset"]["Mask_dir"]
    else:
        mask_dir = None

    V_matrix = torch.load(config_dic["PCA"]["V_Matrix"]).to(torch.float32) if config_dic.get("PCA") is not None and config_dic["PCA"].get("V_Matrix") is not None else None
    # V_matrix = None 
    print("V_Matrix Shape: " + str(V_matrix.shape) if V_matrix is not None else "V_Matrix: None") 

    datas_dir = config_dic["Dataset"]["Directory"] + '/'

    fv_dic = {}

    for set_dir in os.listdir(datas_dir):
      data_dir = os.path.join(datas_dir, set_dir)
      classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

      images = []
      labels = []
      mask_paths = []

      for class_idx, class_name in enumerate(classes):
          class_path = os.path.join(data_dir, class_name)
          files = os.listdir(class_path)
          
          for file in files:
              img_path = os.path.join(class_path, file)
              images.append(img_path)
              labels.append(class_idx)
              
              if mask_dir is not None:
                  mask_class_path = os.path.join(mask_dir, set_dir, class_name, "lung masks")
                  mask_path = os.path.join(mask_class_path, file)
                  mask_paths.append(mask_path)

      # Create dataset and DataLoader
      dataset = CustomDataset(images, labels, mask_paths if mask_dir is not None else None, 
                              preprocess=config_dic.get("Pre-processing"), CLAHE_clip_limit=config_dic.get("CLAHE_clip_limit"))
      data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

      all_features = []
      all_image_labels = []

      print("Extracting features from " + set_dir + " set:\n")
      with torch.no_grad():
          for batch_images, batch_labels in tqdm(data_loader):
              # batch_images = batch_images.permute(0, 3, 1, 2).float().to(device)
              batch_images = batch_images.float().to(device)
              features = model(batch_images).cpu()
              
              if V_matrix is not None:
                  features = torch.matmul(features, V_matrix)  # PCA Reduction
              
              all_features.append(features.numpy())
              all_image_labels.append(batch_labels.numpy())

      np_batch = np.concatenate(all_features)
      np_labels = np.concatenate(all_image_labels).reshape(-1, 1)
      np_images_labels = np.hstack((np.array(images).reshape(-1, 1), np_labels))
      
      fv_dic[set_dir] = (np_batch, np_images_labels)

    for set_name in fv_dic.keys():
      print(set_name + " X Shape: " + str(fv_dic[set_name][0].shape))
      print(set_name + " y Shape: " + str(fv_dic[set_name][1].shape))

    data_df_dict = {}
    for set_name in fv_dic.keys():
      data_df_dict[set_name] = {"X": pd.DataFrame(fv_dic[set_name][0]), "y": pd.DataFrame(fv_dic[set_name][1])}


    if fe_layer == "last_conv":    
      output_dir = model_dir + '/Feature_Vectors_FromFlatten/'
    else:
      output_dir = model_dir + '/Feature_Vectors/'
      
    if kwargs.get("aux_dataset") is True:
      output_dir = output_dir + config_dic["Dataset"]["Dataset_Name"] + "/"

    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    # Saving the data as CSV
    for set_name in fv_dic.keys():
        pd.DataFrame(fv_dic[set_name][0]).to_csv(os.path.join(output_dir, f"data_df_X_{set_name}.csv"), header=False, index=False)
        pd.DataFrame(fv_dic[set_name][1]).to_csv(os.path.join(output_dir, f"data_df_y_{set_name}.csv"), header=False, index=False)

    if kwargs.get("aux_dataset") is not True:
      return fv_dic["train"][0], pd.DataFrame(fv_dic["train"][1]), fv_dic["val"][0], pd.DataFrame(fv_dic["val"][1])

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
    
  if config_dic.get("Dataset") is not None and config_dic["Dataset"].get("Auxiliary_Datasets") is not None:
    auxiliary_datasets = config_dic["Dataset"]["Auxiliary_Datasets"]
    main_dataset_name = config_dic["Dataset"]["Dataset_Name"]
    main_dataset_dir = config_dic["Dataset"]["Directory"]
    for aux_dataset in auxiliary_datasets:
      config_dic["Dataset"]["Dataset_Name"] = aux_dataset["Dataset_Name"]
      config_dic["Dataset"]["Directory"] = aux_dataset["Directory"]
      print("Extracting features from " + aux_dataset["Dataset_Name"] + " dataset\n")
      Extract_Features(trainings_dir, fe_layer=fe_layer, config_dic = config_dic, aux_dataset=True)

    config_dic["Dataset"]["Dataset_Name"] = main_dataset_name
    config_dic["Dataset"]["Directory"] = main_dataset_dir
    
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
# # config_file_path = "/nfs/home/nvasconcellos.it/softLinkTests/X-MAN/Prototype-Based_Training/src/X_MAN/FineTuning/config/DDSM_Dataset/parameters_Offline_PCA_PyTorch_GPU1_LowEpochs.json"
# # config_file_path = "/nfs/home/nvasconcellos.it/softLinkTests/X-MAN/Prototype-Based_Training/src/X_MAN/FineTuning/config/DDSM_Dataset/parameters_Offline_PCA287c_PyTorch_GPU0_LowEpochs_AuxDatasets_ROIs.json"
# # config_file_path = "/nfs/home/nvasconcellos.it/softLinkTests/X-MAN/Prototype-Based_Training/src/X_MAN/FineTuning/config/INbreast/parameters_Offline_PCA287c_PyTorch_GPU0_LowEpochs.json"
# config_file_path = "/nfs/home/nvasconcellos.it/softLinkTests/X-MAN/Prototype-Based_Training/src/X_MAN/FineTuning/config/NeuralNetJournal_Dataset_Exp/parameters_Offline_PCA_PyTorch_GPU1_LowEpochs_xDNNEvolving.json"
# config_dic = json.load(open(config_file_path))
# config_dic["Hardware"]["GPU"] = 0
# config_dic["PCA"]["V_Matrix"] = None
# # config_dic["Dataset"]["Auxiliary_Datasets"] = None


# # Extract_Features("/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/NeuralNetJournal_Dataset_Exp/Prototype-Based_Training/From_xDNN_Offline/Tests_by_Parts/Only_xDNN_Offline",
# # Extract_Features("/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/Benchmark_Datasets/CBIS_DDSM_Dataset/Prototype-Based_Training/Test_ROIs/xDNN_Offline",
# # Extract_Features("/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/Benchmark_Datasets/CBIS_DDSM_Dataset/Prototype-Based_Training/FeatureVec_INbreast/xDNN_Offline",
# Extract_Features("/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/NeuralNetJournal_Dataset_Exp/Prototype-Based_Training/From_xDNN_Offline/Tests_by_Parts/Traditional_FineTuning_NewSplit/Optuna/Study-Extra_NeuralNetJournal_Dataset_Traditional_FineTuning/Trial_Manual_1_FixBatchSizeAndLearningRate/Base_Line/T1",
#                   fe_layer = "last_fc", 
#                   config_dic = config_dic,
#                   )