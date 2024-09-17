from re import T
import numpy as np
import pandas as pd

from X_MAN.xDNN.utils import load_csv
from X_MAN.xDNN.utils import xDNN_run_vOptuna as xDNN_run
from X_MAN.xDNN.utils import Feature_Extraction_VGG16_PyTorch_Batch as featureExtraction

from numpy import genfromtxt

import copy 

import optuna

# (X_train, y_train, X_val, y_val) = load_csv.load()

import pickle

import time

# # X_vectors_path = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/NeuralNetJournal_Dataset_Exp/Prototype-Based_Training/PCA/Base_Line/Feature_Vectors_FromFlatten/Pickle/Original_25088comp/inputs.pkl"
# # X_vectors_path = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/COVID-QU-Ex_Dataset/Prototype-Based_Training/PCA/PCA_Emulator/Datasets/PCA_Using_Only_Train/Dataset_Pickle/inputs.pkl"
# X_vectors_path = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/COVID-QU-Ex_Dataset/Prototype-Based_Training/PCA/Test_Base_Line_FullDataset/Feature_Vectors_FromFlatten_v3/FeatureVectors.pkl"
# with open(X_vectors_path, "rb") as file:
#     # (X_train, X_val) = pickle.load(file)
#     (X_train, y_train, X_val, y_val) = pickle.load(file)
#     print("(X_train, X_test) loaded!!!")

# # featureVectors_dir = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/NeuralNetJournal_Dataset_Exp/Prototype-Based_Training/PCA/Base_Line/Feature_Vectors_FromFlatten"
# # featureVectors_dir = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/COVID-QU-Ex_Dataset/Prototype-Based_Training/PCA/Test_Base_Line_FullDataset/Feature_Vectors_FromFlatten_v3"

# featureVectors_dir = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/Benchmark_Datasets/CBIS_DDSM_Dataset/Prototype-Based_Training/Test_1/xDNN_Offline/Feature_Vectors_FromFlatten"

# y_train_file_path = featureVectors_dir + '/data_df_y_train.csv'
# y_test_file_path  = featureVectors_dir + '/data_df_y_val.csv'

# y_train = pd.read_csv(y_train_file_path, delimiter=',',header=None)
# y_val = pd.read_csv(y_test_file_path, delimiter=',',header=None)

# print ("###################### Data Loaded ######################")
# print("Data Shape:   ")        

# print("X train: ",X_train.shape)
# print("Y train: ",y_train.shape)

# print("X Val: ",X_val.shape)
# print("Y Val: ",y_val.shape)

# # input("\nPress Enter to continue...\n")

# models_dir = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/NeuralNetJournal_Dataset_Exp/Prototype-Based_Training/PCA/"
# models_dir = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/COVID-QU-Ex_Dataset/Prototype-Based_Training/PCA/Test_Base_Line_SubDataset/"
# models_dir = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/COVID-QU-Ex_Dataset/Prototype-Based_Training/PCA/Test_Base_Line_FullDataset/"
# models_dir = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/Benchmark_Datasets/CBIS_DDSM_Dataset/Prototype-Based_Training/Test_1/xDNN_Offline/"
models_dir = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/Benchmark_Datasets/CBIS_DDSM_Dataset/Prototype-Based_Training/FeatureVec_CMMD/xDNN_Offline/"

def objective(trial):

    trial_numComponents = trial.suggest_int("num_Components",3,1000,log=False)
    trial_CLAHE_clip_limit = trial.suggest_float("CLAHE_clip_limit",0.005, 0.05,log=True)
    # trial_numComponents = 30

    import json

    config_file_path = "/nfs/home/nvasconcellos.it/softLinkTests/X-MAN/Prototype-Based_Training/src/X_MAN/FineTuning/config/CMMD/parameters_Offline_PCA136c_PyTorch_GPU0_LowEpochs.json"
    config_dic = json.load(open(config_file_path))
    config_dic["Hardware"]["GPU"] = 0
    config_dic["PCA"]["V_Matrix"] = None
    config_dic["Pre-processing"] = True
    config_dic["CLAHE_clip_limit"] = trial_CLAHE_clip_limit

    model_dir = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/Benchmark_Datasets/CBIS_DDSM_Dataset/Prototype-Based_Training/FeatureVec_CMMD/xDNN_Offline"
    featureExtraction.Extract_Features(model_dir,
                      fe_layer = "last_conv", 
                      config_dic = config_dic,
                      )
    featureVectors_dir = model_dir + "/Feature_Vectors_FromFlatten"
    # Load the files, including features, images and labels.         
    X_train_file_path = featureVectors_dir + '/data_df_X_train.csv'
    y_train_file_path = featureVectors_dir + '/data_df_y_train.csv'
    X_val_file_path  = featureVectors_dir + '/data_df_X_test.csv'
    y_val_file_path  = featureVectors_dir + '/data_df_y_test.csv'
    # X_val_file_path  = featureVectors_dir + '/data_df_X_val.csv'
    # y_val_file_path  = featureVectors_dir + '/data_df_y_val.csv'

    X_train = genfromtxt(X_train_file_path, delimiter=',')
    y_train = pd.read_csv(y_train_file_path, delimiter=',',header=None)            

    # Print the shape of the data

    print ("###################### Data Loaded ######################")
    print("Data Shape:   ")        
    
    print("X train: ",X_train.shape)
    print("Y train: ",y_train.shape)
    
    X_val = genfromtxt(X_val_file_path, delimiter=',')        
    y_val = pd.read_csv(y_val_file_path, delimiter=',',header=None)

    print("X Val: ",X_val.shape)
    print("Y Val: ",y_val.shape)
    result = xDNN_run.run(models_dir = models_dir, 
        single_training=True, 
        model_name="Optuna/%s/xDDN_Offline_ProtFilt_OutliersFilt_PCA_%d_trial_%d" % (studyName,trial_numComponents,trial.number), 
        # model_name="Optuna/%s/Test" % (studyName),
        fe_layer = "last_conv", 
        num_PCA_Components = trial_numComponents, 
        feature_vectors =(copy.deepcopy(X_train),
                            copy.deepcopy(y_train),
                            copy.deepcopy(X_val),
                            copy.deepcopy(y_val)), 
                            # copy.deepcopy(X_train),
                            # copy.deepcopy(y_train)),
        xDNN_Subprocesses = ["Offline", "Prot_Filtering", "Outliers_Filtering"])
        # xDNN_Subprocesses = ["Evolving"])

    accuracy = result["AvrgAccuracy"]
    
    time.sleep(2)
    
    return (accuracy, sum(result["Prototypes"].values()), result["Train_Time"])

# studyName = 'NeuralNetJournal_Dataset_Prot-Based_Train_PCA_PyTorch_NewSplit'
# studyName = 'COVID-QU-Ex_Dataset_Prot-Based_Train_PCA_PyTorch_FVFromLastConv_xDNNEvolving'
# studyName = 'CBIS-DDSM_Dataset_Prot-Based_Train_PCA_PyTorch_FVFromLastConv_xDNNEvolvingRight'
# studyName = 'INbreast_Dataset_Prot-Based_Train_PCA_PyTorch_FVFromLastConv_xDNNOffline'
studyName = 'CMMD_Dataset_Prot-Based_Train_PCA_PyTorch_and_CLAHEClipLimit_FVFromLastConv_xDNNOffline_Test'
# studyName = 'Study_Extra_NeuralNetJournal_Dataset_Prot-Based_Train_PCA_PyTorch_NewSplit_Offline_FVFromLastConv'
study = optuna.create_study(directions=['maximize', "minimize", "minimize"],storage="sqlite:///xDNN_test.db",study_name=studyName, load_if_exists=True)

# for i in range(3, 1001):
# for i in range(3, 1001):
#     study.enqueue_trial({"num_Components": 	i})

study.optimize(objective, n_trials=998)

# objective(None)