import numpy as np
import pandas as pd

from X_MAN.xDNN.utils import load_csv
from X_MAN.xDNN.utils import xDNN_run_vCDE_Plot as xDNN_run

import copy 

import optuna

(X_train, y_train, X_val, y_val) = load_csv.load()

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

# # y_train_file_path = featureVectors_dir + '/data_df_y_train.csv'
# # y_test_file_path  = featureVectors_dir + '/data_df_y_val.csv'

# # y_train = pd.read_csv(y_train_file_path, delimiter=',',header=None)
# # y_val = pd.read_csv(y_test_file_path, delimiter=',',header=None)

# print ("###################### Data Loaded ######################")
# print("Data Shape:   ")        

# print("X train: ",X_train.shape)
# print("Y train: ",y_train.shape)

# print("X Val: ",X_val.shape)
# print("Y Val: ",y_val.shape)

# # input("\nPress Enter to continue...\n")

models_dir = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/NeuralNetJournal_Dataset_Exp/Prototype-Based_Training/PCA/"
# models_dir = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/COVID-QU-Ex_Dataset/Prototype-Based_Training/PCA/Test_Base_Line_SubDataset/"
# models_dir = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/COVID-QU-Ex_Dataset/Prototype-Based_Training/PCA/Test_Base_Line_FullDataset/"

def objective(trial):

    # trial_bandwidth = trial.suggest_float("bandwidth",0.64,0.82,log=True)
    trial_bandwidth = trial.suggest_int("bandwidth",500,2000,log=True)
    # trial_numComponents = 2

    result = xDNN_run.run(models_dir = models_dir, 
        single_training=True, 
        # model_name="Optuna/%s/xDDN_Offline_ProtFilt_OutliersFilt_PCA_%d_trial_%d" % (studyName,trial_numComponents,trial.number), 
        model_name="Optuna/%s/Test" % (studyName),
        fe_layer = "last_conv",
        feature_vectors =(copy.deepcopy(X_train),
                            copy.deepcopy(y_train),
                            copy.deepcopy(X_val),
                            copy.deepcopy(y_val)),
        Nomalization = None,
        bandwidth = trial_bandwidth,
        xDNN_Subprocesses = ["Mean-Shift"], 
        classifier = "kNN", 
        k_neighbors = 1)

    accuracy = result["AvrgAccuracy"]
    
    time.sleep(2)
    
    return (accuracy, sum(result["Prototypes"].values()), result["Train_Time"])

# studyName = 'NeuralNetJournal_Dataset_Prot-Based_Train_PCA_PyTorch_NewSplit'
studyName = 'NeuralNetJournal_Dataset_NewSplit_MeanShift_Bandwidth_Study_FVFromLastConv_WithoutNormalization'
# studyName = 'Study_Extra_NeuralNetJournal_Dataset_Prot-Based_Train_PCA_PyTorch_NewSplit_Offline_FVFromLastConv'
study = optuna.create_study(directions=['maximize', "minimize", "minimize"],storage="sqlite:///xDNN_test.db",study_name=studyName, load_if_exists=True)

study.optimize(objective, n_trials=100)

# objective(None)