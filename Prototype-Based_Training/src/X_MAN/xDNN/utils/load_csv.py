from numpy import genfromtxt
import numpy as np
import pandas as pd

# import sys
# sys.path.insert(0, '/nfs/home/nvasconcellos.it/softLinkTests/X-MAN/Prototype-Based_Training/src')
# from X_MAN.xDNN.utils import xDNN_run_vOptuna as xDNN_run

def load():
    # featureVectors_dir = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/COVID-QU-Ex_Dataset/Prototype-Based_Training/Test_Base_Line_FullDataset/Feature_Vectors_FromFlatten"
    featureVectors_dir = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/COVID-QU-Ex_Dataset/Prototype-Based_Training/PCA/Test_Base_Line_FullDataset/Feature_Vectors_FromFlatten_v3"
    # featureVectors_dir = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/COVID-QU-Ex_Dataset/Prototype-Based_Training/PCA/Test_Base_Line_SubDataset/Feature_Vectors_FromFlatten"
    # featureVectors_dir = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/COVID-QU-Ex_Dataset/Prototype-Based_Training/PCA/PCA_Emulator/Datasets/Tran_Set_SubDivision/Feature_Vectors_FromFlatten"
    
    # featureVectors_dir = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/NeuralNetJournal_Dataset_Exp/Prototype-Based_Training/PCA/Base_Line/Feature_Vectors_FromFlatten"
    # featureVectors_dir = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/NeuralNetJournal_Dataset_Exp/Prototype-Based_Training/From_xDNN_Offline/Tests_by_Parts/xDNN_Offline_And_Traditional_FineTuning/Optuna/Study-Extra_NeuralNetJournal_Dataset_Traditional_FineTuning/Trial_Manual_2_FixBatchSizeAndLearningRate/Base_Line/T1/Feature_Vectors"
    # featureVectors_dir = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/NeuralNetJournal_Dataset_Exp/Prototype-Based_Training/From_xDNN_Offline/Tests_by_Parts/xDNN_Offline_And_Traditional_FineTuning/Optuna/Study-Extra_NeuralNetJournal_Dataset_Traditional_FineTuning/Trial_Manual_2_FixBatchSizeAndLearningRate/Base_Line/T1/Feature_Vectors_FromFlatten"
    
    # featureVectors_dir = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/NeuralNetJournal_Dataset_Exp/Prototype-Based_Training/From_xDNN_Offline/Tests_by_Parts/Traditional_FineTuning_NewSplit/Optuna/Study-Extra_NeuralNetJournal_Dataset_Traditional_FineTuning/Trial_Manual_1_FixBatchSizeAndLearningRate/Base_Line/T1/Feature_Vectors"
    # featureVectors_dir = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/NeuralNetJournal_Dataset_Exp/Prototype-Based_Training/From_xDNN_Offline/Optuna/NeuralNetJournal_Prot-Based_TrainWithPCA_Torch_199c_CDEAlpha_Study_LowEpochs_New_Split_FVFromLastConv_xDNNOffline_v3/Trial_0/Base_Line/T1/Feature_Vectors_FromFlatten"

    # featureVectors_dir = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/Benchmark_Datasets/CBIS_DDSM_Dataset/Prototype-Based_Training/Test_1/xDNN_Offline/Feature_Vectors_FromFlatten"
    # featureVectors_dir = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/Benchmark_Datasets/CBIS_DDSM_Dataset/Prototype-Based_Training/Test_ROIs/xDNN_Offline/Feature_Vectors_FromFlatten"
    
    featureVectors_dir = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/Benchmark_Datasets/CBIS_DDSM_Dataset/Prototype-Based_Training/FeatureVec_INbreast/xDNN_Offline/Feature_Vectors_FromFlatten"

    X_train_file_path = featureVectors_dir + '/data_df_X_train.csv'
    y_train_file_path = featureVectors_dir + '/data_df_y_train.csv'
    X_test_file_path  = featureVectors_dir + '/data_df_X_val.csv'
    y_test_file_path  = featureVectors_dir + '/data_df_y_val.csv'

    X_train = genfromtxt(X_train_file_path, delimiter=',')
    y_train = pd.read_csv(y_train_file_path, delimiter=',',header=None)

    print ("###################### Data Loaded ######################")
    print("Data Shape:   ")        

    print("X train: ",X_train.shape)
    print("Y train: ",y_train.shape)

    X_test = genfromtxt(X_test_file_path, delimiter=',')        
    y_test = pd.read_csv(y_test_file_path, delimiter=',',header=None)

    print("X Val: ",X_test.shape)
    print("Y Val: ",y_test.shape)

    return (X_train, y_train, X_test, y_test)

# run(models_dir=models_dir,single_training=True,model_name="xDDN_Offline_ProtFilt_OutliersFilt_PCA_4096",fe_layer = "last_conv",num_PCA_Components = 4096, feature_vectors = (X_train, y_train, X_test, y_test),xDNN_Subprocesses = ["Offline", "Prot_Filtering", "Outliers_Filtering"])