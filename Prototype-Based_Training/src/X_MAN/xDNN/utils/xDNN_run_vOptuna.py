#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Please cite:
Angelov, P., & Soares, E. (2020). Towards explainable deep neural networks (xDNN). Neural Networks.

"""

###############################################################################
from multiprocessing.connection import wait
from pyexpat import model
from re import T, X
from networkx import karate_club_graph
import pandas as pd
import numpy as np

# from xDNN_class_MODIFIED import *
import sys
sys.path.insert(0, '/nfs/home/nvasconcellos.it/softLinkTests/X-MAN/Prototype-Based_Training/src')
from X_MAN.xDNN.Models.xDNN_class_Offlinev1 import *
# from xDNN_class_Offlinev1 import *
# from xDNN_class_Offlinev2 import *
# from xDNN_class_RepMatLab import *
from numpy import genfromtxt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import time

import os
import datetime
import seaborn as sns
from math import pi

import math
import pickle

resultsModel_dir = ''
folder_dir = ''

# Functions
def displayMetrics(x_labels, y_values):
    
    fig, ax = plt.subplots()
    tab2 = [['%.5f' % y] for x, y in zip(x_labels, y_values)]
    rcolors = plt.cm.BuPu(np.full(len(y_values), 0.1))
    ccolors = plt.cm.BuPu(np.full(1, 0.1))
    

    ytable = plt.table(cellText=tab2, rowLabels=x_labels, colLabels=['Metrics'], rowColours=rcolors,
                        colWidths=[.3]*5, loc='center', colColours=ccolors)
    ytable.set_fontsize(24)
    ytable.scale(1, 4)
    plt.gcf().set_size_inches(8, 6)
    plt.axis('off')    
    
    
    figure_name = resultsModel_dir + '/Performance_Metrics.png'
    plt.savefig(figure_name)
   
def displayConfMatrix(cf_matrix, class_names): 
    group_names = ['True Neg','False Pos','False Neg', 'True Pos']

    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]

    # group_percentages = ["{0:.2%}".format(value) for value in
    #                     cf_matrix.flatten()/np.sum(cf_matrix)]
    group_percentages = []
    
    for i in range(cf_matrix.shape[0]):
        for value in (cf_matrix[i].flatten()/np.sum(cf_matrix[i])):
            group_percentages.append("{0:.2%}".format(value))

    # labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
    #         zip(group_names,group_counts,group_percentages)]
    labels = [f"{v2}\n{v3}" for v2, v3 in
            zip(group_counts,group_percentages)]

    num_classes = len(class_names)
    labels = np.asarray(labels).reshape(num_classes, num_classes)

    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')

    ax.set_title('Confusion Matrix with labels\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')

    # ## Ticket labels - List must be in alphabetical order
    # ax.xaxis.set_ticklabels(['Non-Covid', 'Covid'])
    # ax.yaxis.set_ticklabels(['Non-Covid', 'Covid'])
    ax.xaxis.set_ticklabels(class_names)
    ax.yaxis.set_ticklabels(class_names)

    ## Display the visualization of the Confusion Matrix.  
    figure_name = resultsModel_dir + '/Confution_Matrix.png'
    plt.gcf().set_size_inches(8, 6)
    plt.savefig(figure_name)

def displayImages(origImage, protNames, predLabel, img_folder_dir):
    labelNames = ('Covid', 'Non-Covid')
    
    # setting values to rows and column variables
    rows = 2
    columns = int((len(protNames[0][0]) + 2)/rows)
    
    # reading images
    if origImage[0] == 'C':
        folder = 'COVID/'
    else:
        folder = 'non-COVID/'        
    originalImage_path = folder_dir + '/input/sarscov2-ctscan-dataset/' + folder + origImage    
    images_paths = ([originalImage_path] + [folder_dir + '/input/sarscov2-ctscan-dataset/COVID/' + prot for prot in protNames[0][0]], 
                    [originalImage_path] + [folder_dir + '/input/sarscov2-ctscan-dataset/non-COVID/' + prot for prot in protNames[1][0]])        
    titles = (['Original Image'] + ['Prototype ' + str(i) + ' (' + str(dst) + ')' for i, dst in zip(range(1, len(protNames[0][0]) + 1), protNames[0][1])],
              ['Original Image'] + ['Prototype ' + str(i) + ' (' + str(dst) + ')' for i, dst in zip(range(1, len(protNames[1][0]) + 1), protNames[1][1])])
    
    
    fig, ax = plt.subplots()
    # turn off the axes
    ax.set_axis_off()
    fig.set_size_inches(12, 7)
    fig.suptitle('Selected Class', fontsize=16)
    j = 0
    for i in range(rows*columns): 
        if i == columns:
            j = 1
            continue       
        # Adds a subplot at the next position
        fig.add_subplot(rows, columns, i+1)
        
        # showing image
        
        # img = image.load_img(images_paths[predLabel][i-j])
        # plt.imshow(img)
        plt.axis('off')
        plt.title(titles[predLabel][i-j])
    figure_name = img_folder_dir + '/Selected_Class.png'
    plt.savefig(figure_name)
    
    fig, ax = plt.subplots()
    # turn off the axes
    ax.set_axis_off()
    fig.set_size_inches(12, 7)
    fig.suptitle('Non Selected Class', fontsize=16)
    j=0
    for i in range(rows*columns): 
        if i == columns:
            j=1
            continue       
        # Adds a subplot at the next position
        fig.add_subplot(rows, columns, i+1)
        
        # showing image
        
        # img = image.load_img(images_paths[1-predLabel][i-j], target_size=(224, 224))
        # plt.imshow(img)
        plt.axis('off')
        plt.title(titles[1-predLabel][i-j])
        
    figure_name = img_folder_dir + '/Non-Selected_Class.png'
    plt.savefig(figure_name)  
    
def displayDistancesByProt(predLabel, if_thenRules, image_name, img_folder_dir):
    labels = ['Prototype 1', 'Prototype 2', 'Prototype 3', 'Prototype 4']
    selected_class = if_thenRules[predLabel][1]
    non_selected_class = if_thenRules[1-predLabel][1]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, selected_class, width, label=labelNames[predLabel] + '(Selected Class)')
    rects2 = ax.bar(x + width/2, non_selected_class, width, label=labelNames[1-predLabel])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Feature Vector Distance')
    ax.set_xlabel('Image Name: ' + image_name)
    
    ax.set_title('Distances (or difference degrees) by prototype and class')
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()
    plt.gcf().set_size_inches(12, 7)
    plt.legend(loc='lower right', title='Classes')   
    figure_name = img_folder_dir + '/Distances_By_Prototype.png'
    plt.savefig(figure_name) 
    
def displayBoxPlot(data):
    fig = plt.figure(figsize =(10, 7))
 
    # Creating plot
    plt.boxplot(data)
    
    # show plot
    plt.show()
    
# # models_dir = '/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/NeuralNetJournal_Dataset_Exp/FineTuningProgr_Train72.2_Test27.8/Base_Line_VGG16/'
models_dir = '/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/NeuralNetJournal_Dataset_Exp/Optuna/FineTuningProgr_Train72.2_Test27.8/Test_All_Layers_WithSeed_XavierUniformInitializer/From_fc2_EachBlock_Nicolax_Trial3/FineTuning_block1_conv1_To_End/T2_FineTuning_block1_conv1_To_End_VGG16_model_2023-01-1704_26_26.786295/'

# # featureVectors_dir = models_dir + "Feature_Vectors"
# featureVectors_dir = models_dir + "Feature_Vectors_FromFlatten"

# # Load the files, including features, images and labels.         
# X_train_file_path = featureVectors_dir + '/data_df_X_train.csv'
# y_train_file_path = featureVectors_dir + '/data_df_y_train.csv'
# X_test_file_path  = featureVectors_dir + '/data_df_X_test.csv'
# y_test_file_path  = featureVectors_dir + '/data_df_y_test.csv'

# X_train = genfromtxt(X_train_file_path, delimiter=',')
# y_train = pd.read_csv(y_train_file_path, delimiter=';',header=None)
# X_test = genfromtxt(X_test_file_path, delimiter=',')
# y_test = pd.read_csv(y_test_file_path, delimiter=';',header=None)

# # X_train_file_path = r'data_df_X_train_covid.csv'
# # y_train_file_path = r'data_df_y_train_covid.csv'
# # X_test_file_path = r'data_df_X_test_covid.csv'
# # y_test_file_path = r'data_df_y_test_covid.csv'
    
# # X_train = genfromtxt(X_train_file_path, delimiter=',')
# # y_train = pd.read_csv(y_train_file_path, delimiter=',',header=None)
# # X_test = genfromtxt(X_test_file_path, delimiter=',')
# # y_test = pd.read_csv(y_test_file_path, delimiter=',',header=None)


# # Print the shape of the data

# print ("###################### Data Loaded ######################")
# print("Data Shape:   ")
# print("X train: ",X_train.shape)
# print("Y train: ",y_train.shape)
# print("X test: ",X_test.shape)
# print("Y test: ",y_test.shape)

# # Images are the images file names
# # Labels are the output representation of the class "Covid" -> "0" and "Non-Covid" -> "1"
# # X set have the feature vector of 4096 high-level characteristics of each image  
# pd_y_train_labels = y_train[1]
# pd_y_train_images = y_train[0]

# pd_y_test_labels = y_test[1]
# pd_y_test_images = y_test[0]



# # Convert Pandas to Numpy
# y_train_labels = pd_y_train_labels.to_numpy()
# y_train_images = pd_y_train_images.to_numpy()

# y_test_labels = pd_y_test_labels.to_numpy()
# y_test_images = pd_y_test_images.to_numpy()

def run(models_dir=models_dir, outputDir= None, **kwargs):
    trainings = next(os.walk(models_dir))[1]
    trainings.sort()
    accuracy_accum = 0
    numOfPrototypes_mean = {}
    numTrainings = 0
    
    if kwargs.get('single_training') is not None:
        single_training = kwargs.get('single_training')
    else:
        single_training = False
    
    if kwargs.get('config_dic') is not None:
        config_dic = kwargs.get('config_dic')
    else:
        config_dic = None
        
    if kwargs.get('inference_TestSet') is not None:
        inference_TestSet = kwargs.get('inference_TestSet')
    else:
        inference_TestSet = False
    
    for model_dir in trainings:
        print(model_dir)
        
        # if not model_dir.__contains__("Base_Line_GoogLeNet_v3"):
        # if not model_dir.__contains__("Base_Line_VGG16"):
        # if not model_dir.__contains__("ArxivDataset_VGG16_80Train_20Test"):
        # if not model_dir.__contains__("VGG16_Prot_Training_Trial_14"):
        #     continue
               
        
        # Directories
        global folder_dir
        folder_dir = '/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning'
        if single_training:
            model_dir = ""
            models_dir = models_dir[:-1]
        
        if kwargs.get('fe_layer') == "last_fc":
            featureVectors_folder = '/Feature_Vectors'
            # featureVectors_folder = '/Feature_Vectors_PCA289c_xDNNOffline'
        elif kwargs.get('fe_layer') == "last_conv":
            featureVectors_folder = '/Feature_Vectors_FromFlatten'
            # featureVectors_folder = '/Feature_Vectors_FromFlatten_PCA402c_xDNNEvolving'
            # featureVectors_folder = '/Feature_Vectors_FromFlatten_Test'
        elif kwargs.get('fe_layer') is None:
            # featureVectors_folder = '/Feature_Vectors'
            # featureVectors_folder = '/Feature_Vectors_FromMatLab_Rep'
            # featureVectors_folder = '/Feature_Vectors_FromMatLabSplit_Keras'
            # featureVectors_folder = '/Feature_Vectors_FromMatLabSplit_PyTorch'
            # featureVectors_folder = '/Feature_Vectors_FromMatLabSplit_PyTorch_FromKeras_ConvertedFromOnnx'
            featureVectors_folder = '/Feature_Vectors_FromFlatten'
        else:
            raise ValueError("Unknown feature layer: " + kwargs.get('fe_layer'))
        
        featureVectors_dir = models_dir + model_dir + featureVectors_folder
            
        print("featureVectors_dir: " + featureVectors_dir)
        
        if kwargs.get('model_name') is not None:
            model_name = "/" + kwargs.get('model_name')
        else:
            # model_name = '/xDDN_RefactEuclideanDis_ProtFilteringTest_'+ model_dir
            # model_name = '/xDDN_RefactEuclideanDis_NormalizationByFeature_'+ model_dir
            # model_name = '/xDDN_RefactEuclideanDis_OrigExpRepTest_FVFromMatLab_'+ model_dir
            # model_name = '/xDDN_RefactEuclideanDis_OrigExpRepTest_'+ model_dir
            # model_name = '/xDDN_Offline_EuclideanDis_'+ model_dir
            # model_name = '/xDDN_RepAlg_OnlyValProcess_'+ model_dir
            model_name = '/xDDN_Offline_ProtFilt_OutliersFilt_'+ model_dir
        
        global resultsModel_dir
        if outputDir != None:            
            resultsModel_dir = outputDir + model_dir + model_name
        else:
            resultsModel_dir = models_dir + model_dir + model_name
        
        # resultsModel_dir = models_dir + model_name
        if not os.path.exists(resultsModel_dir):
            os.makedirs(resultsModel_dir)
            
        if kwargs.get('feature_vectors') is not None:
            X_train, y_train, X_test, y_test = kwargs.get('feature_vectors')
        else:
            # Load the files, including features, images and labels.         
            X_train_file_path = featureVectors_dir + '/data_df_X_train.csv'
            y_train_file_path = featureVectors_dir + '/data_df_y_train.csv'
            if inference_TestSet:
                if kwargs.get("aux_dataset_name") is not None:
                    featureVectors_dir = featureVectors_dir + "/" + kwargs.get("aux_dataset_name")
                    print("Using auxiliary dataset: " + kwargs.get("aux_dataset_name"))
                    print("featureVectors_dir: " + featureVectors_dir)
                X_test_file_path  = featureVectors_dir + '/data_df_X_test.csv'
                y_test_file_path  = featureVectors_dir + '/data_df_y_test.csv'
            else:                
                X_test_file_path  = featureVectors_dir + '/data_df_X_val.csv'
                y_test_file_path  = featureVectors_dir + '/data_df_y_val.csv'

            # X_train = genfromtxt(X_train_file_path, delimiter=',')
            # y_train = pd.read_csv(y_train_file_path, delimiter=';',header=None)
            # X_test = genfromtxt(X_test_file_path, delimiter=',')
            # y_test = pd.read_csv(y_test_file_path, delimiter=';',header=None)

            # X_train_file_path = r'data_df_X_train_covid.csv'
            # y_train_file_path = r'data_df_y_train_covid.csv'
            # X_test_file_path = r'data_df_X_test_covid.csv'
            # y_test_file_path = r'data_df_y_test_covid.csv'            
            
            X_train = genfromtxt(X_train_file_path, delimiter=',')
            y_train = pd.read_csv(y_train_file_path, delimiter=',',header=None)            

            # Print the shape of the data

            print ("###################### Data Loaded ######################")
            print("Data Shape:   ")        
            
            print("X train: ",X_train.shape)
            print("Y train: ",y_train.shape)
            
            X_test = genfromtxt(X_test_file_path, delimiter=',')        
            y_test = pd.read_csv(y_test_file_path, delimiter=',',header=None)

            print("X Val: ",X_test.shape)
            print("Y Val: ",y_test.shape)

        if kwargs.get('num_PCA_Components') is not None:
            from sklearn.decomposition import PCA
            import torch
            import random

            numComponents = kwargs.get('num_PCA_Components')
            train_size = X_train.shape[0]
            n_features = X_train.shape[1]
            # name_features = ["eixo_"+str(i) for i in range(n_features)]
            # # dictionary = dict(zip(name_features, np.hstack((X_train.transpose(), X_test.transpose()))))
            # dictionary = dict(zip(name_features, X_train.transpose()))
            # pca = PCA(n_components=numComponents, random_state=123)
            # # X_train_and_test = pca.fit_transform(pd.DataFrame.from_dict(dictionary))
            # # X_train = X_train_and_test[0:train_size]
            # # X_test = X_train_and_test[train_size:]
            # pca = pca.fit(pd.DataFrame.from_dict(dictionary))
            # X_train = pca.transform(pd.DataFrame.from_dict(dictionary))
            # dictionary = dict(zip(name_features, X_test.transpose()))
            # X_test = pca.transform(pd.DataFrame.from_dict(dictionary))

            # Reproducibility (Deterministic) Setup
            pca_seed = 123
            def seed_worker(worker_id):
                random.seed(pca_seed)
                torch.manual_seed(pca_seed)
                torch.cuda.manual_seed(pca_seed)
                np.random.seed(pca_seed)
            
            g = torch.Generator()
            g.manual_seed(pca_seed)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.backends.cudnn.benchmark = False
            # torch.backends.cudnn.benchmark = True
            torch.use_deterministic_algorithms(True)
            random.seed(pca_seed)
            torch.manual_seed(pca_seed)
            torch.cuda.manual_seed(pca_seed)
            np.random.seed(pca_seed)

            X_train_torch = torch.from_numpy(X_train.copy())
            X_test_torch = torch.from_numpy(X_test.copy())

            # (U, S, V) = torch.pca_lowrank(X_train_torch, q=min(n_features, train_size))            
            # global V_Matrix
            # k = numComponents
            # V = V_Matrix[:, :k]

            if kwargs.get('Use_calculated_V_Matrix') is True:
                # V = torch.load(resultsModel_dir + "/PCA_V_matrix.pt")
                V = torch.load(os.path.dirname(resultsModel_dir) + "/xDDN_Offline_OutliersFilt" + "/PCA_V_matrix.pt")
            else:
                (U, S, V) = torch.pca_lowrank(X_train_torch, q=numComponents)

            X_train = torch.matmul(X_train_torch, V).numpy()
            X_test = torch.matmul(X_test_torch, V).numpy()
            
            torch.save(V, resultsModel_dir + "/PCA_V_matrix.pt")


            print ("###################### PCA ######################")
            print("Data Shape:   ")        
            
            print("X train: ",X_train.shape)
            print("Y train: ",y_train.shape)

            print("X Val: ",X_test.shape)
            print("Y Val: ",y_test.shape)


        # Images are the images file names
        # Labels are the output representation of the class "Covid" -> "0" and "Non-Covid" -> "1"
        # X set have the feature vector of 4096 high-level characteristics of each image  
        # pd_y_train_labels = y_train[1]
        # # pd_y_train_labels = y_train[0]
        # pd_y_train_images = y_train[0]

        # pd_y_test_labels = y_test[1]
        # # pd_y_test_labels = y_test[0]
        # pd_y_test_images = y_test[0]

        # # Convert Pandas to Numpy
        # y_train_labels = pd_y_train_labels.to_numpy()
        # y_train_images = pd_y_train_images.to_numpy()

        # y_test_labels = pd_y_test_labels.to_numpy()
        # y_test_images = pd_y_test_images.to_numpy()

        # Convert Pandas to Numpy
        y_train_labels = y_train[1].to_numpy(np.int32)
        y_train_images = y_train[0].to_numpy()

        y_test_labels = y_test[1].to_numpy(np.int32)
        y_test_images = y_test[0].to_numpy()
        
        train_size = X_train.shape[0]
        numFeatures = X_train.shape[1]
        test_size = y_test.shape[0]
        
        # Sample num_test_samples samples from the test set
        if config_dic is not None and config_dic["Dataset"].get('num_test_samples') is not None:
            num_test_samples = config_dic["Dataset"].get('num_test_samples')
            sample_seed = config_dic["Dataset"].get('sample_seed')
            import random
            random.seed(sample_seed)
            test_samples = random.sample(range(test_size), num_test_samples)
            X_test = X_test[test_samples]
            y_test_labels = y_test_labels[test_samples]
            y_test_images = y_test_images[test_samples]
            test_size = num_test_samples
            
            print("New test set size: ", X_test.shape)



        # Model Learning
        Input1 = {}

        Input1['Images'] = y_train_images
        Input1['Features'] = X_train
        Input1['Labels'] = y_train_labels

        Mode1 = 'Learning'
             
        
        # Define the config_dic for xDNN
        if config_dic is None:
            config_dic = {}

        start = time.time()
        if "initialRadius" in kwargs.keys():
            config_dic["initial_Radius"]=kwargs["initialRadius"]
        else:
            config_dic["initial_Radius"]= 30
        
        if "c_W_k" in kwargs.keys():
            config_dic["c_W_k"]=kwargs["c_W_k"]
        else:
            config_dic["c_W_k"]=0.5
        
        if type(kwargs.get('xDNN_Subprocesses')) is list:
            config_dic["xDNN_Subprocesses"] = kwargs.get('xDNN_Subprocesses')
        else:
            config_dic["xDNN_Subprocesses"] = ["Evolving"]
        
        if kwargs.get("Model_Params_dir") is not None:
            model_params_dir = kwargs.get("Model_Params_dir")
            print("Loading model parameters from: ", model_params_dir)
            with open(model_params_dir, "rb") as file:
                Output1 = pickle.load(file)
        else:
            Output1 = xDNN(Input1,Mode1, config_dic=config_dic)
        
        # Output1_dic_path = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/NeuralNetJournal_Dataset_Exp/Prototype-Based_Training/From_xDNN_Offline/Base_Prototypes"
        # with open(Output1_dic_path, "rb") as file:
        #     Output1 = pickle.load(file)

        end = time.time()

        with open(resultsModel_dir + "/Output_xDNN_LearningDic", "wb") as file:
            pickle.dump(Output1, file)

        print ("###################### Model Trained ####################")

        print("Time: ",round(end - start,2), "seconds")
        ###############################################################################

        # Load the files, including features, images and labels for the validation mode

        Input2 = {}
        Input2['xDNNParms'] = Output1['xDNNParms']
        Input2['Images'] = y_test_images 
        Input2['Features'] = X_test
        Input2['Labels'] = y_test_labels
        
        # numProtNonCovid = Output1['xDNNParms']['Parameters'][0]['Support'].shape[0]
        # numProtCovid = Output1['xDNNParms']['Parameters'][1]['Support'].shape[0]
        
        # Get Classes Names
        if config_dic.get("Dataset") is not None:
            if config_dic["Dataset"].get("Classes") is not None:
                classes = config_dic["Dataset"]["Classes"]
            else:
                classes = os.listdir(config_dic["Dataset"]["Directory"] + "/train")
                classes.sort()
        else:
            classes = ["Class_" + str(i) for i in range(Output1['xDNNParms']['CurrentNumberofClass'])]
        
        numClasses = len(classes)
        
        numOfPrototypes = {}
        for i, class_name in enumerate(classes):
            numOfPrototypes[class_name] = Output1['xDNNParms']['Parameters'][i]['Support'].shape[0]
                
        startValidation = time.time()
        Mode2 = 'Validation'
        Output2 = xDNN(Input2,Mode2)
        endValidation = time.time()

        print ("###################### Results ##########################")
        
        # average = 'binary'
        # average = 'micro'
        # average = 'macro'
        if numClasses == 2:
            average = 'binary'
        else:
            average = 'macro'

        # Elapsed Time
        # train_time = round(end - start,2)
        train_time = end - start
        validation_time = round(endValidation - startValidation,2)
        print("Validation Time: ", validation_time, "seconds")
        # accuracy: (tp + tn) / (p + n)
        accuracy = accuracy_score(y_test_labels , Output2['EstLabs'])
        print('Accuracy: %f' % accuracy)
        # precision tp / (tp + fp)
        precision = precision_score(y_test_labels , Output2['EstLabs'], average=average)
        print('Precision: %f' % precision)
        # recall: tp / (tp + fn)
        recall = recall_score(y_test_labels , Output2['EstLabs'],average=average)
        print('Recall: %f' % recall)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(y_test_labels , Output2['EstLabs'], average=average)
        print('F1 score: %f' % f1)
        # kappa
        kappa = cohen_kappa_score(y_test_labels , Output2['EstLabs'])
        print('Cohens kappa: %f' % kappa)
        from sklearn.metrics import roc_auc_score
        if numClasses > 2:
            roc_auc = roc_auc_score(y_test_labels, Output2['Scores'], multi_class='ovr', average="macro")
        else:
            roc_auc = roc_auc_score(y_test_labels, Output2['Scores'][:,1])
        print('ROC AUC: %f' % roc_auc)

        # confusion matrix
        cf_matrix = confusion_matrix(y_test_labels , Output2['EstLabs'])
        print("Confusion Matrix:\n",cf_matrix)
        
        print("Prototypes:\n")
        for class_name in numOfPrototypes.keys():
            print("\t%s: %d" % (class_name, numOfPrototypes[class_name]))
                
        # print("\tNon-Covid: %d" % numProtNonCovid)
        # print("\tCovid: %d" % numProtCovid)

        metricsLabels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Cohens kappa']
        metricsValues = [accuracy, precision, recall, f1, kappa]

        
        displayConfMatrix(cf_matrix, classes)
        displayMetrics(metricsLabels, metricsValues)
        plt.close('all')

        
        # Save Results in a Text File
        f = open(resultsModel_dir + "/Experiment_Results.txt","w+")

        # covid_samples = 1252
        # non_covid_samples = 1229

        f.write("Experiment Parameters:\r\n")
        f.write("\tFeatures Input Layer:\tVGG16 CNN\r\n")
        f.write('\tTrain Size:\t\t\t\t%d Images\r\n' % train_size)
        f.write('\tTest Size:\t\t\t\t%d Images\r\n' % test_size)
        # f.write(f'\tCovid Samples:\t\t\t{covid_samples} Images\r\n')
        # f.write(f'\tNon-Covid Samples:\t\t{non_covid_samples} Images\r\n')
        f.write('\tNum. Classes:\t\t\t\t%d\r\n' % numClasses)
        f.write('\tClasses Names:\t\t\t%s\r\n' % classes)
        f.write('\tIden. Features:\t\t\t%d\r\n' % numFeatures)
        f.write('\tNum. Classes:\t\t\t%d\r\n' % numClasses)
        f.write('\tModel Name:\t\t\t\t%s\r\n' % model_name)

        f.write("\nExperiment Results:\r\n")
        for metric, value in zip(metricsLabels, metricsValues):
            f.write('\t%s:\t\t\t\t%.6f\r\n' % (metric, value))
        f.write('\tTraining Time:\t\t%.2f seconds\r\n' % train_time)

        f.write("\nPrototypes:\r\n")
        for class_name in numOfPrototypes.keys():
            f.write("\t%s: %d\n" % (class_name, numOfPrototypes[class_name]))
            
        f.write("\nSubprocesses: " + str(config_dic["xDNN_Subprocesses"]) + "\r\n")
        
        f.write("\nValidation Time: %f seconds\r\n" % validation_time)
        f.write("\nROC AUC: %f\r\n" % roc_auc)
        # f.write("\tNon-Covid: %d\r\n" % numProtNonCovid)
        # f.write("\tCovid: %d\r\n" % numProtCovid)

        # f.write("\n------------------ Tests Results Explanation ------------------\r\n\n")
        # f.write(protByImageAndClass)
        f.close()
        
        numTrainings += 1
        
        accuracy_accum += accuracy        
        for class_name in numOfPrototypes.keys():
            if class_name in numOfPrototypes_mean.keys():
                numOfPrototypes_mean[class_name] = numOfPrototypes_mean[class_name]*((numTrainings-1)/numTrainings) + numOfPrototypes[class_name]/numTrainings
            else:
                numOfPrototypes_mean[class_name] = numOfPrototypes[class_name]
            
        # Non_Covid_Prototypes_accum += numProtNonCovid
        # Covid_Prototypes_accum += numProtCovid   
        
        best_samples = Output2['BestSamples']
        for class_idx in best_samples.keys():
            sample_name = Input2['Images'][best_samples[class_idx][0]-1]
            distance = best_samples[class_idx][1]
            prototype_name = best_samples[class_idx][2]
            print("Class: %d, Sample: %s, Distance: %f, Prototype: %s" % (class_idx, sample_name, distance, prototype_name))
        
        if single_training:
            break
    
    return {"AvrgAccuracy": accuracy_accum / numTrainings, 
            "Prototypes": numOfPrototypes_mean,
            "Train_Time": train_time,
            "ROC_AUC": roc_auc,
            "Output_Dict": Output1}


# feature_vectors_path = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/COVID-QU-Ex_Dataset/Prototype-Based_Training/PCA/Test_Base_Line_FullDataset/Feature_Vectors_FromFlatten_v3/FeatureVectors.pkl"
# with open(feature_vectors_path, "rb") as file:
#     (X_train, y_train, X_val, y_val) = pickle.load(file)
#     print("(X_train, y_train, X_val, y_val) loaded!!!")

# featureVectors_dir = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/COVID-QU-Ex_Dataset/Prototype-Based_Training/PCA/Test_Base_Line_FullDataset/Feature_Vectors_FromFlatten"

# y_train_file_path = featureVectors_dir + '/data_df_y_train.csv'
# y_test_file_path  = featureVectors_dir + '/data_df_y_val.csv'

# y_train = pd.read_csv(y_train_file_path, delimiter=',',header=None)
# y_test = pd.read_csv(y_test_file_path, delimiter=',',header=None)

# run(models_dir='/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/COVID-QU-Ex_Dataset/Prototype-Based_Training/PCA/Test_Base_Line_FullDataset',
#     single_training=True,
#     model_name="Test",
#     fe_layer = "last_conv",
#     num_PCA_Components = 325, 
#     feature_vectors = (X_train, y_train, X_val, y_val),
#     xDNN_Subprocesses = ["Offline", "Prot_Filtering", "Outliers_Filtering"])


# import json
# config_file_path = "/nfs/home/nvasconcellos.it/softLinkTests/X-MAN/Prototype-Based_Training/src/X_MAN/FineTuning/config/DDSM_Dataset/parameters_Offline_PCA_PyTorch_GPU0_LowEpochs.json"
# config_dic = json.load(open(config_file_path))
# # result = run("/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/Benchmark_Datasets/CBIS_DDSM_Dataset/Prototype-Based_Training/Test_3/xDNN_Offline/", 
# result = run("/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/Benchmark_Datasets/CBIS_DDSM_Dataset/Prototype-Based_Training/Test_1/xDNN_Offline/Optuna/CBIS-DDSM_Dataset_Prot-Based_TrainWithPCA_Torch_730c_CDEAlpha_Study_LowEpochs_FVFromLastConv_xDNNOffline_RecalcPCA_v3/Trial_4/FineTuning_22.weight_To_End_/T1/", 
result = run("/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/Benchmark_Datasets/CBIS_DDSM_Dataset/Prototype-Based_Training/Test_1/xDNN_Offline/Optuna/INbreast_Dataset_Prot-Based_TrainWithPCA_Torch_136c_BatchSizeLearnRateCDEAlpha_Study_LowEpochs_FVFromLastConv_xDNNOffline/Trial_205/FineTuning_20.weight_To_End_/T1/", 
# result = run("/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/Benchmark_Datasets/CBIS_DDSM_Dataset/Prototype-Based_Training/Test_1/xDNN_Offline/Optuna/INbreast_Dataset_Prot-Based_TrainWithPCA_Torch_136c_BatchSizeLearnRateCDEAlpha_Study_LowEpochs_FVFromLastConv_xDNNOffline/Trial_383/FineTuning_6.weight_To_End_/T1/", 
# result = run("/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/Benchmark_Datasets/CBIS_DDSM_Dataset/Prototype-Based_Training/Test_1/xDNN_Offline/Optuna/CBIS-DDSM_Dataset_Prot-Based_TrainWithPCA_Torch_730c_CDEAlpha_Study_LowEpochs_FVFromLastConv_xDNNOffline_RecalcPCA_v3/Trial_4/Base_Line/T1/", 
# # result = run("/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/Benchmark_Datasets/CBIS_DDSM_Dataset/Prototype-Based_Training/Test_1/xDNN_Offline/Optuna/CBIS-DDSM_Dataset_Prot-Based_TrainWithPCA_Torch_615c_CDEAlpha_Study_LowEpochs_FVFromLastConv_xDNNEvolvingRight/Trial_3/FineTuning_25.weight_To_End_/T1/", 
# # result = run("/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/Benchmark_Datasets/CBIS_DDSM_Dataset/Prototype-Based_Training/Test_1/xDNN_Offline/Optuna/CBIS-DDSM_Dataset_Prot-Based_TrainWithPCA_Torch_615c_CDEAlpha_Study_LowEpochs_FVFromLastConv_xDNNEvolvingRight/Trial_3/Base_Line/T1/", 
# result = run("/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/Benchmark_Datasets/CBIS_DDSM_Dataset/Prototype-Based_Training/Test_1/xDNN_Offline/Optuna/INbreast_Dataset_Prot-Based_TrainWithPCA_Torch_136c_BatchSizeLearnRateCDEAlpha_Study_LowEpochs_FVFromLastConv_xDNNOffline/Trial_383/Base_Line/T1/", 
# result = run("/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/Benchmark_Datasets/CBIS_DDSM_Dataset/Prototype-Based_Training/Test_1/xDNN_Offline/Optuna/INbreast_Dataset_Prot-Based_TrainWithPCA_Torch_136c_BatchSizeLearnRateCDEAlpha_Study_LowEpochs_FVFromLastConv_xDNNOffline/Trial_205/Base_Line/T1/", 
# result = run("/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/COVID-QU-Ex_Dataset/Prototype-Based_Training/Optuna/COVID-QU-Ex_Dataset_Prot-Based_TrainWithPCA_Torch_325c_CCEAlpha_Study_GridSearch_LowEpochs_v3/Trial_84/Base_Line/T1/", 
# result = run("/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/COVID-QU-Ex_Dataset/Prototype-Based_Training/Base_Line_From_last_fc/", 
# result = run("/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/COVID-QU-Ex_Dataset/Prototype-Based_Training/Optuna/COVID-QU-Ex_Dataset_Prot-Based_TrainWithPCA_Torch_30c_CDEAlpha_Study_LowEpochs_FVFromLastConv_xDNNEvolving/Trial_16/FineTuning_8.weight_To_End_/T1/", 
# result = run("/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/NeuralNetJournal_Dataset_Exp/Prototype-Based_Training/From_xDNN_Offline/Optuna/NeuralNetJournal_Prot-Based_TrainWithPCA_Torch_199c_CDEAlpha_Study_LowEpochs_New_Split_FVFromLastConv_xDNNOffline_v3/Trial_19/FineTuning_1.weight_To_End_/T1/", 
# result = run("/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/NeuralNetJournal_Dataset_Exp/Prototype-Based_Training/From_xDNN_Offline/Optuna/NeuralNetJournal_Prot-Based_TrainWithPCA_Torch_402c_CDEAlpha_Study_LowEpochs_New_Split_FVFromLastConv_xDNNEvolving/Trial_28/FineTuning_22.weight_To_End_/T1/", 
# result = run("/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/NeuralNetJournal_Dataset_Exp/Prototype-Based_Training/From_xDNN_Offline/Tests_by_Parts/Traditional_FineTuning_NewSplit/Optuna/Study-Extra_NeuralNetJournal_Dataset_Traditional_FineTuning/Trial_Manual_1_FixBatchSizeAndLearningRate/Base_Line/T1/", 
    single_training=True, 
    # model_name=config_dic["xDNN"]["Name"],
    model_name="xDDN_TestSet_AUC_v2",
    fe_layer = "last_conv",
    # fe_layer = "last_fc",
    # xDNN_Subprocesses = config_dic["xDNN"]["xDNN_Subprocesses"])
    # inference_TestSet = True,
    # num_PCA_Components = 730,
    # Use_calculated_V_Matrix = True,
    # Model_Params_dir = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/NeuralNetJournal_Dataset_Exp/Prototype-Based_Training/From_xDNN_Offline/Optuna/NeuralNetJournal_Prot-Based_TrainWithPCA_Torch_402c_CDEAlpha_Study_LowEpochs_New_Split_FVFromLastConv_xDNNEvolving/Trial_28/FineTuning_22.weight_To_End_/T1/xDDN_Evolving/Output_xDNN_LearningDic",
    # xDNN_Subprocesses = ["Evolving"],
    xDNN_Subprocesses = ['Offline', 'Prot_Filtering', 'Outliers_Filtering']
    )

# run(models_dir='/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/NeuralNetJournal_Dataset_Exp/Prototype-Based_Training/From_xDNN_Offline/VGG16_Prot_Training_FromFlatten/', single_training=True)