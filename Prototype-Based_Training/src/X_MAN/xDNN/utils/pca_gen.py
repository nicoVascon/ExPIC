import load_csv
import time

import numpy as np
import pandas as pd

# (X_train, y_train, X_val, y_val) = load_csv.load()

def run(num_PCA_Components, fv):
    (X_train, y_train, X_test, y_test)  = fv

    from sklearn.decomposition import PCA

    start = time.time()

    numComponents = num_PCA_Components
    train_size = X_train.shape[0]
    n_features = X_train.shape[1]
    name_features = ["eixo_"+str(i) for i in range(n_features)]
    # dictionary = dict(zip(name_features, np.hstack((X_train.transpose(), X_test.transpose()))))
    dictionary = dict(zip(name_features, X_train.transpose()))
    pca = PCA(n_components=numComponents, random_state=123)
    # X_train_and_test = pca.fit_transform(pd.DataFrame.from_dict(dictionary))
    # X_train = X_train_and_test[0:train_size]
    # X_test = X_train_and_test[train_size:]
    pca = pca.fit(pd.DataFrame.from_dict(dictionary))
    X_train = pca.transform(pd.DataFrame.from_dict(dictionary))
    dictionary = dict(zip(name_features, X_test.transpose()))
    X_test = pca.transform(pd.DataFrame.from_dict(dictionary))

    print ("###################### PCA ######################")
    print("Data Shape:   ")        
    
    print("X train: ",X_train.shape)
    print("Y train: ",y_train.shape)

    print("X Val: ",X_test.shape)
    print("Y Val: ",y_test.shape)

    end = time.time()

    print("PCA Time: " + str(end - start))

    return (X_train, y_train, X_test, y_test)

import pickle

data_path = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/COVID-QU-Ex_Dataset/Prototype-Based_Training/PCA/Test_Base_Line_FullDataset/Feature_Vectors_FromFlatten_v3/FeatureVectors.pkl"
with open(data_path, "rb") as file:
    (X_train, y_train, X_val, y_val) = pickle.load(file=file)
    print("------------- Input Data Loaded!!!! -------------")

(X_train_pca, y_train_pca, X_val_pca, y_val_pca) = run(478, (X_train, y_train, X_val, y_val))

output_dir = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/NeuralNetJournal_Dataset_Exp/Prototype-Based_Training/PCA/Base_Line/Feature_Vectors_FromFlatten/Pickle/PCA_478comp"

# output_path = output_dir + "/outputs.pkl"

# with open(output_path, "wb") as file:
#     pickle.dump((X_train_pca, X_val_pca), file)

# print("(X_train_pca, X_val_pca) Saved!!!")

output_path = output_dir + "/inputs.pkl"

with open(output_path, "wb") as file:
    pickle.dump((X_train, X_val), file)

print("(X_train, X_val) Saved!!!")