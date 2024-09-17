from multiprocessing.sharedctypes import Value
import torch
import copy
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
#from torchvision.models import VGG16_Weights
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from PIL import Image

import math
import numpy as np
import random

# import matplotlib.pyplot as plt
import json
import datetime
import sys
sys.path.insert(0, '/nfs/home/nvasconcellos.it/softLinkTests/X-MAN/Prototype-Based_Training/src')
import X_MAN
import X_MAN.FineTuning.utils.Generic_Functions as gf
import X_MAN.FineTuning.utils.Loss_Functions as lf

# from X_MAN.Models.VGG16.Model.Converted_VGG16 import Converted_VGG16
from X_MAN.Models.PCA_Emulator.PCA_Emulator import PCA_Emulator

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    # Configure the GPU devices
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        tf.config.experimental.set_virtual_device_configuration(device, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=50)])

import pickle
import shutil
import os

tbf = None # Gloabal Variable to load the TensorBoard_Functions Library

class DatasetSequence(Dataset):
    def __init__(self, x_set, y_set):
        self.x = torch.from_numpy(x_set).to(torch.float32)
        self.y = torch.from_numpy(y_set).to(torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        x = self.x[idx, :]
        y = self.y[idx, :]
        return x, y

def train_model(model, resultsDir, train_dl, val_dl, config_dic):

    checkpoints_dir = resultsDir + '/checkpoints/'
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)   
    
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    
    non_frozen_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(non_frozen_parameters, lr=config_dic["Hyperparameters"]["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer, mode=config_dic["ReduceLROnPlateau"]["mode"], factor=config_dic["ReduceLROnPlateau"]["factor"], patience=config_dic["ReduceLROnPlateau"]["patience"],
                                  threshold=config_dic["ReduceLROnPlateau"]["threshold"], threshold_mode=config_dic["ReduceLROnPlateau"]["threshold_mode"],
                                  cooldown=config_dic["ReduceLROnPlateau"]["cooldown"], min_lr=config_dic["ReduceLROnPlateau"]["min_lr"], eps=config_dic["ReduceLROnPlateau"]["eps"],
                                  verbose=config_dic["ReduceLROnPlateau"]["verbose"])    

    filepath = checkpoints_dir + 'best_model.pt'
    earlyStopping = gf.EarlyStopping(patience=config_dic["EarlyStopping"]["patience"], verbose=config_dic["EarlyStopping"]["verbose"], delta=config_dic["EarlyStopping"]["delta"], path=filepath, trace_func=print)
        
        
    # History
    history = {"train": {"loss": [], "MAE": [], "MAPE": [], "DisAng": [], "Accuracy": []}, 
                "val": {"loss": [], "MAE": [], "MAPE": [], "DisAng": [], "Accuracy": []}}
    num_epochs = config_dic["Hyperparameters"]["epochs"]
    for epoch in range(num_epochs):
        train_loss = 0
        val_loss = 0 
          
        train_MAE = 0
        val_MAE = 0
                
        train_MAPE = 0
        val_MAPE = 0
        
        train_DisAng = 0
        val_DisAng = 0

        # Training loop
        model.train()
        # for i, (x_batch, y_batch) in enumerate(train_dl):
        # for x_batch, y_batch, y_classes in gf.progressbar(train_dl, "Train\tEpoch %d: " % (epoch + 1), "Batches", 40):

        import time
        start = time.time()
        for x_batch, y_batch in gf.progressbar(train_dl, "Train\tEpoch %d: " % (epoch + 1), "Batches", 40):
            
            x_batch = x_batch.to(config_dic["Hardware"]["Device"])
            y_batch = y_batch.to(config_dic["Hardware"]["Device"])
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            # Outputs Normalization
            # outputs = outputs / torch.sqrt(outputs.pow(2).sum(axis=1, keepdim=True))
            
            # Loss Calculation
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item() # loss per batch
            train_loss += batch_loss
            
            # MAE Value Calculation
            batch_MAE = torch.mean(torch.abs(y_batch - outputs))
            train_MAE += batch_MAE
            # MAPE Value Calculation
            # perc_error_vector = torch.abs(y_batch - outputs) / (y_batch + epsilon)
            # batch_MAPE = perc_error_vector.sum() / (perc_error_vector.shape[0] * perc_error_vector.shape[1]) # MAPE per batch
            perc_error_vector = torch.abs(y_batch[y_batch != 0] - outputs[y_batch != 0]) / torch.abs(y_batch[y_batch != 0])
            batch_MAPE = torch.mean(perc_error_vector) # MAPE per batch
            train_MAPE += batch_MAPE
            # Dissimilarity Angle Value Calculation
            norm_y_batch = y_batch / torch.sqrt(y_batch.pow(2).sum(axis=1, keepdim=True)) # Normalize y_batch
            outputs = outputs / torch.sqrt(outputs.pow(2).sum(axis=1, keepdim=True))
            batch_DisAng = torch.mean(torch.acos((norm_y_batch * outputs).sum(axis=1))) * 180 / math.pi
            train_DisAng += batch_DisAng

            a = y_batch.detach().cpu()
            del a
            # criterion.wrong_closest_prototypes.detach().cpu()
            
            import gc
            gc.collect()

            # debug_file_dir = resultsDir + '/debug_memory.txt'                
            # if not os.path.exists(debug_file_dir):
            #     debug_file = open(debug_file_dir, 'w+')
            # else:
            #     debug_file = open(debug_file_dir, 'a+')
            # debug_file.write("\n\n--------- In Train Loop Epoch: %d ---------\n" % (epoch + 1))
            # # import inspect
            # # callers_local_vars = inspect.currentframe().f_back.f_locals.items()
            # for obj in gc.get_objects():
            # # for name, value in callers_local_vars:
            #     try:
            #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            #         # if torch.is_tensor(value) or (hasattr(value, 'data') and torch.is_tensor(value.data)):
            #             debug_file.write("Name: " + str(obj.name) + "\tType: " + str(type(obj)) + "\t" + str(obj.size()) + "\n")
            #             # debug_file.write("\tType: " + str(type(obj)) + "\t" + str(obj.size()) + "\n")
            #     except:
            #         pass
            # debug_file.close()
            
            # print('  batch {} \tLoss: {}\tMAE: {}\n\t\tMAPE: {}\tDis. Angle: {}'.format(i + 1, batch_loss, batch_MAE, batch_MAPE, batch_DisAng))
        end = time.time()
        print("Train Epoch Time: " + str(end-start))
        # Validation loop
        model.eval()
        start = time.time()
        with torch.no_grad():            
            # for i, (x_batch, y_batch) in enumerate(val_dl):
            # for x_batch, y_batch, y_classes in gf.progressbar(val_dl, "Val. \tEpoch %d: " % (epoch + 1), "Samples", 40):
            for x_batch, y_batch in gf.progressbar(val_dl, "Val. \tEpoch %d: " % (epoch + 1), "Batches", 40):
                x_batch = x_batch.to(config_dic["Hardware"]["Device"])
                y_batch = y_batch.to(config_dic["Hardware"]["Device"])
                
                outputs = model(x_batch)
                # Outputs Nomalization
                # outputs = outputs / torch.sqrt(outputs.pow(2).sum(axis=1, keepdim=True))

                # Loss Value Calculation
                loss = criterion(outputs, y_batch)                
                val_loss += loss.item()             
                
                # MAE Value Calculation
                val_MAE += torch.mean(torch.abs(y_batch - outputs))
                # MAPE Value Calculation
                # perc_error_vector = torch.abs(y_batch - outputs) / (y_batch + epsilon)
                # val_MAPE += perc_error_vector.sum() / (perc_error_vector.shape[0] * perc_error_vector.shape[1])
                perc_error_vector = torch.abs(y_batch[y_batch != 0] - outputs[y_batch != 0]) / torch.abs(y_batch[y_batch != 0])
                val_MAPE += torch.mean(perc_error_vector)
                # Dissimilarity Angle Value Calculation
                norm_y_batch = y_batch / torch.sqrt(y_batch.pow(2).sum(axis=1, keepdim=True))
                outputs = outputs / torch.sqrt(outputs.pow(2).sum(axis=1, keepdim=True))
                val_DisAng += torch.mean(torch.acos((norm_y_batch * outputs).sum(axis=1))) * 180 / math.pi

                a = y_batch.detach().cpu()
                del a
                # criterion.wrong_closest_prototypes.detach().cpu()
                import gc
                gc.collect()

                # debug_file_dir = resultsDir + '/debug_memory.txt'                
                # if not os.path.exists(debug_file_dir):
                #     debug_file = open(debug_file_dir, 'w+')
                # else:
                #     debug_file = open(debug_file_dir, 'a+')
                # debug_file.write("\n\n--------- In Validation Loop Epoch: %d ---------\n" % (epoch + 1))
                # # import inspect
                # # callers_local_vars = inspect.currentframe().f_back.f_locals.items()
                # for obj in gc.get_objects():
                # # for name, value in callers_local_vars:
                #     try:
                #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                #         # if torch.is_tensor(value) or (hasattr(value, 'data') and torch.is_tensor(value.data)):
                #             # debug_file.write("Name: " + str(name) + "\tType: " + str(type(value)) + "\t" + str(value.size()) + "\n")
                #             debug_file.write("\tType: " + str(type(obj)) + "\t" + str(obj.size()) + "\n")
                #     except:
                #         pass
                # debug_file.close()

        end = time.time()
        print("Val Epoch Time: " + str(end-start))

        train_loss /= len(train_dl)
        val_loss /= len(val_dl)
        scheduler.step(val_loss)
        
        train_MAE /= len(train_dl)
        val_MAE /= len(val_dl)
        
        train_MAPE /= len(train_dl)
        val_MAPE /= len(val_dl)
        
        train_DisAng /= len(train_dl)
        val_DisAng /= len(val_dl)
        
        # Save Train and Validation Metrics' History
        train_MAE = train_MAE.cpu().detach().numpy()
        train_MAPE = train_MAPE.cpu().detach().numpy()
        train_DisAng = train_DisAng.cpu().detach().numpy()
        
        val_MAE = val_MAE.cpu().detach().numpy()
        val_MAPE = val_MAPE.cpu().detach().numpy()
        val_DisAng = val_DisAng.cpu().detach().numpy()
        
        history["train"]["loss"].append(train_loss)        
        history["train"]["MAE"].append(train_MAE)        
        history["train"]["MAPE"].append(train_MAPE)        
        history["train"]["DisAng"].append(train_DisAng)
        
        history["val"]["loss"].append(val_loss)        
        history["val"]["MAE"].append(val_MAE)        
        history["val"]["MAPE"].append(val_MAPE)        
        history["val"]["DisAng"].append(val_DisAng)      
        
        # Save Train and Validation Metrics
        if config_dic["General"]["TensorBoard"]:
            train_dic = {"loss": train_loss, "MAE": train_MAE, "MAPE": train_MAPE, "DisAng": train_DisAng}
            val_dic = {"loss": val_loss, "MAE": val_MAE, "MAPE": val_MAPE, "DisAng": val_DisAng}

            tbf.write_scalars(config_dic["TensorBoard"]["train_summary_writer"], train_dic, epoch)            
            tbf.write_scalars(config_dic["TensorBoard"]["val_summary_writer"], val_dic, epoch)
            if config_dic["xDNN"]["Prot_Recalculation"]:
                val_dic = {"loss": val_loss, "MAE": val_MAE, "MAPE": val_MAPE, "DisAng": val_DisAng}
                tbf.write_scalars(config_dic["TensorBoard"]["val_summary_writer"], val_dic, epoch)
            
        # Report Validation Loss if Trial is in kwargs
        if config_dic["General"].get("Trial") is not None:
            config_dic["General"]["Trial"].report(val_loss, epoch)
                
        print(f"Epoch {epoch+1}/{num_epochs}, \tTrain Loss: {train_loss:e},\t\tVal Loss: {val_loss:e}\n\t\tTrain MAE: {train_MAE:e},\t\tVal MAE: {val_MAE:e}")
        print(f"\t\tTrain MAPE: {train_MAPE},\tVal MAPE: {val_MAPE}\n\t\tTrain DisAng: {train_DisAng},\tVal DisAng: {val_DisAng}")

        # print("first layer:")
        # print(list(model.parameters())[1])
        
        # print("last layer:")
        # print(list(model.parameters())[-1])
        
        # Earliy Stopping and Saving best model based on validation loss
        earlyStopping(val_loss, model)
        
        if earlyStopping.early_stop:
            print("\nEarly stopping!!!")
            break
            
    with open(resultsDir + '/history.pkl', 'wb') as f:
        pickle.dump(history, f)

    return history, earlyStopping.val_loss_min

config_dir = X_MAN.FineTuning.__path__[0] + "/config/"


config_file_path_default = config_dir + "COVID-QU-Ex_Dataset/parameters_test_2.json"

config_dic_default = json.load(open(config_file_path_default))

def run(config_dic : dict = config_dic_default, **kwargs):
    
    if kwargs.get("batch_size") is not None:
        config_dic["Hyperparameters"]["batch_size"] = kwargs.get("batch_size")
    
    if kwargs.get("lr") is not None:
        config_dic["Hyperparameters"]["learning_rate"] = kwargs.get("lr")   
        
    if kwargs.get("layer_from_FT") is not None:
        config_dic["Fine_Tuning"]["UnFreezing_Threshold_Layer"] = kwargs.get("layer_from_FT")
        
    if kwargs.get("weights") is not None:
        config_dic["Fine_Tuning"]["weights"] = kwargs.get("weights")
    else:
        config_dic["Fine_Tuning"]["weights"] = "imagenet"

    # Directories Definition
    base_algoritm_dir = config_dic["Directories"]["base_algoritm_dir"]
    if kwargs.get("exp_name") is not None:
        exp_name = kwargs.get("exp_name")
    else:
        exp_name = "VGG16_Prot_Training_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # exp_name = "VGG16_Prot_Training_" + "Testv2_2"
    resultsDir = base_algoritm_dir + exp_name
    config_dic["General"]["Experiment_Name"] = exp_name

    # dataset_dir = config_dic["Dataset"]["Directory"]
    # inputsDic_dir = dataset_dir + "/inputsDic"
    # labelsDic_dir = dataset_dir + "/labelsDic"
    
    # Save Configuration
    if not os.path.exists(resultsDir):
        os.makedirs(resultsDir)
    # shutil.copy(config_file_path, resultsDir + "/config.json")
    config_dic_json = json.dumps(config_dic, indent=4)
    with open(resultsDir + "/config.json", "w") as f:
        f.write(config_dic_json)

    # Hardware Definition
    device_indx = config_dic["Hardware"]["GPU"]
    avail_devices_count = torch.cuda.device_count()
    actual_device_indx = device_indx if device_indx < avail_devices_count else avail_devices_count - 1
    torch_device = "cuda:" + str(actual_device_indx)
    
    # Optuna Configuration
    if kwargs.get("trial") is not None:
        config_dic["General"]["Trial"] = kwargs.get("trial")

    # # Dataset Loading
    # with open(inputsDic_dir, "rb") as file:
    #     inputsDic = pickle.load(file)
        
    # with open(labelsDic_dir, "rb") as file:
    #     labelsDic = pickle.load(file)

    if config_dic["General"]["TensorBoard"]:
        import tensorflow as tf
        global tbf
        import X_MAN.FineTuning.utils.TensorBoard_Functions as tbf
        # TensorBoard: Set up summary writers
        tensorflowBoardDir = base_algoritm_dir + "/Tensorboard" + "/" + exp_name
        if not os.path.exists(tensorflowBoardDir):
            os.makedirs(tensorflowBoardDir)
        
        train_log_dir = tensorflowBoardDir + '/train_'
        val_log_dir = tensorflowBoardDir + '/val_'
        summary_log_dir = tensorflowBoardDir + '/markdown/summary'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)
        config_dic["TensorBoard"] = {"train_summary_writer": train_summary_writer, "val_summary_writer": val_summary_writer}
        if config_dic["xDNN"]["Prot_Recalculation"]:
            xDNN_log_dir = tensorflowBoardDir + '/xDNN_'
            xDNN_summary_writer = tf.summary.create_file_writer(xDNN_log_dir)
            config_dic["TensorBoard"]["xDNN_summary_writer"] = xDNN_summary_writer        
        tbf.write_Experiment_Setup(summary_log_dir, config_dic)


    # Reproducibility (Deterministic) Setup
    def seed_worker(worker_id):
        random.seed(config_dic["Dataset"]["Seed"])
        torch.manual_seed(config_dic["Dataset"]["Seed"])
        torch.cuda.manual_seed(config_dic["Dataset"]["Seed"])
        np.random.seed(config_dic["Dataset"]["Seed"])
    
    g = torch.Generator()
    g.manual_seed(config_dic["Dataset"]["Seed"])
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(True)
    random.seed(config_dic["Dataset"]["Seed"])
    torch.manual_seed(config_dic["Dataset"]["Seed"])
    torch.cuda.manual_seed(config_dic["Dataset"]["Seed"])
    np.random.seed(config_dic["Dataset"]["Seed"])

    # train_ds = DatasetSequence(inputsDic["train"], labelsDic["train"])
    # val_ds = DatasetSequence(inputsDic["val"], labelsDic["val"])
    
    # config_dic["Dataset"]["Classes"] = os.listdir(config_dic["Dataset"]["Directory"] + "/train")
    # config_dic["Dataset"]["Classes"].sort()

    # inputs_path = config_dic["Dataset"]["Directory"] + "/inputs.pkl"
    inputs_path = config_dic["Dataset"]["Directory"]["inputs"]
    with open(inputs_path, "rb") as file:
        X_train, X_test = pickle.load(file=file)

    # outputs_path = config_dic["Dataset"]["Directory"] + "/outputs.pkl"
    outputs_path = config_dic["Dataset"]["Directory"]["outputs"]
    with open(outputs_path, "rb") as file:
        Y_train, Y_test = pickle.load(file=file)

    print ("###################### PCA ######################")
    print("Data Shape:   ")        

    print("X train: ",X_train.shape)
    print("Y train: ",Y_train.shape)

    print("X test: ",X_test.shape)
    print("Y test: ",Y_test.shape)
    
    train_ds = DatasetSequence(X_train, Y_train)
    val_ds = DatasetSequence(X_test, Y_test)
    
    train_dl = DataLoader(train_ds, batch_size=config_dic["Hyperparameters"]["batch_size"], shuffle=config_dic["Dataset"]["Shuffle"], 
                        num_workers=4,    
                        worker_init_fn=seed_worker,
                        generator=g)
    val_dl = DataLoader(val_ds, batch_size=config_dic["Hyperparameters"]["batch_size"], 
                        num_workers=4,    
                        worker_init_fn=seed_worker,
                        generator=g)

    # Model Initialization
    # print('config_dic["Fine_Tuning"]["weights"] = ' + config_dic["Fine_Tuning"]["weights"])
    # if config_dic["Fine_Tuning"]["weights"] == "imagenet":
    #     pcaEmulator_model = Converted_VGG16(fe_layer = config_dic["xDNN"]["Feature_Extraction_Layer"])
    #     # pcaEmulator_model = Converted_VGG16(fe_layer = config_dic["xDNN"]["Feature_Extraction_Layer"], numClasses = 3)
    # else:
    #     pcaEmulator_model = Converted_VGG16(config_dic["Fine_Tuning"]["weights"], 
    #                                   fe_layer = config_dic["xDNN"]["Feature_Extraction_Layer"])
       
    pcaEmulator_model = PCA_Emulator(origNumComp = config_dic["PCA"]["Original_Num_Comp"], newNumComp = config_dic["PCA"]["New_Num_Comp"])
       
    
    # pcaEmulator_model.eval()
    device = torch.device(torch_device if torch.cuda.is_available() else "cpu")
    pcaEmulator_model = pcaEmulator_model.to(device)
    print(pcaEmulator_model)
    
    # Fine Tuning Setup
    if config_dic["Fine_Tuning"]["Fine_Tuning"]:
        gf.freeze_layers(pcaEmulator_model, config_dic["Fine_Tuning"]["UnFreezing_Threshold_Layer"])
    
    # Parallel Computing
    if config_dic["General"]["Parallel_Computing"]:
        pcaEmulator_model = gf.make_data_parallel(pcaEmulator_model,[1])
    
    device=next(pcaEmulator_model.parameters()).device
    config_dic["Hardware"]["Device"] = device
    
    # Train the Model
    if config_dic["General"]["Train"]:
        history, min_val_loss = train_model(pcaEmulator_model, resultsDir, train_dl, val_dl, config_dic)
    else:
        # Load History
        with open(resultsDir + '/history.pkl', 'rb') as f:
            history = pickle.load(f)
            min_val_loss = min(history["val"]["loss"])


    if config_dic["General"]["Plot Results"]:
        # Figures Directory
        figuresDir = resultsDir + "/" + "figures"

        # Plot the training and validation loss
        gf.plot_metrics(history, "loss", figuresDir)
        # Plot the training and validation MAE
        gf.plot_metrics(history, "MAE", figuresDir)
        # Plot the training and validation MAPE
        gf.plot_metrics(history, "MAPE", figuresDir)
        # Plot the training and validation DisAng
        gf.plot_metrics(history, "DisAng", figuresDir)
    
    return min_val_loss, config_dic

# config_file_path = config_dir + "COVID-QU-Ex_Dataset/parameters_Offline_GPU1.json"
config_file_path = "/nfs/home/nvasconcellos.it/softLinkTests/X-MAN/Prototype-Based_Training/src/X_MAN/Models/PCA_Emulator/parameters_PCA_Emul_GPU0.json"

config_dic = json.load(open(config_file_path))
# config_dic["xDNN"]["Base_Prototypes_path"] = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/COVID-QU-Ex_Dataset/Prototype-Based_Training/Optuna/COVID-QU-Ex_Dataset_Prot-Based_Train_Test_v2/Trial_2/Base_Line/T1/xDDN_Offline_OutliersFilt/Output_xDNN_LearningDic"
# run(config_dic)

val_loss, config_dic = run(config_dic=config_dic, 
                                batch_size=280, lr=10000e-6,
                                # layer_from_FT = "29.weight",
                                # weights="imagenet",
                                exp_name= "PCA_Emulator_478Comp_withoutBias_v4")