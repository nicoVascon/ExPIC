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

import json
import datetime
import sys
sys.path.insert(0, '/nfs/home/nvasconcellos.it/softLinkTests/X-MAN/Prototype-Based_Training/src')
import X_MAN
import X_MAN.FineTuning.utils.Generic_Functions as gf
import X_MAN.FineTuning.utils.Loss_Functions as lf

from X_MAN.Models.VGG16.Model.Converted_VGG16 import Converted_VGG16

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
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
    # def __init__(self, x_set, y_set):
    #     self.x = x_set
    #     self.y = y_set
    #     # self.transform = transforms.Compose([
    #     #     transforms.Resize(256),
    #     #     transforms.CenterCrop(224),# VGG-16 Takes 224x224 images as input, so we resize all of them
    #     #     transforms.ToTensor()
    #     #     # ,transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    #     #     ]
    #     # )
    
    def __init__(self, root_dir):
        self.classes = os.listdir(root_dir)
        self.classes.sort()
        self.classes_as_indx = {self.classes[i]: i for i in range(len(self.classes))}
        
        self.x = []
        self.y = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.x.append(img_path)
                self.y.append(self.classes_as_indx[class_name])
                
        random.shuffle(self.x, lambda: 0.5)
        random.shuffle(self.y, lambda: 0.5)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):        
        
        img_path = self.x[idx]
        # img_raw = Image.open(img_path)
        # img = Image.new("RGB", img_raw.size)
        # img.paste(img_raw)
        # img = self.transform(img)

        # # x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
        # x = Variable(img.float(), requires_grad=False)
        
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        # x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        x = torch.from_numpy(x.copy())
        
        # y = torch.Tensor(self.y[idx]).view(-1)
        
        # x = torch.rand(1, 224, 224, 3)
        return x, self.y[idx]

def train_model(model, resultsDir, train_dl, val_dl, config_dic):

    checkpoints_dir = resultsDir + '/checkpoints/'
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    with open(config_dic["xDNN"]["Base_Prototypes_path"], "rb") as file:
        prototypesTrainDic = pickle.load(file)

    prototypes_dict = {}
    # prototypes_dict[0] = prototypesTrainDic['xDNNParms']['Parameters'][0]["Centre"]
    # prototypes_dict[1] = prototypesTrainDic['xDNNParms']['Parameters'][1]["Centre"]
    for class_indx in prototypesTrainDic['xDNNParms']['Parameters'].keys():
        prototypes_dict[class_indx] = prototypesTrainDic['xDNNParms']['Parameters'][class_indx]["Centre"]
    
    # Load PCA V Matrix 
    if config_dic.get("PCA") is not None and config_dic["PCA"].get("V_Matrix") is not None:
        V_matrix = torch.load(config_dic["PCA"]["V_Matrix"]).to(torch.float32)
        V_matrix = V_matrix.to(config_dic["Hardware"]["Device"])
    else: 
        V_matrix = None  
    
    # criterion = nn.CrossEntropyLoss()
    if config_dic["Loss"]["Function"] == "MSE":
        criterion = nn.MSELoss()
    else:
        criterion = lf.CCE(clusters=prototypes_dict, alpha=config_dic["Loss"]["alpha"], 
                        epsilon=config_dic["Loss"]["epsilon"], device=config_dic["Hardware"]["Device"])
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

        train_acc = torch.Tensor([0]).to(config_dic["Hardware"]["Device"])
        val_acc = torch.Tensor([0]).to(config_dic["Hardware"]["Device"])
        
        gradients_dict = {name + "_GradientNorm":0.0 for name, param in model.named_parameters() if param.requires_grad}
        
        train_outputs = None
        val_outputs = None
        train_classes = None
        val_classes = None
        num_outputs_train = 0
        num_outputs_val = 0

        # Training loop
        model.train()
        # for i, (x_batch, y_batch) in enumerate(train_dl):
        # for x_batch, y_batch, y_classes in gf.progressbar(train_dl, "Train\tEpoch %d: " % (epoch + 1), "Batches", 40):

        import time
        start = time.time()
        for x_batch, y_classes in gf.progressbar(train_dl, "Train\tEpoch %d: " % (epoch + 1), "Batches", 40):
            
            x_batch = x_batch.to(config_dic["Hardware"]["Device"])
            # y_batch = y_batch.to(config_dic["Hardware"]["Device"])
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            # PCA Reduction
            if V_matrix is not None:
                outputs = torch.matmul(outputs, V_matrix)
            # Outputs Normalization
            # outputs = outputs / torch.sqrt(outputs.pow(2).sum(axis=1, keepdim=True))            
            
            # Loss Calculation
            # loss = criterion(outputs, y_batch)
            if config_dic["Loss"]["Function"] == "MSE":
                y_batch = torch.zeros(outputs.shape[0], outputs.shape[1]).to(config_dic["Hardware"]["Device"])
                # print("y_classes: " + str(y_classes))
                y_batch[range(0, outputs.shape[0]), y_classes] = 1
                loss = criterion(outputs, y_batch)
                y_pred = torch.argmax(outputs, dim=1)
                acc = torch.sum(y_pred == y_classes.to(config_dic["Hardware"]["Device"]))/outputs.shape[0]
                train_acc += acc
            else:
                loss = criterion(outputs, y_classes)
                # Actual y_batch getting
                y_batch = criterion.actual_closest_prototypes


            loss.backward()
            
            # Gradient Norm Clipping
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
            
            # min_grad_value = 1e-5
            # for index, param in enumerate(model.parameters()):
            #     if torch.abs(param.grad) < min_grad_value:
            #         param.grad = torch.tensor(torch.sign(param.grad)*min_grad_value)
            
            optimizer.step()
            
            batch_loss = loss.item() # loss per batch
            if torch.isnan(loss):
                import optuna
                raise optuna.TrialPruned()           
            
            train_loss += batch_loss
            
            # MAE Value Calculation
            batch_MAE = torch.mean(torch.abs(y_batch - outputs))
            train_MAE += batch_MAE
            # MAPE Value Calculation
            # perc_error_vector = torch.abs(y_batch - outputs) / (y_batch + epsilon)
            # batch_MAPE = perc_error_vector.sum() / (perc_error_vector.shape[0] * perc_error_vector.shape[1]) # MAPE per batch
            perc_error_vector = torch.abs(y_batch[y_batch != 0] - outputs[y_batch != 0]) / y_batch[y_batch != 0]
            batch_MAPE = torch.mean(perc_error_vector) # MAPE per batch
            train_MAPE += batch_MAPE
            # Dissimilarity Angle Value Calculation
            norm_y_batch = y_batch / torch.sqrt(y_batch.pow(2).sum(axis=1, keepdim=True)) # Normalize y_batch
            batch_DisAng = torch.mean(torch.acos((norm_y_batch * outputs).sum(axis=1))) * 180 / math.pi
            train_DisAng += batch_DisAng

            a = y_batch.detach().cpu()
            # b = criterion.actual_closest_prototypes.detach().cpu()
            del a
            # del b
            # criterion.wrong_closest_prototypes.detach().cpu()
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    gradients_dict[name + "_GradientNorm"] += torch.norm(param.grad.detach(), p = 2) # Norm Type p=2
                        
            import gc
            gc.collect()
            
            if train_outputs is None:
                train_outputs = torch.zeros(config_dic["Dataset"]["Train_Size"], outputs.shape[1], dtype=outputs.dtype).to(config_dic["Hardware"]["Device"])
                val_outputs = torch.zeros(config_dic["Dataset"]["Val_Size"], outputs.shape[1], dtype=outputs.dtype).to(config_dic["Hardware"]["Device"])
                train_classes = torch.zeros(config_dic["Dataset"]["Train_Size"], dtype=y_classes.dtype).to(config_dic["Hardware"]["Device"])
                val_classes = torch.zeros(config_dic["Dataset"]["Val_Size"], dtype=y_classes.dtype).to(config_dic["Hardware"]["Device"])
                
                print("train_outputs shape: " + str(train_outputs.shape))
                print("val_outputs shape: " + str(val_outputs.shape))
                print("train_classes shape: " + str(train_classes.shape))
                print("val_classes shape: " + str(val_classes.shape))
                
            train_outputs[range(num_outputs_train, num_outputs_train + outputs.shape[0]), :] = outputs
            train_classes[range(num_outputs_train, num_outputs_train + outputs.shape[0])] = y_classes.to(config_dic["Hardware"]["Device"])
            num_outputs_train += outputs.shape[0]
            
            
            # print('  batch {} \tLoss: {}\tMAE: {}\n\t\tMAPE: {}\tDis. Angle: {}'.format(i + 1, batch_loss, batch_MAE, batch_MAPE, batch_DisAng))
        end = time.time()
        print("Train Epoch Time: " + str(end-start))              
        
        # Validation loop
        model.eval()
        start = time.time()
        with torch.no_grad():            
            # for i, (x_batch, y_batch) in enumerate(val_dl):
            # for x_batch, y_batch, y_classes in gf.progressbar(val_dl, "Val. \tEpoch %d: " % (epoch + 1), "Samples", 40):
            for x_batch, y_classes in gf.progressbar(val_dl, "Val. \tEpoch %d: " % (epoch + 1), "Batches", 40):
                x_batch = x_batch.to(config_dic["Hardware"]["Device"])
                # y_batch = y_batch.to(config_dic["Hardware"]["Device"])
                
                outputs = model(x_batch)
                # PCA Reduction
                if V_matrix is not None:
                    outputs = torch.matmul(outputs, V_matrix)
                
                val_outputs[range(num_outputs_val, num_outputs_val + outputs.shape[0]), :] = outputs
                val_classes[range(num_outputs_val, num_outputs_val + outputs.shape[0])] = y_classes.to(config_dic["Hardware"]["Device"])
                num_outputs_val += outputs.shape[0]
                
                # Outputs Nomalization
                outputs = outputs / torch.sqrt(outputs.pow(2).sum(axis=1, keepdim=True))
                # Loss Value Calculation
                # loss = criterion(outputs, y_batch)
                if config_dic["Loss"]["Function"] == "MSE":
                    y_batch = torch.zeros(outputs.shape[0], outputs.shape[1]).to(config_dic["Hardware"]["Device"])
                    # print("y_classes: " + str(y_classes))
                    y_batch[range(0, outputs.shape[0]), y_classes] = 1
                    loss = criterion(outputs, y_batch)
                    y_pred = torch.argmax(outputs, dim=1)
                    acc = torch.sum(y_pred == y_classes.to(config_dic["Hardware"]["Device"]))/outputs.shape[0]
                    val_acc += acc
                else:
                    loss = criterion(outputs, y_classes)
                    # Actual y_batch getting
                    y_batch = criterion.actual_closest_prototypes
                val_loss += loss.item()             
                
                # MAE Value Calculation
                val_MAE += torch.mean(torch.abs(y_batch - outputs))
                # MAPE Value Calculation
                # perc_error_vector = torch.abs(y_batch - outputs) / (y_batch + epsilon)
                # val_MAPE += perc_error_vector.sum() / (perc_error_vector.shape[0] * perc_error_vector.shape[1])
                perc_error_vector = torch.abs(y_batch[y_batch != 0] - outputs[y_batch != 0]) / y_batch[y_batch != 0]
                val_MAPE += torch.mean(perc_error_vector)
                # Dissimilarity Angle Value Calculation
                norm_y_batch = y_batch / torch.sqrt(y_batch.pow(2).sum(axis=1, keepdim=True))
                val_DisAng += torch.mean(torch.acos((norm_y_batch * outputs).sum(axis=1))) * 180 / math.pi

                a = y_batch.detach().cpu()
                # b = criterion.actual_closest_prototypes.detach().cpu()
                del a
                # del b
                # criterion.wrong_closest_prototypes.detach().cpu()
                import gc
                gc.collect()                

        end = time.time()
        print("Val Epoch Time: " + str(end-start))
        
        print("Outputs Shape: " + str(outputs.shape))
        with open(resultsDir + '/FV_FTLayer_%s_Epoch_%d.pkl' % (config_dic["Fine_Tuning"]["UnFreezing_Threshold_Layer"], epoch + 1), 'wb') as f:
            pickle.dump((train_outputs, train_classes), f)
        with open(resultsDir + '/Prototypes_FTLayer_%s.pkl' % (config_dic["Fine_Tuning"]["UnFreezing_Threshold_Layer"],), 'wb') as f:
            pickle.dump(prototypes_dict, f)
        with open(resultsDir + '/FV_Val_FTLayer_%s_Epoch_%d.pkl' % (config_dic["Fine_Tuning"]["UnFreezing_Threshold_Layer"], epoch + 1), 'wb') as f:
            pickle.dump((val_outputs, val_classes), f)

        train_loss /= len(train_dl)
        val_loss /= len(val_dl)
        scheduler.step(val_loss)
        
        train_MAE /= len(train_dl)
        val_MAE /= len(val_dl)
        
        train_MAPE /= len(train_dl)
        val_MAPE /= len(val_dl)
        
        train_DisAng /= len(train_dl)
        val_DisAng /= len(val_dl)  

        train_acc /= len(train_dl)
        val_acc /= len(val_dl)  
        
        for name in gradients_dict.keys():
            gradients_dict[name] = (gradients_dict[name] / len(train_dl)).cpu().detach().numpy()
        
        # Save Train and Validation Metrics' History
        train_MAE = train_MAE.cpu().detach().numpy()
        train_MAPE = train_MAPE.cpu().detach().numpy()
        train_DisAng = train_DisAng.cpu().detach().numpy()
        train_acc = train_acc.cpu().detach().numpy()
        
        val_MAE = val_MAE.cpu().detach().numpy()
        val_MAPE = val_MAPE.cpu().detach().numpy()
        val_DisAng = val_DisAng.cpu().detach().numpy()
        val_acc = val_acc.cpu().detach().numpy()
        
        history["train"]["loss"].append(train_loss)        
        history["train"]["MAE"].append(train_MAE)        
        history["train"]["MAPE"].append(train_MAPE)        
        history["train"]["DisAng"].append(train_DisAng)
        history["train"]["Accuracy"].append(train_acc)
        
        history["val"]["loss"].append(val_loss)        
        history["val"]["MAE"].append(val_MAE)        
        history["val"]["MAPE"].append(val_MAPE)        
        history["val"]["DisAng"].append(val_DisAng)     
        history["val"]["Accuracy"].append(val_acc)     
        
        # Save Train and Validation Metrics
        if config_dic["General"]["TensorBoard"]:
            train_dic = {"loss": train_loss, "MAE": train_MAE, "MAPE": train_MAPE, "DisAng": train_DisAng}
            val_dic = {"loss": val_loss, "MAE": val_MAE, "MAPE": val_MAPE, "DisAng": val_DisAng}
            if config_dic["Loss"]["Function"] == "MSE":
                train_dic["Accuracy"] = train_acc.item()
                val_dic["Accuracy"] = val_acc.item()
            tbf.write_scalars(config_dic["TensorBoard"]["train_summary_writer"], train_dic, epoch)            
            tbf.write_scalars(config_dic["TensorBoard"]["val_summary_writer"], val_dic, epoch)
            if config_dic["xDNN"]["Prot_Recalculation"]:
                val_dic = {"loss": val_loss, "MAE": val_MAE, "MAPE": val_MAPE, "DisAng": val_DisAng}
                tbf.write_scalars(config_dic["TensorBoard"]["val_summary_writer"], val_dic, epoch)
                
            tbf.write_scalars(config_dic["TensorBoard"]["Gradients_summary_writer"], gradients_dict, epoch)
            
        # Report Validation Loss if Trial is in kwargs
        if config_dic["General"].get("Trial") is not None:
            config_dic["General"]["Trial"].report(val_loss, epoch)

        print("Current Lr: " + str(optimizer.param_groups[0]["lr"]))
                
        print(f"Epoch {epoch+1}/{num_epochs}, \tTrain Loss: {train_loss:e},\t\tVal Loss: {val_loss:e}\n\t\tTrain MAE: {train_MAE:e},\t\tVal MAE: {val_MAE:e}")
        print(f"\t\tTrain MAPE: {train_MAPE},\tVal MAPE: {val_MAPE}\n\t\tTrain DisAng: {train_DisAng},\tVal DisAng: {val_DisAng}")
        print(f"\t\tTrain Acc: {train_acc},\tVal Acc: {val_acc}")

        # print("first layer:")
        # print(list(model.parameters())[1])
        
        # print("last layer:")
        # print(list(model.parameters())[-1])
        
        # Earliy Stopping and Saving best model based on validation loss
        earlyStopping(val_loss, model)
        
        if earlyStopping.early_stop:
            print("\nEarly stopping!!!")
            break
        
        # if config_dic["xDNN"]["Prot_Recalculation"]:
        #     import Feature_Extraction_VGG16_PyTorch as featureExtraction
        #     import xDNN_run_vOptuna as xDNN_run
            
        #     featureExtraction.Extract_Features(resultsDir + "/temp_xDNN", 
        #                           fe_layer = config_dic["xDNN"]["Feature_Extraction_Layer"],
        #                           model = model)
        #     result = xDNN_run.run(resultsDir + "/temp_xDNN" + '/', 
        #                                       single_training=True,
        #                                       model_name=config_dic["xDNN"]["Name"],
        #                                       fe_layer = config_dic["xDNN"]["Feature_Extraction_Layer"],
        #                                       xDNN_Subprocesses = config_dic["xDNN"]["xDNN_Subprocesses"])

        #     prototypesTrainDic = result["Output_Dict"]
            
        #     prototypes_dict[0] = prototypesTrainDic['xDNNParms']['Parameters'][0]["Centre"]
        #     prototypes_dict[1] = prototypesTrainDic['xDNNParms']['Parameters'][1]["Centre"]
            
        #     criterion.setClusters(prototypes_dict)  
            
        #     if config_dic["General"]["TensorBoard"]:
        #         xDNN_accuracy = result["AvrgAccuracy"]
        #         xDNN_Non_Covid_Prototypes = result["Prototypes"]["Non-Covid"]
        #         xDNN_Covid_Prototypes = result["Prototypes"]["Covid"]
        #         xDNN_dic = {"xDNN_Accuracy_PerEpoch": xDNN_accuracy, "Non_Covid_Prototypes_PerEpoch": xDNN_Non_Covid_Prototypes, "Covid_Prototypes_PerEpoch": xDNN_Covid_Prototypes}
        #         tbf.write_scalars(config_dic["TensorBoard"]["xDNN_summary_writer"], xDNN_dic, epoch)
            
    with open(resultsDir + '/history.pkl', 'wb') as f:
        pickle.dump(history, f)

    return history, earlyStopping.val_loss_min

config_dir = X_MAN.FineTuning.__path__[0] + "/config/"

def run(config_dic : dict, **kwargs):
    
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
        
    # if kwargs.get("Prot_Recalculation") is not None:
    #     config_dic["xDNN"]["Prot_Recalculation"] = kwargs.get("Prot_Recalculation")
    # else:
    #     config_dic["xDNN"]["Prot_Recalculation"] = False

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
        
        gradients_log_dir = tensorflowBoardDir + '/gradients'
        gradients_summary_writer = tf.summary.create_file_writer(gradients_log_dir)
        config_dic["TensorBoard"]["Gradients_summary_writer"] = gradients_summary_writer


    # Reproducibility (Deterministic) Setup
    def seed_worker(worker_id):
        random.seed(config_dic["Dataset"]["Seed"])
        torch.manual_seed(config_dic["Dataset"]["Seed"])
        torch.cuda.manual_seed(config_dic["Dataset"]["Seed"])
        np.random.seed(config_dic["Dataset"]["Seed"])
    
    g = torch.Generator()
    g.manual_seed(config_dic["Dataset"]["Seed"])
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(True)
    random.seed(config_dic["Dataset"]["Seed"])
    torch.manual_seed(config_dic["Dataset"]["Seed"])
    torch.cuda.manual_seed(config_dic["Dataset"]["Seed"])
    np.random.seed(config_dic["Dataset"]["Seed"])

    # train_ds = DatasetSequence(inputsDic["train"], labelsDic["train"])
    # val_ds = DatasetSequence(inputsDic["val"], labelsDic["val"])
    
    config_dic["Dataset"]["Classes"] = os.listdir(config_dic["Dataset"]["Directory"] + "/train")
    config_dic["Dataset"]["Classes"].sort()
    
    train_ds = DatasetSequence(config_dic["Dataset"]["Directory"] + "/train")
    val_ds = DatasetSequence(config_dic["Dataset"]["Directory"] + "/val")
    
    train_dl = DataLoader(train_ds, batch_size=config_dic["Hyperparameters"]["batch_size"], shuffle=config_dic["Dataset"]["Shuffle"], 
                        num_workers=4,    
                        worker_init_fn=seed_worker,
                        generator=g)
    val_dl = DataLoader(val_ds, batch_size=config_dic["Hyperparameters"]["batch_size"], 
                        num_workers=4,    
                        worker_init_fn=seed_worker,
                        generator=g)
    
    config_dic["Dataset"]["Train_Size"] = len(train_ds)
    config_dic["Dataset"]["Val_Size"] = len(val_ds)

    # Model Initialization
    print('config_dic["Fine_Tuning"]["weights"] = ' + config_dic["Fine_Tuning"]["weights"])
    if config_dic["Fine_Tuning"]["weights"] == "imagenet":
        config_dic["Fine_Tuning"]["weights"] = None
        
    VGG16_model = Converted_VGG16(config_dic["Fine_Tuning"]["weights"],
                                    fe_layer = config_dic["xDNN"]["Feature_Extraction_Layer"],
                                    pca_emulator=config_dic["PCA"]["PCA_Emulator"],
                                    pca_emul_weights_path=config_dic["PCA"]["Weights"],
                                    pca_origNumComp=config_dic["PCA"]["Original_Num_Comp"],
                                    pca_newNumComp=config_dic["PCA"]["New_Num_Comp"])

    # VGG16_model = Converted_VGG16(fe_layer = config_dic["xDNN"]["Feature_Extraction_Layer"], numClasses = 2)

    # VGG16_model.eval()
    device = torch.device(torch_device if torch.cuda.is_available() else "cpu")
    VGG16_model = VGG16_model.to(device)
    print(VGG16_model)
    
    # Fine Tuning Setup
    if config_dic["Fine_Tuning"]["Fine_Tuning"]:
        gf.freeze_layers(VGG16_model, config_dic["Fine_Tuning"]["UnFreezing_Threshold_Layer"])
    if config_dic["PCA"]["PCA_Emulator"] and not config_dic["PCA"]["Train"]:
        for param in VGG16_model.pca_emulator.parameters():
            param.requires_grad = False
    
    # Parallel Computing
    if config_dic["General"]["Parallel_Computing"]:
        VGG16_model = gf.make_data_parallel(VGG16_model,[1])
    
    device=next(VGG16_model.parameters()).device
    config_dic["Hardware"]["Device"] = device
    
    # Train the Model
    if config_dic["General"]["Train"]:
        history, min_val_loss = train_model(VGG16_model, resultsDir, train_dl, val_dl, config_dic)
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

# config_dic = json.load(open(config_file_path))
# config_dic["xDNN"]["Base_Prototypes_path"] = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/COVID-QU-Ex_Dataset/Prototype-Based_Training/Optuna/COVID-QU-Ex_Dataset_Prot-Based_Train_Test_v2/Trial_2/Base_Line/T1/xDDN_Offline_OutliersFilt/Output_xDNN_LearningDic"
# # run(config_dic)

# val_loss, config_dic = run(config_dic=config_dic, 
#                                 batch_size=32, lr=1e-6,
#                                 layer_from_FT = "29.weight",
#                                 weights="imagenet",
#                                 exp_name= "Test_LossFunctionOptimization")