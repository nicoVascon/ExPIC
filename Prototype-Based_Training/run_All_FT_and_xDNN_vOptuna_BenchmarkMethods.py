import os

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

import traceback
import pickle
import sys

import json

import optuna

import torch

from copy import deepcopy

sys.path.insert(0, '/nfs/home/nvasconcellos.it/softLinkTests/X-MAN/Prototype-Based_Training/src')

mean_accuracy_list = []
xDNN_accs_List = []

initialLayerIndex = 0
initialTraining = 1

def objective(trial):
    print("Start objetive")
    global mean_accuracy_list
    global xDNN_accs_List
    global initialLayerIndex
    global initialTraining

    global config_dic

    # folder = '/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/NeuralNetJournal_Dataset_Exp/Prototype-Based_Training/From_xDNN_Offline/Optuna/%s/' % studyName
    # folder = '/nfs/data/share/nvasconcello/NeuralNetJournal_Dataset_Exp/Prototype-Based_Training/From_xDNN_Offline/Optuna/%s/' % studyName
    # folder = '/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/NeuralNetJournal_Dataset_Exp/Prototype-Based_Training/From_xDNN_Offline/Tests_by_Parts/Traditional_FineTuning_NewSplit/Optuna/%s/' % studyName
    # ExperimentFolderName = 'From_fc2_EachLayer_Trial_%d' % trial.number
    # folder = '/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/COVID-QU-Ex_Dataset/Prototype-Based_Training/Optuna/%s/' % studyName
    folder = '/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/Benchmark_Datasets/CBIS_DDSM_Dataset/Prototype-Based_Training/Test_1/xDNN_Offline/Optuna/%s/' % studyName
    
    if trial is not None:
        trial_number = str(trial.number)
    else:
        # trial_number = "Manual_1"
        trial_number = "15"
    
    pc_name = ""
    
    ExperimentFolderName = 'Trial_%s' % trial_number
    folderDir = folder + ExperimentFolderName + '/'
    
    if not os.path.exists(folderDir):
        os.makedirs(folderDir)
    
    if trial is not None:
        with open(folderDir + 'trial_object', 'wb') as file:
            pickle.dump(trial, file)
    
    print("Before Try")
    try:
        from X_MAN.FineTuning import FineTuning_VGG16_vProt_BasedDS_PyTorch as ft
        from X_MAN.xDNN.utils import Feature_Extraction_VGG16_PyTorch_Batch as featureExtraction
        from X_MAN.xDNN.utils import xDNN_run_vOptuna as xDNN_run
        import ResultsGenerator
        from ResultsGenerator import Experiment

        trial_batch_size = trial.suggest_int("batch_size",20, 40,log=False)
        # trial_batch_size = 64
        
        trial_lr = trial.suggest_float("learning_rate", 7e-4, 1e-1, log=True)
        # trial_lr = 5e-05

        trial_CCE_Alpha = trial.suggest_float("CDE_Alpha", 5e-6, 1e-3, log=True)
        # trial_CCE_Alpha = 1e-7
        
        # trial_CLAHE_clip_limit = trial.suggest_float("CLAHE_clip_limit", 0.01, 0.1, log=False)
        
        layers_From_FT = ["36.weight", "34.weight", "29.weight", "27.weight", "25.weight", "22.weight", "20.weight", "18.weight", "15.weight", "13.weight", "11.weight", "8.weight", "6.weight", "3.weight", "1.weight"]
        # layers_From_FT = ["36.weight", "34.weight", "29.weight", "27.weight", "25.weight", "22.weight", "20.weight", "18.weight", "15.weight", "13.weight", "11.weight"]
        # layers_From_FT = ["36.weight", "34.weight", "29.weight", "27.weight", "25.weight"]
        
        # expts = next(os.walk(folderDir))[1]
        # expts.sort()
        # for exp_name in expts:
            
        
        mean_accuracy_list = []
        mean_numProts_list = []
        
        ft_name = "Optuna/" + studyName + "/" + ExperimentFolderName


        # config_dic = deepcopy(config_dic_orig)
        config_dic = json.load(open(config_file_path))

        
        if config_dic["General"]["TensorBoard"]:
            import tensorflow as tf
            import X_MAN.FineTuning.utils.TensorBoard_Functions as tbf
            base_algoritm_dir = config_dic["Directories"]["base_algoritm_dir"]
            xDNN_log_dir = base_algoritm_dir + "/Tensorboard/" + ft_name + "/xDNN"
            if not os.path.exists(xDNN_log_dir):
                os.makedirs(xDNN_log_dir)
            if len(os.listdir(xDNN_log_dir)) > 0:
                import datetime
                xDNN_log_dir = xDNN_log_dir + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                os.makedirs(xDNN_log_dir)
            xDNN_summary_writer = tf.summary.create_file_writer(xDNN_log_dir)

        if config_dic["xDNN"]["Feature_Extraction_Layer"] == "last_conv":
            initialLayerIndex = 2
        
        for layer_index in range(initialLayerIndex, len(layers_From_FT)):
            layer_from_FT = layers_From_FT[layer_index]
            
            print("oiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")

            # config_dic = deepcopy(config_dic_orig)
            config_dic = json.load(open(config_file_path))

            # CCE Loss parameters Setup
            config_dic["Loss"]["alpha"] = trial_CCE_Alpha
            # config_dic["Preprocessing"]["Normalization"]["CLAHE_Clip_Limit"] = trial_CLAHE_clip_limit

            if (config_dic["xDNN"]["Feature_Extraction_Layer"] == "last_fc" and layer_from_FT == '36.weight') or (config_dic["xDNN"]["Feature_Extraction_Layer"] == "last_conv" and layer_from_FT == "29.weight"):
            # if (config_dic["xDNN"]["Feature_Extraction_Layer"] == "last_fc" and layer_from_FT == '36.weight') or (config_dic["xDNN"]["Feature_Extraction_Layer"] == "last_conv" and layer_from_FT == "1.weight"):
                best_checkpoint_dir = "imagenet"
                exp_name = 'Base_Line/T1'
                (X_train, y_train, X_val, y_val) = featureExtraction.run(folderDir + exp_name + '/',
                                    fe_layer = config_dic["xDNN"]["Feature_Extraction_Layer"],
                                    config_dic = config_dic)
                result = xDNN_run.run(folderDir + exp_name + '/', 
                                    single_training=True, 
                                    model_name=config_dic["xDNN"]["Name"],
                                    fe_layer = config_dic["xDNN"]["Feature_Extraction_Layer"],
                                    # num_PCA_Components = config_dic["PCA"]["New_Num_Comp"],
                                    # feature_vectors = (X_train, y_train, 
                                    #                    X_val, y_val),
                                    xDNN_Subprocesses = config_dic["xDNN"]["xDNN_Subprocesses"],
                                    config_dic = config_dic)
            else:
                exp_name = 'FineTuning_'+ layers_From_FT[layer_index - 1] + ('_To_End_%s/T1' % (pc_name))
            
                checkpoints_dir = folderDir + exp_name + '/checkpoints'
                checkpoints = os.listdir(checkpoints_dir)
                checkpoints.sort()
                best_checkpoint = checkpoints[-1]
                best_checkpoint_dir = checkpoints_dir + '/' + best_checkpoint
                
                # prototypes_path = folderDir + exp_name + '/' + config_dic["xDNN"]["Name"] + "/Output_xDNN_LearningDic"
                # config_dic["xDNN"]["Base_Prototypes_path"] = prototypes_path
            prototypes_path = folderDir + exp_name + '/' + config_dic["xDNN"]["Name"] + "/Output_xDNN_LearningDic"
            config_dic["xDNN"]["Base_Prototypes_path"] = prototypes_path
            
            exp_name = 'FineTuning_'+ layer_from_FT +('_To_End_%s/T1' % (pc_name))
            
            print(exp_name)
            print("Prototypes_Path: " + config_dic["xDNN"]["Base_Prototypes_path"])
            
            val_loss, config_dic = ft.run(config_dic=config_dic, 
                                    batch_size=trial_batch_size, lr=trial_lr,
                                    layer_from_FT = layer_from_FT,
                                    weights=best_checkpoint_dir,
                                    exp_name= ft_name + "/" + exp_name)            
            # config_dic["PCA"]["V_Matrix"] = None
            (X_train, y_train, X_val, y_val) = featureExtraction.run(folderDir + exp_name + '/',
                                  fe_layer = config_dic["xDNN"]["Feature_Extraction_Layer"],
                                  config_dic = config_dic)
            result = xDNN_run.run(folderDir + exp_name + '/', 
                                  single_training=True, 
                                  model_name=config_dic["xDNN"]["Name"],
                                  fe_layer = config_dic["xDNN"]["Feature_Extraction_Layer"],
                                #   num_PCA_Components = config_dic["PCA"]["New_Num_Comp"],
                                #   feature_vectors = (X_train, y_train, 
                                #                      X_val, y_val),
                                  xDNN_Subprocesses = config_dic["xDNN"]["xDNN_Subprocesses"],
                                  config_dic = config_dic)
            
            if best_checkpoint_dir != "imagenet":
                os.remove(best_checkpoint_dir)
            
            mean_accuracy = result["AvrgAccuracy"]
            # mean_Non_Covid_Prototypes = result["Prototypes"]["Non-Covid"]
            # mean_Covid_Prototypes = result["Prototypes"]["Covid"]
            numOfPrototypes_mean = result["Prototypes"]
            
            # if trial is not None:
            #     trial.report(mean_accuracy, layer_index)
            
            if config_dic["General"]["TensorBoard"]:
                # import tensorflow as tf
                # import TensorBoard_Functions as tbf
                # base_algoritm_dir = config_dic["Directories"]["base_algoritm_dir"]
                # xDNN_log_dir = base_algoritm_dir + "/Tensorboard/" + ft_name + "/xDNN"
                # if not os.path.exists(xDNN_log_dir):
                #     os.makedirs(xDNN_log_dir)
                #     xDNN_summary_writer = tf.summary.create_file_writer(xDNN_log_dir)            
                # val_dic = {"xDNN_Accuracy": mean_accuracy, "Non_Covid_Prototypes": mean_Non_Covid_Prototypes, "Covid_Prototypes": mean_Covid_Prototypes}     
                
                xDNN_dict = {"Prototypes_" + class_name : numOfPrototypes for class_name, numOfPrototypes in numOfPrototypes_mean.items()}
                xDNN_dict["xDNN_Accuracy"] = mean_accuracy
                           
                tbf.write_scalars(xDNN_summary_writer, xDNN_dict, layer_index)
            
            if trial is not None:
                with open(folderDir + 'trial_object', 'wb') as file:
                    pickle.dump(trial, file)

            mean_accuracy_list.append(mean_accuracy)
            mean_numProts_list.append(sum(numOfPrototypes_mean.values()))
            
            initialTraining = 1
            
            # Importing gc module
            import gc            
            # Returns the number of objects it has collected and deallocated
            collected = gc.collect()            
            # Prints Garbage collector
            print("Garbage collector: collected %d objects." % collected)
            torch.cuda.empty_cache()
        
        # Delete the Checkpoint of the Last Re-trained Layer
        exp_name = 'FineTuning_'+ layers_From_FT[layer_index] + ('_To_End_%s/T1' % (pc_name))            
        checkpoints_dir = folderDir + exp_name + '/checkpoints'
        checkpoints = os.listdir(checkpoints_dir)
        checkpoints.sort()
        best_checkpoint = checkpoints[-1]
        best_checkpoint_dir = checkpoints_dir + '/' + best_checkpoint
        os.remove(best_checkpoint_dir)
        
        # outputName = 'Results_NeuralNetJournal_Dataset_xDNN_Offline_ProtBasedTrain_OptunaResults_Trial_%d.xlsx' % trial.number
        outputName = 'Results_%s_OptunaResults_Trial_%s_%s.xlsx' % (studyName, trial_number, pc_name)
        outputDir = folderDir + outputName
        results_generator = ResultsGenerator.ExcelResultsGenerator(outputDir)
        
        # Experiment_Name = 'xDNN Offline'
        Experiment_Name = config_dic["xDNN"]["Name"]
        FolderNamePatern = config_dic["xDNN"]["Name"]
        experiment = Experiment(Experiment_Name, folderDir, FolderNamePatern)
        results_generator.addExperiment(experiment)
        
        results_generator.generateResults()
        print('End Script!!!')

        initialLayerIndex = 0
        best_layer_index = torch.argmax(torch.Tensor(mean_accuracy_list)).item()
        return mean_accuracy_list[best_layer_index], mean_numProts_list[best_layer_index]

    except KeyboardInterrupt:
        print('\nInterrupted')        
        print("\nScript in Pause...")
        InvalidOption = True
        while InvalidOption:
            print("\nDo you want to Resume the program?")
            option = input("\n[Y] Yes\n[N] No, Exit\nSelect an option: ").upper()
            InvalidOption = (option != 'Y' and option != 'N')
            if InvalidOption:
                print("\nInvalid Option!!!\nTry Again\n")
        if option == 'Y':
            invalidNumber = True
            while invalidNumber:
                try:
                    initialTraining = int(input("\nEnter the initial training number: "))
                    invalidNumber = False
                except Exception:
                    print("\nInvalid Number!!\n\nTry Again...")
                    
            print("\n------------------------ Program Resumed!!!! ------------------------\n")
            # return objective(trial)
        else:
            sys.exit(0)
            # return None
            
    except optuna.TrialPruned:
        # Delete the Checkpoint of the Last Re-trained Layer
        if best_checkpoint_dir != "imagenet":
            os.remove(best_checkpoint_dir)
        # Delete the Checkpoint of the Current Re-trained Layer
        exp_name = 'FineTuning_'+ layers_From_FT[layer_index] + ('_To_End_%s/T1' % (pc_name))            
        checkpoints_dir = folderDir + exp_name + '/checkpoints'
        checkpoints = os.listdir(checkpoints_dir)
        checkpoints.sort()
        if len(checkpoints) > 0:
            best_checkpoint = checkpoints[-1]
            best_checkpoint_dir = checkpoints_dir + '/' + best_checkpoint
            os.remove(best_checkpoint_dir)
        
        if len(mean_accuracy_list) > 0:
            best_layer_index = torch.argmax(torch.Tensor(mean_accuracy_list)).item()
            return mean_accuracy_list[best_layer_index], mean_numProts_list[best_layer_index]
        else:
            raise optuna.TrialPruned()

    except Exception as e:
        error_file_dir = folderDir + 'errors.txt'
        if not os.path.exists(error_file_dir):
            if not os.path.exists(folderDir):
                os.makedirs(folderDir)
            error_file = open(folderDir + 'errors.txt', 'w+')
        else:
            error_file = open(folderDir + 'errors.txt', 'a+')

        import datetime
        curr_dt_time = datetime.datetime.now()
        error_file.write("\nDate: " + str(curr_dt_time) + '\n\n')
        error_file.write("Exception in user code:\n")
        error_file.write("-"*60 + '\n')
        traceback.print_exc(file=error_file)
        error_file.write("-"*60 + '\n')
        error_file.close()

        # sys.exit(0)
        if len(mean_accuracy_list) > 0:
            best_layer_index = torch.argmax(torch.Tensor(mean_accuracy_list)).item()
            return mean_accuracy_list[best_layer_index], mean_numProts_list[best_layer_index]
        else:
            print("Error: ", e)
            raise optuna.TrialPruned()

config_file_path = "/nfs/home/nvasconcellos.it/softLinkTests/X-MAN/Prototype-Based_Training/src/X_MAN/FineTuning/config/INbreast/parameters_Evolving_PCA88c_PyTorch_GPU0_LowEpochs.json"

config_dic_orig = json.load(open(config_file_path))

studyName = "INbreast_Dataset_Prot-Based_TrainWithPCA_Torch_88c_BatchSizeLearnRateCDEAlpha_Study_LowEpochs_FVFromLastConv_xDNNEvolving"

study = optuna.create_study(directions=['maximize', 'minimize'],storage="sqlite:////nfs/home/nvasconcellos.it/softLinkTests/xDNN_test.db",study_name=studyName, load_if_exists=True)

study.optimize(objective, n_trials=100)

# objective(None)