import optuna 
import pickle

study_path = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/Benchmark_Datasets/CBIS_DDSM_Dataset/Prototype-Based_Training/Test_1/xDNN_Offline/Optuna/INbreast_Dataset_Prot-Based_TrainWithPCA_Torch_136c_BatchSizeLearnRateCDEAlpha_Study_LowEpochs_FVFromLastConv_xDNNOffline"

studyName = "INbreast_Dataset_Prot-Based_TrainWithPCA_Torch_136c_BatchSizeLearnRateCDEAlpha_Study_LowEpochs_FVFromLastConv_xDNNOffline"

# studyName = "Study-Extra_NeuralNetJournal_Dataset_Traditional_FineTuning"

study = optuna.create_study(directions=['maximize', 'minimize'],storage="sqlite:////nfs/home/nvasconcellos.it/softLinkTests/xDNN_test.db",study_name=studyName, load_if_exists=True)

trials_results = {
    257: [0.760684, 27], 
    258: [0.760684, 37], 
    264: [0.769231, 46], 
    265: [0.760684, 13], 
    266: [0.769231, 41], 
    
}

for trial_num in trials_results:
    trial_obj_path = f"{study_path}/Trial_{trial_num}/trial_object"
    with open(trial_obj_path, 'rb') as f:
        trial_obj = pickle.load(f)
    study.tell(trial_obj, trials_results[trial_num])