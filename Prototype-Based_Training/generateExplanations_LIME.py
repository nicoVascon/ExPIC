import sys
sys.path.insert(0, '/nfs/home/nvasconcellos.it/softLinkTests/X-MAN/Prototype-Based_Training/src')

from X_MAN import xDNN_LIME
from X_MAN import VGG16_FE_Layers

Covid_image = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/dataset_NeuralNet_Journal/Split_Train72.2_Test27.8/train/1_CT_COVID/2020.02.10.20021584-p6-52%14.png"
Non_Covid_image = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/dataset_NeuralNet_Journal/Split_Train72.2_Test27.8/train/0_CT_NonCOVID/5%6.jpg"

# image_path = Covid_image
image_path = Non_Covid_image
output_dir = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/results/LIME/Test_NonCovid_hide_color_200"
xDNN_learned_dict_path = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/NeuralNetJournal_Dataset_Exp/Prototype-Based_Training/From_xDNN_Offline/Optuna/Prot-Based_Train_FT_EachLayer_LastConv_xDNN_Online/From_fc2_EachLayer_Trial_2/FineTuning_8.weight_To_End/T1/xDDN_Offline_ProtFilt_OutliersFilt/Output_xDNN_LearningDic"

weights_path = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/NeuralNetJournal_Dataset_Exp/Prototype-Based_Training/From_xDNN_Offline/Optuna/Prot-Based_Train_FT_EachLayer_LastConv_xDNN_Online/From_fc2_EachLayer_Trial_2/FineTuning_8.weight_To_End/T1/checkpoints/best_model.pt"

xDNN_LIME.explain_prediction(image_path, xDNN_learned_dict_path, output_dir, 
                       weights_path = weights_path, 
                       fe_layer = VGG16_FE_Layers.LAST_CONV,
                       model = None,
                       device_indx = 0, classes_names=["0_Non-COVID", "1_COVID"], 
                       numSamples = 10000, numFeatures = 20, 
                       batchSize = 32, hide_color = 0, 
                       numFeatures_explainer = 100000)