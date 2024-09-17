import sys
sys.path.insert(0, '/nfs/home/nvasconcellos.it/softLinkTests/X-MAN/Prototype-Based_Training/src')

from X_MAN.Explanations import CNN_LIME
from X_MAN import VGG16_FE_Layers

Covid_image = "/nfs/home/nvasconcellos.it/softLinkTests/Datasets/COVID-QU-Ex_Dataset/Lung_Segmentation_Data/Lung_Segmentation_Data/test/COVID-19/covid_989.png"
Non_Covid_image = "/nfs/home/nvasconcellos.it/softLinkTests/Datasets/COVID-QU-Ex_Dataset/Lung_Segmentation_Data/Lung_Segmentation_Data/test/Non-COVID/non_COVID (3300).png"
Normal_image = "/nfs/home/nvasconcellos.it/softLinkTests/Datasets/COVID-QU-Ex_Dataset/Lung_Segmentation_Data/Lung_Segmentation_Data/test/Normal/Normal (84).png"

# image_path = Covid_image
# image_path = Non_Covid_image
image_path = Normal_image


output_dir = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/results/LIME/CNN/Test_Normal_hideColor_0"

weights_path = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/COVID-QU-Ex_Dataset/Prototype-Based_Training/Test_LossFunctionOptimization/checkpoints/best_model.pt"

CNN_LIME.explain_prediction(image_path, 3, output_dir, 
                       weights_path = weights_path, 
                       fe_layer = VGG16_FE_Layers.LAST_CONV,
                       model = None,
                       device_indx = 0, classes_names=["Covid", "Non-Covid", "Normal"], 
                       numSamples = 10000, numFeatures = 15, 
                       batchSize = 32, hide_color = 0, 
                       numFeatures_explainer = 100000)