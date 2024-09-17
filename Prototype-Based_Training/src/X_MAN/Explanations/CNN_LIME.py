import matplotlib.pyplot as plt
import numpy as np

import os, json
import pickle

import torch
from lime import lime_image
from skimage.segmentation import mark_boundaries

from X_MAN import Converted_VGG16
from X_MAN import VGG16_FE_Layers
from X_MAN import xDNN

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input


def explain_prediction(img_path : str, numClasses : int, output_dir : str, 
                       weights_path : str = "Pretrained", fe_layer : str = None,
                       model : torch.nn.Module = None,
                       device_indx : int = 0, 
                       classes_names : list = None,
                       numSamples : int = 1000, numFeatures : int = 5, 
                       numFeatures_explainer : int = 100000, batchSize : int = 32,
                       hide_color : int = 0) -> None:
    
    if model is None:
        if fe_layer is None:
            raise Exception("When model is None, fe_layer must be provided")
        
        if fe_layer not in VGG16_FE_Layers.available_layers():
            raise Exception("Error!!!! Unvalid fe_layer. Available fe_layers: " + VGG16_FE_Layers.available_layers())   
        
        if weights_path != "Pretrained" and os.path.exists(weights_path):
            model = Converted_VGG16(weights_path, fe_layer = fe_layer, numClasses = numClasses)
        else:
            model = Converted_VGG16(fe_layer = fe_layer, numClasses = numClasses)            
        
        avail_devices_count = torch.cuda.device_count()
        actual_device_indx = device_indx if device_indx < avail_devices_count else avail_devices_count - 1
        torch_device = "cuda:" + str(actual_device_indx)
        device = torch.device(torch_device if torch.cuda.is_available() else "cpu")
        model.to(device)
    model.eval()
    device=next(model.parameters()).device    
   
    
    # Load image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    
    # Create Perturbations Directory 
    import datetime
    perturbations_dir = output_dir + "/Perturbations_" + str(datetime.datetime.now()).replace(" ", "_").replace(":", "-") + "/"
    if not os.path.exists(perturbations_dir):
        os.makedirs(perturbations_dir)

    global batch_num
    batch_num = 0
    batch_show = [0, 5, 20, 100, 200, 250, 300]
    
    import random
    random.seed(0)
    def batch_predict(images):
        global batch_num
        # for i in images:
        #   plt.imshow(i/255.0)
        
        if batch_num in batch_show:
            plt.imshow(images[0]/255.0)
            plt.savefig(perturbations_dir + f'img0_batch_{batch_num}.png')
        batch_num += 1
        
        batch = torch.stack(tuple(torch.from_numpy(preprocess_input(i).copy()) for i in images), dim=0)    
        batch = batch.to(device)
            
        y = model(batch)
        y = y.cpu().detach().numpy()
        
        return y

    # Initialize LimeImageExplainer
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(x, 
                                            batch_predict, # classification function
                                            top_labels=numClasses, 
                                            hide_color=hide_color, 
                                            num_samples=numSamples, # number of images that will be sent to classification function
                                            random_seed=123,
                                            batch_size = batchSize,
                                            num_features = numFeatures_explainer) 
    
    # ------------------ Visualize Explanation ------------------
    if classes_names is None:
        predicted_class = explanation.top_labels[0]
    else:
        predicted_class = classes_names[explanation.top_labels[0]]
    
    output_dir = output_dir + "/" + str(predicted_class) + "/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Display Original Image
    plt.imshow(x/255.0)
    plt.savefig(output_dir + '/Original.png')
    
    # Positive Boundary
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=numFeatures, hide_rest=False)
    img_boundry1 = mark_boundaries(temp/255.0, mask)
    plt.imshow(img_boundry1)
    plt.savefig(output_dir+ f'img_boundry1_OnlyPositive_PredictedClass_{predicted_class}.png')

    # Complete Boundary
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=numFeatures, hide_rest=False)
    img_boundry2 = mark_boundaries(temp/255.0, mask)
    plt.imshow(img_boundry2)
    plt.savefig(output_dir+ f'img_boundry2_All_PredictedClass_{predicted_class}.png')
