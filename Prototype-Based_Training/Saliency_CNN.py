# Boilerplate imports.
import numpy as np
import PIL.Image
from matplotlib import pylab as P
import torch
from torchvision import models, transforms

# From our repository.
import saliency.core as saliency

# %matplotlib inline

# My Imports

import os
import sys
sys.path.insert(0, '/nfs/home/nvasconcellos.it/softLinkTests/X-MAN/Prototype-Based_Training/src')

import X_MAN
import X_MAN.FineTuning.utils.Generic_Functions as gf
from X_MAN.Models.VGG16.Model.Converted_VGG16 import Converted_VGG16

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Select Device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") # We can modify this line to select other GPU

# Boilerplate methods.
def ShowImage(im, title='', ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    P.imshow(im)
    P.title(title)

def ShowGrayscaleImage(im, title='', ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
    P.title(title)

def ShowHeatMap(im, title, ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    P.imshow(im, cmap='inferno')
    P.title(title)

def LoadImage(file_path):
    im = PIL.Image.open(file_path)
    im = im.resize((299, 299))
    im = np.asarray(im)
    return im

transformer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
def PreprocessImages(images):
    # assumes input is 4-D, with range [0,255]
    #
    # torchvision have color channel as first dimension
    # with normalization relative to mean/std of ImageNet:
    #    https://pytorch.org/vision/stable/models.html
    # images = np.array(images)
    # images = images/255
    # images = np.transpose(images, (0,3,1,2))
    # images = torch.tensor(images, dtype=torch.float32)
    # images = transformer.forward(images)
    # return images.requires_grad_(True)
    images = np.array(images)
    x = preprocess_input(images)
    x = torch.from_numpy(x.copy()).to(torch.float32)
    x = x.to(device)
    return x.requires_grad_(True)

# model = models.inception_v3(pretrained=True, init_weights=False)
# eval_mode = model.eval()

# Load VGG16 Converted
weights_path = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/COVID-QU-Ex_Dataset/Prototype-Based_Training/Test_LossFunctionOptimization/checkpoints/best_model.pt"
fe_layer="last_conv"
model = Converted_VGG16(weights_path, fe_layer = fe_layer, numClasses = 3)
model.eval()
model.to(device)

# Register hooks for Grad-CAM, which uses the last convolution layer
# conv_layer = model.Mixed_7c
last_conv_layer_idx = 29
conv_layer = model.net[last_conv_layer_idx]

conv_layer_outputs = {}
def conv_layer_forward(m, i, o):
    # move the RGB dimension to the last dimension
    conv_layer_outputs[saliency.base.CONVOLUTION_LAYER_VALUES] = torch.movedim(o, 1, 3).cpu().detach().numpy()
def conv_layer_backward(m, i, o):
    # move the RGB dimension to the last dimension
    conv_layer_outputs[saliency.base.CONVOLUTION_OUTPUT_GRADIENTS] = torch.movedim(o[0], 1, 3).clone().cpu().detach().numpy()
conv_layer.register_forward_hook(conv_layer_forward)
# conv_layer.register_full_backward_hook(conv_layer_backward)

class_idx_str = 'class_idx_str'
def call_model_function(images, call_model_args=None, expected_keys=None):
    images = PreprocessImages(images)
    target_class_idx =  call_model_args[class_idx_str]
    output = model(images)
    # m = torch.nn.Softmax(dim=1)
    # output = m(output)
    if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:
        outputs = output[:,target_class_idx]
        grads = torch.autograd.grad(outputs, images, grad_outputs=torch.ones_like(outputs))
        # grads = torch.movedim(grads[0], 1, 3)
        grads = grads[0]
        gradients = grads.cpu().detach().numpy()
        return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
    else:
        one_hot = torch.zeros_like(output)
        one_hot[:,target_class_idx] = 1
        model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)
        return conv_layer_outputs


# Load the image
# im_orig = LoadImage('./doberman.png')
# im_tensor = PreprocessImages([im_orig])

Covid_image = "/nfs/home/nvasconcellos.it/softLinkTests/Datasets/COVID-QU-Ex_Dataset/Lung_Segmentation_Data/Lung_Segmentation_Data/test/COVID-19/covid_989.png"
Non_Covid_image = "/nfs/home/nvasconcellos.it/softLinkTests/Datasets/COVID-QU-Ex_Dataset/Lung_Segmentation_Data/Lung_Segmentation_Data/test/Non-COVID/non_COVID (3300).png"
Normal_image = "/nfs/home/nvasconcellos.it/softLinkTests/Datasets/COVID-QU-Ex_Dataset/Lung_Segmentation_Data/Lung_Segmentation_Data/test/Normal/Normal (84).png"

# image_path = Covid_image
# image_path = Non_Covid_image
image_path = Normal_image

img = image.load_img(image_path, target_size=(224, 224))
im_orig = image.img_to_array(img)
im_orig_expand = np.expand_dims(im_orig, axis=0)
im_tensor = PreprocessImages(im_orig_expand)
# Show the image
ShowImage(im_orig)

predictions = model(im_tensor)
predictions = predictions.cpu().detach().numpy()
prediction_class = np.argmax(predictions[0])
call_model_args = {class_idx_str: prediction_class}

print("Prediction class: " + str(prediction_class))  # Should be a doberman, class idx = 236
im = im_orig.astype(np.float32)

# -------------------------------------------------------------------------------------
# --------------------------------- Directories ---------------------------------------
# -------------------------------------------------------------------------------------
output_dir = "/nfs/home/nvasconcellos.it/softLinkTests/X-MAN/Prototype-Based_Training/Saliency_Results_CNN_Class_" + str(prediction_class) 
import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# -------------------------------------------------------------------------------------
# ------------------------------ Vanilla Gradient & SmoothGrad ------------------------
# -------------------------------------------------------------------------------------
print("Generating Explanation: Vanilla Gradient & SmoothGrad")
# Construct the saliency object. This alone doesn't do anthing.
gradient_saliency = saliency.GradientSaliency()

# Compute the vanilla mask and the smoothed mask.
vanilla_mask_3d = gradient_saliency.GetMask(im, call_model_function, call_model_args)
smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(im, call_model_function, call_model_args)

# Call the visualization methods to convert the 3D tensors to 2D grayscale.
vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_mask_3d)
smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d)

# Set up matplot lib figures.
ROWS = 1
COLS = 2
UPSCALE_FACTOR = 10
P.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))

# Render the saliency masks.
ShowGrayscaleImage(vanilla_mask_grayscale, title='Vanilla Gradient', ax=P.subplot(ROWS, COLS, 1))
ShowGrayscaleImage(smoothgrad_mask_grayscale, title='SmoothGrad', ax=P.subplot(ROWS, COLS, 2))
P.savefig(output_dir + "/Vanilla_Gradient_and_SmoothGrad.png")

# -------------------------------------------------------------------------------------
# -------------------------- Integrated Gradients & SmoothGrad ------------------------
# -------------------------------------------------------------------------------------
print("Generating Explanation: Integrated Gradients & SmoothGrad")

# Construct the saliency object. This alone doesn't do anthing.
integrated_gradients = saliency.IntegratedGradients()

# Baseline is a black image.
baseline = np.zeros(im.shape)

# Compute the vanilla mask and the smoothed mask.
vanilla_integrated_gradients_mask_3d = integrated_gradients.GetMask(
  im, call_model_function, call_model_args, x_steps=25, x_baseline=baseline, batch_size=20)
# Smoothed mask for integrated gradients will take a while since we are doing nsamples * nsamples computations.
smoothgrad_integrated_gradients_mask_3d = integrated_gradients.GetSmoothedMask(
  im, call_model_function, call_model_args, x_steps=25, x_baseline=baseline, batch_size=20)

# Call the visualization methods to convert the 3D tensors to 2D grayscale.
vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_integrated_gradients_mask_3d)
smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_integrated_gradients_mask_3d)

# Set up matplot lib figures.
ROWS = 1
COLS = 2
UPSCALE_FACTOR = 10
P.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))

# Render the saliency masks.
ShowGrayscaleImage(vanilla_mask_grayscale, title='Vanilla Integrated Gradients', ax=P.subplot(ROWS, COLS, 1))
ShowGrayscaleImage(smoothgrad_mask_grayscale, title='Smoothgrad Integrated Gradients', ax=P.subplot(ROWS, COLS, 2))
P.savefig(output_dir + "/Integrated_Gradients_and_SmoothGrad.png")


# -------------------------------------------------------------------------------------
# ------------------------------------ XRAI -------------------------------------------
# -------------------------------------------------------------------------------------
# print("Generating Explanation: XRAI")
# # Construct the saliency object. This alone doesn't do anthing.
# xrai_object = saliency.XRAI()

# # Compute XRAI attributions with default parameters
# xrai_attributions = xrai_object.GetMask(im, call_model_function, call_model_args, batch_size=20)

# # Set up matplot lib figures.
# ROWS = 1
# COLS = 3
# UPSCALE_FACTOR = 20
# P.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))

# # Show original image
# ShowImage(im_orig, title='Original Image', ax=P.subplot(ROWS, COLS, 1))

# # Show XRAI heatmap attributions
# ShowHeatMap(xrai_attributions, title='XRAI Heatmap', ax=P.subplot(ROWS, COLS, 2))

# # Show most salient 30% of the image
# mask = xrai_attributions >= np.percentile(xrai_attributions, 70)
# im_mask = np.array(im_orig)
# im_mask[~mask] = 0
# ShowImage(im_mask, title='Top 30%', ax=P.subplot(ROWS, COLS, 3))
# P.savefig(output_dir + "/XRAI.png")

# ----------------- XRAI Fast -----------------
# print("Generating Explanation: XRAI Fast")
# # Create XRAIParameters and set the algorithm to fast mode which will produce an approximate result.
# xrai_params = saliency.XRAIParameters()
# xrai_params.algorithm = 'fast'

# # Compute XRAI attributions with fast algorithm
# xrai_attributions_fast = xrai_object.GetMask(im, call_model_function, call_model_args, extra_parameters=xrai_params, batch_size=20)

# # Set up matplot lib figures.
# ROWS = 1
# COLS = 3
# UPSCALE_FACTOR = 20
# P.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))

# # Show original image
# ShowImage(im_orig, title='Original Image', ax=P.subplot(ROWS, COLS, 1))

# # Show XRAI heatmap attributions
# ShowHeatMap(xrai_attributions_fast, title='XRAI Heatmap', ax=P.subplot(ROWS, COLS, 2))

# # Show most salient 30% of the image
# mask = xrai_attributions_fast >= np.percentile(xrai_attributions_fast, 70)
# im_mask = np.array(im_orig)
# im_mask[~mask] = 0
# ShowImage(im_mask, 'Top 30%', ax=P.subplot(ROWS, COLS, 3))
# P.savefig(output_dir + "/XRAI_Fast.png")


# # -------------------------------------------------------------------------------------
# # ---------------------------------- Grad-CAM -----------------------------------------
# # -------------------------------------------------------------------------------------
# print("Generating Explanation: Grad-CAM")
# # Compare Grad-CAM and Smoothgrad with Grad-CAM.

# # Construct the saliency object. This alone doesn't do anthing.
# grad_cam = saliency.GradCam()

# # Compute the Grad-CAM mask and Smoothgrad+Grad-CAM mask.
# grad_cam_mask_3d = grad_cam.GetMask(im, call_model_function, call_model_args)
# smooth_grad_cam_mask_3d = grad_cam.GetSmoothedMask(im, call_model_function, call_model_args)

# # Call the visualization methods to convert the 3D tensors to 2D grayscale.
# grad_cam_mask_grayscale = saliency.VisualizeImageGrayscale(grad_cam_mask_3d)
# smooth_grad_cam_mask_grayscale = saliency.VisualizeImageGrayscale(smooth_grad_cam_mask_3d)

# # Set up matplot lib figures.
# ROWS = 1
# COLS = 2
# UPSCALE_FACTOR = 10
# P.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))

# # Render the saliency masks.
# ShowGrayscaleImage(grad_cam_mask_grayscale, title='Grad-CAM', ax=P.subplot(ROWS, COLS, 1))
# ShowGrayscaleImage(smooth_grad_cam_mask_grayscale, title='Smoothgrad Grad-CAM', ax=P.subplot(ROWS, COLS, 2))
# P.savefig(output_dir + "/Grad-CAM.png")


# -------------------------------------------------------------------------------------
# ---------------------------------- Guided IG ----------------------------------------
# -------------------------------------------------------------------------------------
print("Generating Explanation: Guided IG")
# Construct the saliency object. This doesn't yet compute the saliency mask, it just sets up the necessary ops.
integrated_gradients = saliency.IntegratedGradients()
guided_ig = saliency.GuidedIG()

# Baseline is a black image for vanilla integrated gradients.
baseline = np.zeros(im.shape)

# Compute the vanilla mask and the Guided IG mask.
vanilla_integrated_gradients_mask_3d = integrated_gradients.GetMask(
  im, call_model_function, call_model_args, x_steps=25, x_baseline=baseline, batch_size=20)
guided_ig_mask_3d = guided_ig.GetMask(
  im, call_model_function, call_model_args, x_steps=25, x_baseline=baseline, max_dist=1.0, fraction=0.5)

# Call the visualization methods to convert the 3D tensors to 2D grayscale.
vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_integrated_gradients_mask_3d)
guided_ig_mask_grayscale = saliency.VisualizeImageGrayscale(guided_ig_mask_3d)

# Set up matplot lib figures.
ROWS = 1
COLS = 3
UPSCALE_FACTOR = 20
P.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))

# Render the saliency masks.
ShowImage(im_orig, title='Original Image', ax=P.subplot(ROWS, COLS, 1))
ShowGrayscaleImage(vanilla_mask_grayscale, title='Vanilla Integrated Gradients', ax=P.subplot(ROWS, COLS, 2))
ShowGrayscaleImage(guided_ig_mask_grayscale, title='Guided Integrated Gradients', ax=P.subplot(ROWS, COLS, 3))
P.savefig(output_dir + "/Guided_IG.png")


# -------------------------------------------------------------------------------------
# ---------------------------------- Blur IG ------------------------------------------
# -------------------------------------------------------------------------------------
print("Generating Explanation: Blur IG")
# Construct the saliency object. This alone doesn't do anthing.
integrated_gradients = saliency.IntegratedGradients()
blur_ig = saliency.BlurIG()

# Baseline is a black image for vanilla integrated gradients.
baseline = np.zeros(im.shape)

# Compute the vanilla mask and the Blur IG mask.
vanilla_integrated_gradients_mask_3d = integrated_gradients.GetMask(
  im, call_model_function, call_model_args, x_steps=25, x_baseline=baseline, batch_size=20)
blur_ig_mask_3d = blur_ig.GetMask(
  im, call_model_function, call_model_args, batch_size=20)

# Call the visualization methods to convert the 3D tensors to 2D grayscale.
vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_integrated_gradients_mask_3d)
blur_ig_mask_grayscale = saliency.VisualizeImageGrayscale(blur_ig_mask_3d)

# Set up matplot lib figures.
ROWS = 1
COLS = 2
UPSCALE_FACTOR = 10
P.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))

# Render the saliency masks.
ShowGrayscaleImage(vanilla_mask_grayscale, title='Vanilla Integrated Gradients', ax=P.subplot(ROWS, COLS, 1))
ShowGrayscaleImage(blur_ig_mask_grayscale, title='Blur Integrated Gradients', ax=P.subplot(ROWS, COLS, 2))
P.savefig(output_dir + "/Blur_IG.png")


# -------------------------------------------------------------------------------------
# ----------------------------- Smoothgrad with BlurIG --------------------------------
# -------------------------------------------------------------------------------------
print("Generating Explanation: Smoothgrad with BlurIG")
# Compare BlurIG and Smoothgrad with BlurIG. Note: This will take a long time to run.

# Construct the saliency object. This alone doesn't do anthing.
blur_ig = saliency.BlurIG()

# Compute the Blur IG mask and Smoothgrad+BlurIG mask.
blur_ig_mask_3d = blur_ig.GetMask(im, call_model_function, call_model_args, batch_size=20)
# Smoothed mask for BlurIG will take a while since we are doing nsamples * nsamples computations.
smooth_blur_ig_mask_3d = blur_ig.GetSmoothedMask(im, call_model_function, call_model_args, batch_size=20)

# Call the visualization methods to convert the 3D tensors to 2D grayscale.
blur_ig_mask_grayscale = saliency.VisualizeImageGrayscale(blur_ig_mask_3d)
smooth_blur_ig_mask_grayscale = saliency.VisualizeImageGrayscale(smooth_blur_ig_mask_3d)

# Set up matplot lib figures.
ROWS = 1
COLS = 2
UPSCALE_FACTOR = 10
P.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))

# Render the saliency masks.
ShowGrayscaleImage(blur_ig_mask_grayscale, title='Blur Integrated Gradients', ax=P.subplot(ROWS, COLS, 1))
ShowGrayscaleImage(smooth_blur_ig_mask_grayscale, title='Smoothgrad Blur IG', ax=P.subplot(ROWS, COLS, 2))
P.savefig(output_dir + "/Smoothgrad_with_BlurIG.png")
