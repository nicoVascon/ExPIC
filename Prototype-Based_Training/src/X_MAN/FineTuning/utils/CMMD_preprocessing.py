import numpy as np
from X_MAN.FineTuning.utils.triangle_thresholding import ComputeTriangleThreshold
from skimage import measure, morphology
from skimage.transform import resize
import skimage as ski

# Function to pad an image to make it square
def pad_to_square(image, desired_size=2048):
    # Get current dimensions of the image
    current_height, current_width = image.shape[:2]

    # If the width is less than the desired size, pad on the right side
    if current_width < desired_size:
        padding_width = desired_size - current_width
        # Create padding array
        if len(image.shape) == 3:
            padding = np.zeros((current_height, padding_width, image.shape[2]), dtype=image.dtype)
        else:
            padding = np.zeros((current_height, padding_width), dtype=image.dtype)
        # Concatenate the image with the padding on the right side
        padded_image = np.hstack((image, padding))
    elif current_width > desired_size:
        # If the width is greater than the desired size, crop the right side
        padded_image = image[:, :desired_size]
    else:
        padded_image = image  # No padding needed

    return padded_image

def preprocess_image(img : np.ndarray, new_height=2048, CLAHE_clip_limit=0.015):
    foreground_mask = ComputeTriangleThreshold.compute_mask(img)
    foreground_img = img * foreground_mask

    # Compute connected components
    labels = measure.label(foreground_mask)
    # Get the properties of the connected components
    regions = measure.regionprops(labels)
    # Find the largest connected component (assumed to be the breast region)
    largest_region = max(regions, key=lambda region: region.area)
    # Create a binary mask for the largest connected component (the breast region)
    breast_mask = np.zeros_like(labels, dtype=bool)
    breast_mask[labels == largest_region.label] = True
    # Optionally, remove small objects/noise
    breast_mask = morphology.remove_small_objects(breast_mask, min_size=500)
    # Apply the mask to the original image to remove other small regions
    breast_image = np.copy(foreground_img)
    breast_image[~breast_mask] = 0  # Set pixels outside the breast region to 0

    # Locate the breast region at the left side of the image
    width = breast_mask.shape[1]
    num_pixels_left = np.sum(breast_mask[:, :width//2])
    num_pixels_right = np.sum(breast_mask[:, width//2:])
    breast_in_left = num_pixels_left > num_pixels_right
    if not breast_in_left:
        breast_mask_in_left = np.fliplr(breast_mask)
        breast_image_in_left = np.fliplr(breast_image)
    else:
        breast_mask_in_left = breast_mask
        breast_image_in_left = breast_image
    
    # Crop the breast region and resize it to a square image
    # Get the bounding box of the breast region (height range)
    rows = np.any(breast_mask_in_left, axis=1)
    min_row, max_row = np.where(rows)[0][[0, -1]]
    # Crop the image to the height of the breast
    cropped_image = breast_image_in_left[min_row:max_row + 1, :]
    # Resize the cropped image to 2048 pixels in height while maintaining the aspect ratio
    aspect_ratio = cropped_image.shape[1] / cropped_image.shape[0]
    new_width = int(new_height * aspect_ratio)
    # Resize the image while keeping the aspect ratio
    resized_image = resize(cropped_image, (new_height, new_width), anti_aliasing=True, preserve_range=True).astype(breast_image_in_left.dtype)
    # Pad the right side with zero-intensity pixels to make the image square (2048x2048)
    square_image = pad_to_square(resized_image, desired_size=new_height)

    # Apply adaptive histogram equalization and scale the image to the range [0, 1]
    img_filtered = ski.exposure.equalize_adapthist(square_image, clip_limit=CLAHE_clip_limit)
    img_scaled = (img_filtered - np.min(img_filtered)) / (np.max(img_filtered) - np.min(img_filtered))
    return img_scaled