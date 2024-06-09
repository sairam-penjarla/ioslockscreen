import numpy as np
import cv2
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def get_model(model_path):
    """
    Load and return the image segmentation model.

    Args:
        model_path (str): Path to the model file.

    Returns:
        ImageSegmenter: Loaded image segmenter model.
    """
    segmentation_base_options = python.BaseOptions(model_asset_path=model_path)
    seg_options = vision.ImageSegmenterOptions(base_options=segmentation_base_options, output_category_mask=True)
    return vision.ImageSegmenter.create_from_options(seg_options)

def crop_background(img):
    """
    Crop the background of the image to the specified aspect ratio.

    Args:
        img (Image): Input image.

    Returns:
        tuple: Cropped image, new width, and new height.
    """
    img = np.asarray(img)
    height, width, _ = img.shape
    crop_height = height
    crop_width = int(crop_height * 9 / 16)
    start_row = 0
    start_col = int((width - crop_width) / 2)
    crop = img[start_row:start_row + crop_height, start_col:start_col + crop_width]
    crop = Image.fromarray(crop)
    return crop, crop_width, crop_height

def get_image_segments(model, image):
    """
    Get image segments using the segmentation model.

    Args:
        model (ImageSegmenter): The image segmenter model.
        image (Image): Input image.

    Returns:
        np.ndarray: Segmented image.
    """
    MASK_COLOR = (0, 0, 0)  # black
    BG_COLOR = (1, 1, 1)  # white
    segmentation_result = model.segment(image)
    category_mask = segmentation_result.category_mask
    image_data = image.numpy_view()
    fg_image = np.zeros(image_data.shape, dtype=np.uint8)
    fg_image[:] = MASK_COLOR
    bg_image = np.zeros(image_data.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2
    segmented_image = np.where(condition, fg_image, bg_image)
    return segmented_image

def apply_segmentation(original_image, segmentation_mask):
    """
    Apply the segmentation mask to the original image.

    Args:
        original_image (Image): The original image.
        segmentation_mask (np.ndarray): The segmentation mask.

    Returns:
        np.ndarray: The image with the segmentation mask applied.
    """
    original_img_as_numpy_array = np.asarray(original_image)
    masked_region = segmentation_mask <= 0
    original_img_with_mask = np.zeros_like(original_img_as_numpy_array)
    original_img_with_mask[masked_region] = original_img_as_numpy_array[masked_region]
    return original_img_with_mask

def resize_template(template, new_height):
    """
    Resize the template image to a new height while maintaining aspect ratio.

    Args:
        template (Image): The template image.
        new_height (int): The new height for the template.

    Returns:
        Image: The resized template image.
    """
    aspect_ratio = 9 / 18
    new_width = int(new_height * aspect_ratio)
    img = cv2.resize(np.asarray(template), (new_width, new_height))
    return Image.fromarray(img)
