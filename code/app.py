import cv2
import streamlit as st
import numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import io
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def get_image_segments(image):
    MASK_COLOR = (0, 0, 0) # black
    BG_COLOR = (1, 1, 1) # white
    # Retrieve the masks for the segmented image
    segmentation_result = segmenter.segment(image)
    category_mask = segmentation_result.category_mask

    # Generate solid color images for showing the output segmentation mask.
    image_data = image.numpy_view()
    fg_image = np.zeros(image_data.shape, dtype=np.uint8)
    fg_image[:] = MASK_COLOR
    bg_image = np.zeros(image_data.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR

    condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2
    segmented_image = np.where(condition, fg_image, bg_image)
    return segmented_image

def draw_contours(original_image, segmentation_mask, bbox_scale=1.0):
    segmentation_mask = cv2.cvtColor(segmentation_mask, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(segmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = contours[0]
    
    # Get the bounding box of the contour
    x, y, w, h = cv2.boundingRect(c)
    
    # Scale the bounding box if desired
    if bbox_scale != 1.0:
        cx = x + w / 2
        cy = y + h / 2
        w *= bbox_scale
        h *= bbox_scale
        x = int(cx - w / 2)
        y = int(cy - h / 2)
    
    hight_gap = 200
    # Crop the images using the bounding box
    cropped_image = original_image[y-hight_gap:y+h, x:x+w].copy()
    cropped_mask = segmentation_mask[y-hight_gap:y+h, x:x+w].copy()
    
    contours, _ = cv2.findContours(cropped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        cv2.drawContours(cropped_image, [c], -1, [255, 255, 255], 60)
        cv2.drawContours(cropped_mask, [c], -1, [255, 255, 255], 60)
    
    return cropped_image,cropped_mask

def apply_segmentation(original_image, segmentation_mask):
    segmentation_mask = cv2.blur(segmentation_mask, (100, 100))
    original_img_as_numpy_array = np.asarray(original_image)
    
    
    
    masked_region = segmentation_mask > 0
    original_img_with_mask = np.zeros_like(original_img_as_numpy_array)
    original_img_with_mask[masked_region] = original_img_as_numpy_array[masked_region]
    
    original_img_with_mask, segmentation_mask = draw_contours(original_img_with_mask, segmentation_mask)
    return original_img_with_mask

def crop_background(img, blur_bg_switch, blur_level=0):
    img = np.asarray(img)
    # Get the dimensions of the input image
    height, width, _ = img.shape

    # Calculate the height and width of the crop
    crop_width = width
    crop_height = int(crop_width * 9 / 16)

    # Determine the starting row and column of the crop
    start_row = int((height - crop_height) / 2)
    start_col = 0

    # Use numpy indexing to extract the crop
    crop = img[start_row:start_row+crop_height, start_col:start_col+crop_width]
    if blur_bg_switch:
        crop = cv2.blur(crop, (blur_level, blur_level))
    crop = Image.fromarray(crop)
    return crop, crop_width, crop_height

def combine_both(cropped_background, img2):
    background, crop_width, crop_height = cropped_background
    
    #####################
    new_height = crop_height
    new_width = int(new_height * img2.shape[1] / img2.shape[0])
    img2_resized = cv2.resize(img2, (new_width, new_height))
    mask = Image.fromarray((img2_resized != [0, 0, 0]).any(axis=2))
    foreground = Image.fromarray(img2_resized)
    foreground.putalpha(mask)
    #####################

    # Calculate the coordinates for the top-left corner
    x = int((background.width - foreground.width) / 2)
    y = int((background.height - foreground.height) / 2)

    background.paste(foreground, (x, y), foreground)
    return background


def main_loop():
    st.title("Youtube Thumbnail Creator")

    foreground_file = st.sidebar.file_uploader("Upload Your Foreground", type=['jpg', 'png', 'jpeg'])
    background_file = st.sidebar.file_uploader("Upload Your Background", type=['jpg', 'png', 'jpeg'])
    blur_bg_switch = st.sidebar.checkbox('Blur background')
    if not foreground_file:
        return None
    if not background_file:
        return None
    input_image = Image.open(foreground_file)
    input_bg = Image.open(background_file)
    if blur_bg_switch:
        blur_level = st.sidebar.slider('Blur level', 1, 100, 1)
        cropped_background = crop_background(input_bg, blur_bg_switch, blur_level)
    else:
        cropped_background = crop_background(input_bg, blur_bg_switch)
    
    with st.spinner('Wait for it...'):
        input_image_numpy = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(input_image))
        segmentation_mask = get_image_segments(input_image_numpy)
        input_seg_and_contour = apply_segmentation(input_image, segmentation_mask)
        thumbnail = combine_both(cropped_background, input_seg_and_contour)
        # Convert the PIL image to bytes
        img_bytes = io.BytesIO()
        thumbnail.save(img_bytes, format="PNG")
        img_bytes = img_bytes.getvalue()

    st.text("Your Thumbnail")
    st.image(thumbnail)
    # Add a download button
    st.download_button(
        label="Download Image",
        data=img_bytes,
        file_name="Thumbnail.png",
        mime="image/png"
    )


if __name__ == '__main__':
    model_path = 'models/square_model.tflite'
    segmentation_base_options = python.BaseOptions(model_asset_path='../model/square_model.tflite')
    seg_options = vision.ImageSegmenterOptions(base_options=segmentation_base_options, output_category_mask=True)
    segmenter = vision.ImageSegmenter.create_from_options(seg_options)
    main_loop()