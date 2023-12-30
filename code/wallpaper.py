import cv2
import streamlit as st
import numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time
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
def apply_segmentation(original_image, segmentation_mask):
    segmentation_mask = cv2.blur(segmentation_mask, (100, 100))
    original_img_as_numpy_array = np.asarray(original_image)
    
    masked_region = segmentation_mask > 0
    original_img_with_mask = np.zeros_like(original_img_as_numpy_array)
    original_img_with_mask[masked_region] = original_img_as_numpy_array[masked_region]
    
    return original_img_with_mask
def crop_background(img):
    img = np.asarray(img)
    # Get the dimensions of the input image
    height, width, _ = img.shape

    # Calculate the height and width of the crop
    crop_height = height
    crop_width = int(crop_height * 9 / 16)

    # Determine the starting row and column of the crop
    start_row = 0
    start_col = int((width - crop_width) / 2)

    # Use numpy indexing to extract the crop
    crop = img[start_row:start_row+crop_height, start_col:start_col+crop_width]
    crop = Image.fromarray(crop)
    return crop, crop_width, crop_height

def combine_both(cropped_background, img2):
    background, crop_width, crop_height = cropped_background
    background = addclock(background)
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
def get_current_time():
    # Get the current time in seconds
    current_time = time.time()
    
    # Convert the time to a time struct
    time_struct = time.localtime(current_time)
    
    # Extract the hours and minutes from the time struct
    hours = time_struct.tm_hour
    minutes = time_struct.tm_min
    
    # Format the time as Hours:Minutes
    time_str = f"{hours:02d}:{minutes:02d}"
    
    return time_str
def get_current_date():
    # Get the current time in seconds
    current_time = time.time()
    
    # Convert the time to a time struct
    time_struct = time.localtime(current_time)
    
    # Extract the day, date, and month from the time struct
    day = time.strftime('%A', time_struct)
    date = time_struct.tm_mday
    month = time.strftime('%B', time_struct)
    
    # Format the date as "Sunday, 12 September"
    date_str = f"{day}, {date} {month}"
    
    return date_str
def addclock(cropped_background):
    # Convert the PIL image to a numpy array
    np_image = np.array(cropped_background)

    # Convert the numpy array to an OpenCV image
    cv2_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = hsv[..., 2] * 0.9
    cv2_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    #########################################################################################
    text = get_current_date()
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 6
    font_thickness = 14
    height_bias = 0.08  # 10% height bias
    shadow_color = (0, 0, 0)
    shadow_offset = 2  # adjust this value to control shadow offset
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    img_height, img_width, _ = cv2_image.shape
    text_x = int((img_width - text_size[0]) / 2)
    text_y = int(img_height * height_bias)  # modified calculation
    cv2.putText(cv2_image, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
    #########################################################################################
    text = get_current_time()
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 20
    font_thickness = 20
    height_bias = 0.22  # 10% height bias
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    img_height, img_width, _ = cv2_image.shape
    text_x = int((img_width - text_size[0]) / 2)
    text_y = int(img_height * height_bias)  # modified calculation
    cv2.putText(cv2_image, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
    #########################################################################################
    img_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    cropped_background = Image.fromarray(img_rgb)
    return cropped_background
def main_loop():
    st.title("iOS 16 Lockscreen Effect")

    foreground_file = st.sidebar.file_uploader("Upload Your Foreground", type=['jpg', 'png', 'jpeg'])
    if not foreground_file:
        return None
    input_image = Image.open(foreground_file)
    cropped_background = crop_background(input_image)
    
    with st.spinner('Wait for it...'):
        cropped_background_with_clock = cropped_background[0]
        input_image_numpy = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(cropped_background[0]))
        segmentation_mask = get_image_segments(input_image_numpy)
        input_seg_and_contour = apply_segmentation(cropped_background_with_clock, segmentation_mask)
        thumbnail = combine_both(cropped_background, input_seg_and_contour)
        # thumbnail = Image.fromarray(input_seg_and_contour)
        
        
        
        # Convert the PIL image to bytes
        img_bytes = io.BytesIO()
        thumbnail.save(img_bytes, format="PNG")
        img_bytes = img_bytes.getvalue()

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