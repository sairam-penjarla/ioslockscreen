import os
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import yaml
from utils import get_model, crop_background, get_image_segments, apply_segmentation, resize_template

class ImageSegmentationApp:
    def __init__(self, config):
        """
        Initialize the ImageSegmentationApp with configuration.

        Args:
            config (dict): Configuration dictionary.
        """
        self.input_path = config['input_path']
        self.output_path = config['output_path']
        self.white_template_path = config['white_template_path']
        self.black_template_path = config['black_template_path']
        self.model_path = config['model_path']
        self.model = get_model(self.model_path)

    def run(self):
        """
        Main method to run the image segmentation and template application process.
        """
        # Read Background
        input_image = Image.open(self.input_path)
        background, new_width, new_height = crop_background(input_image)

        # Extract Foreground
        input_image_numpy = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(background))
        segmentation_mask = get_image_segments(self.model, input_image_numpy)
        input_seg_and_contour = apply_segmentation(background, segmentation_mask)
        img2_resized = cv2.resize(input_seg_and_contour, (new_width, new_height))
        mask = Image.fromarray((img2_resized != [0, 0, 0]).any(axis=2))
        foreground = Image.fromarray(img2_resized)
        foreground.putalpha(mask)
        x = int((background.width - foreground.width) / 2)
        y = int((background.height - foreground.height) / 2)

        # Prepare Output Path
        output_files = [x for x in os.listdir(self.output_path) if x.endswith(".jpg")]
        files_len = int(sorted(output_files)[-1].split("_")[0]) if len(output_files) > 0 else 0

        # Apply and Save Templates
        self.apply_and_save_template(background, foreground, self.white_template_path, "WHITE", files_len, x, y)
        self.apply_and_save_template(background, foreground, self.black_template_path, "BLACK", files_len, x, y)

    def apply_and_save_template(self, background, foreground, template_path, color, files_len, x, y):
        """
        Apply the template to the background and save the output image.

        Args:
            background (Image): The background image.
            foreground (Image): The foreground image.
            template_path (str): Path to the template image.
            color (str): Color label for the output file.
            files_len (int): Current number of output files.
            x (int): x-coordinate for pasting the foreground.
            y (int): y-coordinate for pasting the foreground.
        """
        template = Image.open(template_path)
        template = resize_template(template, background.height)

        bg_img = background.copy()
        bg_img.paste(template, (0, 0), template)
        bg_img.paste(foreground, (x, y), foreground)

        file_name = os.path.join(self.output_path, f"{files_len + 1}_{color}.jpg")
        bg_img.save(file_name)

if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    app = ImageSegmentationApp(config)
    app.run()
