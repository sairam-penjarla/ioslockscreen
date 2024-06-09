# Image Segmentation with MediaPipe

A Python project that performs image segmentation using MediaPipe, and applies different templates to the segmented image.

## Project Description

This project uses MediaPipe to perform image segmentation and then applies different templates to the segmented image. The application crops the background to a specified aspect ratio, segments the image, and pastes the foreground on the templates.

## Features

- Image segmentation using MediaPipe.
- Template application on segmented images.
- Background cropping to specified aspect ratio.

## Getting Started

### Prerequisites

- Python 3.x
- OpenCV
- NumPy
- Pillow
- MediaPipe
- PyYAML

### Installation

1. Clone the repository or download the source code.
    ```
    git clone https://github.com/sairam-penjarla/ioslockscreen.git
    ```
    
2. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

### Usage

1. Ensure your input image is placed in the specified path.
2. Run the script:

    ```sh
    python image_segmentation_app.py
    ```

### Configuration

You can customize the input paths and other parameters by modifying the `config.yaml` file.

### Example `config.yaml`

```yaml
input_path: "../input/b.jpg"
output_path: "../output"
white_template_path: "../assets/white.png"
black_template_path: "../assets/black.png"
model_path: "../model/deeplab_v3.tflite"
```