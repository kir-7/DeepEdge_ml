# Image Generation and Analysis Pipeline

## Overview
This project implements a comprehensive pipeline for generating images from text descriptions using **Stable Diffusion**, analyzing the generated images with **CLIP**, and performing instance segmentation using **Segment Anything Model 2 (SAM2)**. The entire system is accessible through a RESTful API built with Flask, allowing for seamless integration and usage in various applications.

### Project Flow
1. **Image Generation**: A user sends a text prompt to the `/generate` endpoint. The system uses Stable Diffusion to create an image based on the prompt.
2. **Image Analysis**: Once the image is generated, users can send it to the `/analyze` endpoint for analysis. The system performs a CLIP analysis to identify concepts in the image and uses SAM2 for instance segmentation.
3. **Responses**: The API returns the generated image and analysis results in a structured JSON format.

## API Documentation

### Base URL
http://localhost:5001/api


### Endpoints

#### 1. Generate Image
- **Endpoint**: `/generate`
- **Method**: `POST`
- **Request Body**:
    ```json
    {
        "prompt": "A beautiful sunset over a mountain range"
    }
    ```
- **Response**:
    ```json
    {
        "request_id": "unique_id",
        "generated_image": "base64_encoded_image"
    }
    ```

#### 2. Analyze Image
- **Endpoint**: `/analyze`
- **Method**: `POST`
- **Request Body**:
    ```json
    {
        "image": "base64_encoded_image",
        "texts": ["sunset", "mountain"],
        "roi": [[x1, y1, x2, y2]]  // Optional: List of regions of interest for segmentation
    }
    ```
- **Response**:
    ```json
    {
        "request_id": "unique_id",
        "image": "base64_encoded_image",
        "clip_analysis": {
            "concepts": ["sunset", "mountain"],
            "confidence_scores": [0.95, 0.85]
        },
        "basic_segmentation": {
            "masks": ["mask_data"],
            "iou_scores": [0.75],
            "polygons": ["polygon_data"]
        }
    }
    ```

## Environment Setup

### Prerequisites

- Python 3.8 or higher
- Pip (Python package installer)

### Clone the Repository

```bash
git clone https://github.com/kir-7/DeepEdge_ml.git
cd my_pipeline
```
## Install Dependencies

* if installing via pip venv
```python
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```
 * if install via conda, first check if conda is available
 ```python
 conda activate [env_name]  # ensure you have crated a conda envirinment
 pip install -r requirements.txt
 ```

 ## GPU Setup
To utilize a GPU for model inference:

* Ensure you have NVIDIA GPU drivers installed.
* Install the CUDA toolkit and cuDNN library compatible with your PyTorch version. Refer to PyTorch installation   guide for details.
* The models will automatically utilize the GPU if available thorugh torch.cuda.is_available()

## CPU Setup

For running the models on CPU:

*Simply ensure that you have installed PyTorch without GPU support, or configure your environment to run on CPU explicitly by setting the device to 'cpu'. (this will automatically be done the modules through device="cpu")

## Model Configurations

* Stable Diffusion: Uses the CompVis/stable-diffusion-v1-4 model for generating images. You can configure parameters like image dimensions and inference steps in the model_pipeline/models.py file.
* CLIP: Utilizes openai/clip-vit-base-patch32 for analyzing the generated images.
* Segment Anything Model 2 (SAM2): Leverages the facebook/sam-vit-huge model for image segmentation tasks.

## Example Requests

The repository has a file test.py that contains the code for sending requests and recieving requests from the server through the api, if required change the prompts in test.py file, if required you can add additional functionality to test.py file to save the generated images or to view the masking of the images through functions provided in model_pipeline/utils.py

## Running the Application
To run the API server, execute the following command:
```bash
python -m api.app.py
```
the above code will execute the app as a package.
The server will start on http://localhost:5001/api.

## Acknowledgements
* Hugging Face for the pretrained models.
* Flask for building the API.
* PyTorch for model implementation.