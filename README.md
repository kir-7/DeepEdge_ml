# Image Generation and Analysis Pipeline

## Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [API Reference](#api-reference)
- [Model Details](#model-details)
- [Performance Considerations](#performance-considerations)
- [Examples](#examples)
- [Acknowledgements](#acknowledgements)

## Overview
This project implements a comprehensive pipeline for generating images from text descriptions using **Stable Diffusion**, analyzing the generated images with **CLIP**, and performing instance segmentation using **Segment Anything Model (SAM)**. The entire system is accessible through a RESTful API built with Flask, allowing for seamless integration and usage in various applications.

## System Architecture

### Components

```
project/
├── api/
│   ├── app.py          # Main Flask application
│   └── routes.py       # API endpoint definitions
├── model_pipeline/
│   ├── pipeline.py     # Core pipeline implementation
│   ├── models.py       # Model configurations
│   └── utils.py        # Utility functions
|
├── requirements.txt    # Project dependencies
|── README.md           # Project documentation
└── test.py             # a file to send example requests to the api
```

### Project Flow
1. **Image Generation**: A user sends a text prompt to the `/generate` endpoint. The system uses Stable Diffusion to create an image based on the prompt.
2. **Image Analysis**: Once the image is generated, users can send it to the `/analyze` endpoint for analysis. The system performs a CLIP analysis to identify concepts in the image and uses SAM2 for instance segmentation.
3. **Responses**: The API returns the generated image and analysis results in a structured JSON format.

## Installation

### Prerequisites

- Python 3.8 or higher
- Pip (Python package installer)
- conda (for environment management) [download_from](https://anaconda.org/)
- NVIDIA GPU with 8GB+ VRAM (recommended) or CPU with 16GB+ RAM

### Environment Setup
1. **Clone Repository**
   ```bash
   git clone https://github.com/kir-7/DeepEdge_ml.git
    cd my_pipeline
   ```

2. **Create Environment**

```bash
   # Using conda
   conda create -n img-pipeline python=3.8
   conda activate img-pipeline

   # OR using venv
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```


 ## GPU Setup
To utilize a GPU for model inference:

* Ensure you have NVIDIA GPU drivers installed.
* You need cuda and cuDNN for installing pytorch with gpu support.
* You can get your cuda version through the command
```bash
nvidia-smi
```
* Using the cuda version you can install the cuda toolkit through anaconda, follow this [link](https://medium.com/@leennewlife/how-to-setup-pytorch-with-cuda-in-windows-11-635dfa56724b) for instructions to install cuda toolkit and compatible pytorch gpu support 

* Once the gpu support is installed The models will automatically utilize the GPU if available thorugh torch.cuda.is_available()
* Note that since heavy models are used you would need minimum of 8GB of GPU and atleast a Tesla T4 GPU is recommended  

## CPU Setup

For running the models on CPU:

* Simply ensure that you have installed PyTorch without GPU support, or configure your environment to run on CPU explicitly by setting the device to 'cpu'. (this will automatically be done the modules through device="cpu")

* Running on CPU is support but it is not recommended as the models are large it will take extremely long for a single inference 
* Clonning the repository and installing the requirements will automatically install the pytorch version that supports cpu, for pytorch to support gpu look at gpu setup, and install the compatible version.   

## API Reference

### Base URL
http://localhost:5001/api


### Endpoints

#### 1. Generate Image (`POST /generate`)
Generates image(s) from text prompt(s).

**Request Body:**
```json
{
    "prompt": "a serene lake at sunset"
    // OR
    "prompt": ["sunset lake", "morning lake"]
}
```

**Response:**
```json
{
    "request_id": "uuid-string",
    "generated_image": "base64_encoded_image"
}
```

#### 2. Advanced Generate (`POST /adv/generate`)
Generates image and performs immediate analysis.

**Request Body:**
```json
{
    "prompt": "text prompt",
    "texts": ["concept1", "concept2"],
    "roi": [[x1, y1, x2, y2]]
}
```

**Response:**
```json
{
    "request_id": "uuid-string",
    "generated_image": "base64_encoded_image",
    "clip_analysis": {
        "all_concepts": ["concept1", "concept2"],
        "all_confidence_scores": [0.95, 0.85],
        "global_confident_concept": "concept1"
    },
    "segmentation": {
        "masks": ["base64_encoded_mask"],
        "iou_scores": [0.75],
        "polygons": ["polygon_data"],
        "segmented_regions": [
            {
                "request_id": "uuid-string",
                "clip_analysis": {
                    "concepts": ["concept1", "concept2"],
                    "confidence_scores": [0.9, 0.8],
                    "max_confident_concept": "concept1"
                },
                "polygon": "polygon_data"
            }
        ]
    }
}
```

#### 3. Analyze Image (`POST /analyze`)
Analyzes existing image using CLIP and SAM2.

**Request Body:**
```json
{
    "image": "base64_encoded_image",
    "texts": ["concept1", "concept2"],
    "roi": [[x1, y1, x2, y2]]
}
```

**Response:**
```json
{
    "request_id": "uuid-string",
    "image": "base64_encoded_image",
    "clip_analysis": {
        "concepts": ["concept1", "concept2"],
        "confidence_scores": [0.95, 0.85],
        "max_confidence_concept": "concept1"
    },
    "basic_segmentation": {
        "masks": ["mask_data"],
        "iou_scores": [0.75],
        "polygons": ["polygon_data"]
    }
}
```


## Model Details

### Stable Diffusion
- **Version**: CompVis/stable-diffusion-v1-4
- **Purpose**: Text-to-image generation
- **Configuration**: Configurable image dimensions and inference steps

### CLIP
- **Version**: openai/clip-vit-base-patch32
- **Purpose**: Image-text similarity analysis
- **Features**: Zero-shot classification capabilities

### SAM2
- **Version**: facebook/sam-vit-huge
- **Purpose**: Instance segmentation
- **Features**: Point and box prompting supported

## Examples

The repository has a test.py file that provides basic example code for sending requests and recieving requests from the server through the api, if required change the prompts in test.py file, if required you can add additional functionality to test.py file to save the generated images or to view the masking of the images through functions provided in model_pipeline/utils.py

### Running the Application and Sending Requests (Locally)
To run the API server, execute the following command:
```bash
python -m api.app.py
```
the above code will execute the app as a package.
The server will start on http://localhost:5001/api.

<br>

To test the api requets, open another command prompt/terminal and execute the following command:
```python
python test.py
```
You will see the response in the cmd. You can alter test.py tp save results

## Acknowledgements
* Hugging Face for the pretrained models.
* Flask for building the API.
* PyTorch for model implementation.