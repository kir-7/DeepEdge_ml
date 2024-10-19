from flask import Blueprint, request, jsonify
from model_pipeline.pipeline import Pipeline
from model_pipeline.utils import encode_images, decode_image

import random

api = Blueprint('api', __name__)

# Initialize models

pipeline = Pipeline()

@api.route('/generate', methods=['POST'])
def generate():

    data = request.json
    prompt = data.get('prompt')

    encoded_image = pipeline.generate(prompt)

    response = {
        "request_id": random.randint(10000, 100000000),  # need to create a db that can store each request to keep track
        "generated_image": encoded_image,
    }

    return jsonify(response)

@api.route('/analyze', methods=['POST'])
def analyze():
    data = request.json

    image = data.get('image')  # Base64 encoded image
    texts = data.get('texts')  # texts for clip analysis
    roi = data.get("roi")      # get roi for segment analysis

    # Decode the image
    decoded_image = decode_image(image)
    
    confidence_scores = pipeline.analyze_clip(decoded_image, texts)   # returns the confidence scores
    masks, iou_scores, polygons = pipeline.analyze_sam(decoded_image, roi)
    
    response = {
        "request_id": random.randint(10000, 1000000),  # again keep a db to track these reqests
        "image":image,
        "clip_analysis": {
            "concepts":texts,
            "confidence_scores":confidence_scores
        },
        "basic_segmentation": {
            "masks": masks,
            "iou_scores":iou_scores,
            "polygons": polygons
        }
    }

    return jsonify(response)
