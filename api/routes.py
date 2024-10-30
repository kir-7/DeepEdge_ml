from flask import Blueprint, request, jsonify
from model_pipeline.pipeline import Pipeline
from model_pipeline.utils import encode_images, decode_image

import uuid

api = Blueprint('api', __name__)

# Initialize models

pipeline = Pipeline()

@api.route("/", methods=['GET'])
def home():

    return '''<p>This is the api for a Machine Learning Pipeline! 
                    /generate is a post method to generate images 
                    /analyze is a post method to analyze the generated images.</p>'''


@api.route('/generate', methods=['POST'])
def generate():

    data = request.json
    prompt = data.get('prompt')

    if not prompt or not (isinstance(prompt, str) or isinstance(prompt, list)):
        return jsonify({"error": "Invalid prompt. It should be a non-empty string or non empty list of strings"}), 400

    if pipeline:
        encoded_image = pipeline.generate(prompt)
    else:
        encoded_image = "dummy_encoded_image"

    response = {
        "request_id": str(uuid.uuid4()),                    # need to create a db that can store each request to keep track
        "generated_image": encoded_image,
    }

    return jsonify(response)

@api.route("/adv/generate", methods=["POST"])
def adv_generate():

    data = request.json
    prompt = data.get(prompt)

    if not prompt or not (isinstance(prompt, str) or isinstance(prompt, list)):
        return jsonify({"error": "Invalid prompt. It should be a non-empty string or non empty list of strings"}), 400
    
    if pipeline:
        encoded_image = pipeline.generate(prompt)
    else:
        encoded_image = "Dummy_encoded_image"
    
    if encoded_image == "Dummy_encoded_image":
        return jsonify({"error":"Pipeline has not been initiated on server side."})

    image = decode_image(encoded_image)

    texts = data.get("texts")
    roi = data.get('roi')

    if pipeline:
        max_confidence_text, confidence_scores = pipeline.analyze_clip(image, texts)   # returns the confidence scores
        masks, iou_scores, polygons = pipeline.analyze_sam(image, roi)
    else:
        max_confidence_text, confidence_scores =  None, None
        masks, iou_scores, polygons = None, None, None
    
    if any(True if item == None else False for item in (max_confidence_text, confidence_scores, masks, iou_scores, polygons)):
        return jsonify({"error":"Pipeline has not been properly initialized on server side."})
    
    
    response = {
                "request_id":str(uuid.uuid4()),
                "generated_image":encoded_image,
                "clip_analysis":{
                    "all_concepts":texts,
                    "all_confidence_scores":confidence_scores,
                    "global_confident_concept":max_confidence_text
                },
                "segmentation":{
                    "masks":masks,
                    "iou_scores":iou_scores,
                    "polygons":polygons,
                    "segmented_regions":[{
                        "request_id":str(uuid.uuid4()),
                        "clip_analysis":{i:j for i, j in zip(("concepts", "confidence_scores", "max_confident_concept"), (texts, )+pipeline.analyze_clip(polygon, texts))},
                        "polygon":polygon,
                    } for polygon in polygons]
                }
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
    if pipeline:
        max_confidence_text, confidence_scores = pipeline.analyze_clip(decoded_image, texts)   # returns the confidence scores
        masks, iou_scores, polygons = pipeline.analyze_sam(decoded_image, roi)
    else:
        max_confidence_text, confidence_scores =  None, None
        masks, iou_scores, polygons = None, None, None

    response = {
        "request_id": str(uuid.uuid4()),             # again keep a db to track these reqests
        "image":image,
        "clip_analysis": {
            "concepts":texts,
            "confidence_scores":confidence_scores,
            "max_confidence_concept": max_confidence_text
        },
        "basic_segmentation": {
            "masks": masks,
            "iou_scores":iou_scores,
            "polygons": polygons
        }
    }

    return jsonify(response)