import requests
import base64
from PIL import Image
import io


######  A file to test the working of the api

url = 'http://localhost:5001/api/'
## first test out the /generate option

extension = "generate"
data = {
    "prompt": "A fantasy landscape with mountains and a river"
}

response = requests.post(url+extension, json=data)

if response.status_code == 200:
    result = response.json()
    print("Request_id:", result["request_id"])
    print("Generated Image (base64):", result['generated_image'])
else:
    print("Error:", response.status_code, response.text)



## testing the /analyze option

image_base64 = result['generated_image']
print("")
extension = 'analyze'

data = {
    "image": image_base64,
    "texts": ["astronaut", "dog", 'space', 'whie'],
    "roi":[[400, 250]]
}

response = requests.post(url+extension, json=data)

if response.status_code == 200:
    result = response.json()
    print("Request_id:", result['request_id'])
    print("CLIP Analysis:")
    print("     Concepts:", result['clip_analysis']['concepts'])
    print("     Confidence_scores:", result['clip_analysis']['confidence_scores'])
    print("     Most Confident Concept:", result['clip_analysis']['max_confidence_concept'])
    
    print("Basic Segmentation:")
    print("     Masks:", result['basic_segmentation']['masks'])
    print("     IOU Scores:", result['basic_segmentation']['iou_scores'])
    print("     Polygons:", result['basic_segmentation']['polygons'])
else:
    print("Error:", response.status_code, response.text)
