import requests
import base64
from PIL import Image
import io


######  A file to test the working of the api


url = 'http://localhost:5000/api/generate'
data = {
    "prompt": "A fantasy landscape with mountains and a river"
}

response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()
    print("Generated Image (base64):", result['generated_image'])
    print("CLIP Analysis:", result['clip_analysis'])
    print("Basic Segmentation:", result['basic_segmentation'])
else:
    print("Error:", response.status_code, response.text)




image_base64 = result['generate_image']

url = 'http://localhost:5000/api/analyze'
data = {
    "image": image_base64,
    "texts": ["astronaut", "dog", 'space', 'whie'],
    "roi":[[400, 250]]
}

response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()
    print("CLIP Analysis:", result['clip_analysis'])
    print("Basic Segmentation:", result['basic_segmentation'])
else:
    print("Error:", response.status_code, response.text)
