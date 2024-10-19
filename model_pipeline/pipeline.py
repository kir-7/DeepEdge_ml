
import torch
from torch import nn
import torchvision

import numpy as np
from PIL import Image

from model_pipeline.models import CLIPAnalyzer, SegmentModel,  StableDiffusion
from model_pipeline.utils import masks_to_polygons, extract_pixels, get_masked_area, encode_images

from config import SAVE_PATH

import base64
from io import BytesIO


'''
The flow:   StableDiffusion.generate(prompt)->image; will take a list of prompts(List[str]) as inputs and returns a list of PIL images 
            for each prompt.

            CLIPAnalyzer.analyze(image, texts)->probs; will take a PIL image as input and a list of texts that try to define the image and 
            analyze this image for each of the text and will return the probablities that correspond to how much does that particular
            text relate to the image higher the probability means that text describe that image better; len(texts) == len(probs) .

            SegmentModel.generate(image, roi)-> masks, iou_scores; will take a PIL image and region of interest which can be a list of 
            points (x, y) or bounding boxes (x1, y1, x2, y2) and generate 3 masks correspomding to the roi and gives iou_scoes for each
            of the mask generated.

'''

'''
Problems: 1. for SegmentModel it requires to return the mask as well as polygon, given a roi we can return the mask but what does 
             Polygon mean? how should I return, how is it different from bounding box

          2. for CLIP model it requires to identify main concepts but CLIP doesn't robustly give the text describing the image it takes
             image texts as input and will give the best texts among the given that relate to that image, how do I identify main concepts    

          3. Need to properly integrate the CPU and GPU environments as it suggests, probably better to create a Environment class that 
             handles all imports and devices setting downloading the models (think !!).
        
          4. How should the pipeline take the roi for sam and texts for clip as inputs will these provided by the user during he api 
             requets or should a frontend be built where the user directly interacts with the model
                           
'''

class Pipeline:
    def __init__(self, diffusion=None, clip=None, segment=None, save_path=SAVE_PATH, device='cpu') -> None:
        
        print("loading pretrained models from huggingface...")

        if not diffusion:
            self.diffusion = StableDiffusion(device=device)
        else:
            self.diffusion = diffusion
        if not clip:
            self.clip = CLIPAnalyzer(device=device)
        else:
            self.clip = clip
        if not segment:
            self.segment = SegmentModel(device=device)
        else:
            self.segment = segment

        print(end='\r')
        print("models loaded.")

        self.device = device
        self.save = save_path

        self.image = None
        self.confidence_scores = None
        self.iou_scores = None
        self.masks = None

    def generate(self, prompt):
        
        self.image = self.diffusion.generate(prompt)
        
        encoded_image = encode_images(self.image)

        self.image = np.array(self.image)

        return encoded_image

    def analyze_clip(self, image, texts):
        
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        assert isinstance(image, np.ndarray) 

        if isinstance(texts, str):
            self.probs = self.clip.analyze(image, [texts])

        if isinstance(texts, list) and len(texts)>0:
            self.probs = self.clip.analyze(image, texts)

        return torch.max(self.probs), self.probs

    def analyze_sam(self, image, roi):

        if isinstance(image, Image.Image):
            image = np.array(image)
        
        assert isinstance(image, np.ndarray) 

        self.masks, self.iou_scores = self.segment.generate(image, roi)

        masked_images = get_masked_area(image, np.array(self.masks))
        encoded_masked_images = encode_images(masked_images)

        return self.mask, self.iou_scores, encoded_masked_images


    def __call__(self, prompt, texts, roi):

        image = self.generate(prompt)

        max_prob, probs = self.analyze_clip(image, texts)

        masks, iou_scores, polygons = self.analyze_sam(image, roi)

        return image, probs, masks, iou_scores, polygons