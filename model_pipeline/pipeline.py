
import torch
from torch import nn
import torchvision

import numpy as np


from model_pipeline.models import CLIPAnalyzer, SegmentModel,  StableDiffusion
from model_pipeline.utils import masks_to_polygons, extract_pixels


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
                           
'''

class Pipleline:
    def __init__(self, difusion, clip, segment, device) -> None:
        pass

    def generate(self):
        pass

    def analyze(self):
        pass

    def load(self):
        pass

    def define(self):
        pass