import torch

from diffusers import StableDiffusionPipeline
from transformers import SamModel, SamProcessor, CLIPModel, CLIPProcessor

import numpy as np

################     STABLE DIFFUSION      ############

# https://huggingface.co/CompVis/stable-diffusion-v1-4
# https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb#scrollTo=G47gEbg9Z4sJ
# refer above notebook for later use

class StableDiffusion:    
    def __init__(self, model=None, image_dimensions=(512, 512), num_inference_steps=50, seed=False, device='cpu') -> None:
    
        self.model = model
        self.dimensions = image_dimensions
        self.num_inference_steps=num_inference_steps
        self.device = device

        if model == None:
            self.pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)  
        else:
            self.pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16)
        
        self.pipe.to(device)

        if seed:
            self.generator = torch.Generator(device).manual_seed(1024)
        else:
            self.generator = torch.Generator(device)

    def generate(self, prompt, path=None):
        
        #  we might have more than one prompt which will generate a list of images for each prompt
        
        inference = self.pipe(prompt, height=self.dimensions[0], width=self.dimensions[1], generator=self.generator, num_inference_steps=self.num_inference_steps)
        images = inference.images
        nsfw_content_detected = any(inference.nsfw_content_detected)
        if nsfw_content_detected:
            print("NSFW content detected returing ...")
            return None
        if path:
            for i, img in enumerate(images):
                img.save(f"{path}/image_{i}.png")
        return images


##################       CLIP       ################

# https://huggingface.co/docs/transformers/en/model_doc/clip



#  you can include CLIP with flash attention which makes it faster on GPUs so need to properly implement in environments
class CLIPAnalyzer:
    def __init__(self, model=None, processor=None, device="cpu"):

        if not model:
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        else:
            self.model = CLIPModel.from_pretrained(model).to(device)
        if not processor:
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        else:
            self.processor = CLIPProcessor.from_pretrained(processor)
        
        self.device = device
    
    def analyze(self, image, texts):
        

        inputs = self.processor(text=texts, images=image, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        return probs.detach().cpu()


##################   IMAGE SEGMENTATION         ################### 

# https://huggingface.co/docs/transformers/main/en/model_doc/sam
# https://github.com/huggingface/notebooks/blob/main/examples/segment_anything.ipynb
# https://github.com/huggingface/notebooks/blob/main/examples/automatic_mask_generation.ipynb
# use the above notebooks as it contains output segmentation clearing and better masking techniques can be used to properly save masks

class SegmentModel:
    def __init__(self, model_type=None, processor_type=None, device='cpu'):

        self.device = device

        if model_type == None:
            self.model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
        else:
            self.model = SamModel.from_pretrained(model_type).to(device)

        if processor_type == None:
            self.processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        else:
            self.processor = SamProcessor.from_pretrained(processor_type)



    def generate(self, image_rgb, roi=None):

        input_points, input_boxes = [], []
        if roi != None :
            for region in roi:
                if len(region) == 4:
                    input_boxes.append(region)
                else:
                    input_points.append(region)
        
        input_points, input_boxes = [input_points], [input_boxes]
                    
        if input_points[0] and input_boxes[0]:    
            inputs = self.processor(image_rgb, input_points=[input_points], input_boxes=[input_boxes], return_tensors="pt").to(self.device)
        elif input_points[0] and not input_boxes[0]:
            inputs = self.processor(image_rgb, input_points=input_points, return_tensors="pt").to(self.device)
        elif input_boxes[0] and not input_points[0]:
            inputs = self.processor(image_rgb, input_boxes=[input_boxes], return_tensors="pt").to(self.device)
        elif not input_points[0] and not input_boxes[0]:
            inputs = self.processor(image_rgb, return_tensors="pt").to(self.device)
        

        image_embeddings = self.model.get_image_embeddings(inputs["pixel_values"])

        inputs.pop("pixel_values", None)
        inputs.update({"image_embeddings": image_embeddings})
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
        
        return masks[0].detach().cpu(), outputs.iou_scores.detach().cpu()
