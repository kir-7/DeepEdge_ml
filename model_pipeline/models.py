import torch

from diffusers import StableDiffusionPipeline
from transformers import SamModel, SamProcessor, CLIPModel, CLIPProcessor

import logging

################     STABLE DIFFUSION      ############

# https://huggingface.co/CompVis/stable-diffusion-v1-4
# https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb#scrollTo=G47gEbg9Z4sJ
# refer above notebook for later use

class StableDiffusion:
    def __init__(self, model = None, image_dimensions: tuple = (512, 512), 
                 num_inference_steps: int = 50, seed: bool = False, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> None:
        self.model = model
        self.dimensions = image_dimensions
        self.num_inference_steps = num_inference_steps
        self.device = device

        try:
            if model is None:
                self.pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
            else:
                self.pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16)
            
            self.pipe.to(self.device)

            if seed:
                self.generator = torch.Generator(self.device).manual_seed(1024)
            else:
                self.generator = torch.Generator(self.device)

        except Exception as e:
            logging.error(f"Error initializing StableDiffusion: {str(e)}")
            raise

    def generate(self, prompts, batch_size = 1, path = None) :
        
        if isinstance(prompts,str):
            prompts = [prompts]

        results = []
        try:
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i+batch_size]
                inference = self.pipe(batch_prompts, height=self.dimensions[0], width=self.dimensions[1], 
                                      generator=self.generator, num_inference_steps=self.num_inference_steps)
                
                images = inference.images
                nsfw_content_detected = inference.nsfw_content_detected

                for img, nsfw in zip(images, nsfw_content_detected):
                    if nsfw:
                        logging.warning("NSFW content detected, skipping...")
                        results.append(None)
                    else:
                        results.append(img)
                        if path:
                            img.save(f"{path}/image_{len(results)}.png")

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logging.warning("GPU out of memory, falling back to CPU")
                self.device = 'cpu'
                self.pipe.to(self.device)
                return self.generate(prompts, batch_size, path)
            else:
                logging.error(f"Error during generation: {str(e)}")
                raise

        return results

    def __del__(self):
        # Clean up resources
        if hasattr(self, 'pipe'):
            del self.pipe
        torch.cuda.empty_cache()



##################       CLIP       ################

# https://huggingface.co/docs/transformers/en/model_doc/clip



#  you can include CLIP with flash attention which makes it faster on GPUs so need to properly implement in environmentsimport torch
class CLIPAnalyzer:
    def __init__(self, model=None, processor=None, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        model_name = model or "openai/clip-vit-base-patch32"
        processor_name = processor or "openai/clip-vit-base-patch32"

        try:
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(processor_name)
        except Exception as e:
            logging.error(f"Error initializing CLIP model: {str(e)}")
            raise

    def analyze(self, image, texts):
        try:
            inputs = self.processor(text=texts, images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)

            return probs.detach().cpu().numpy()  # Convert to numpy array for easier handling

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logging.warning("GPU out of memory, falling back to CPU")
                self.device = 'cpu'
                self.model.to(self.device)
                return self.analyze(image, texts)
            else:
                logging.error(f"Error during CLIP analysis: {str(e)}")
                raise

    def __del__(self):
        # Clean up resources
        if hasattr(self, 'model'):
            del self.model
        torch.cuda.empty_cache()


##################   IMAGE SEGMENTATION         ################### 

# https://huggingface.co/docs/transformers/main/en/model_doc/sam
# https://github.com/huggingface/notebooks/blob/main/examples/segment_anything.ipynb
# https://github.com/huggingface/notebooks/blob/main/examples/automatic_mask_generation.ipynb
# use the above notebooks as it contains output segmentation clearing and better masking techniques can be used to properly save masks


class SegmentModel:
    def __init__(self, model_type = None, processor_type = None, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        model_name = model_type or "facebook/sam-vit-huge"
        processor_name = processor_type or "facebook/sam-vit-huge"

        try:
            self.model = SamModel.from_pretrained(model_name).to(self.device)
            self.processor = SamProcessor.from_pretrained(processor_name)
        except Exception as e:
            logging.error(f"Error initializing SAM model: {str(e)}")
            raise

    def generate(self, image_rgb, roi= None):
        try:
            input_points, input_boxes = self._prepare_inputs(roi)
            
            inputs = self._process_inputs(image_rgb, input_points, input_boxes)
            
            with torch.no_grad():
                image_embeddings = self.model.get_image_embeddings(inputs.pop("pixel_values"))
                inputs.update({"image_embeddings": image_embeddings})
                outputs = self.model(**inputs)

            masks = self.processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(), 
                inputs["original_sizes"].cpu(), 
                inputs["reshaped_input_sizes"].cpu()
            )
            
            return masks[0].detach().cpu().numpy(), outputs.iou_scores.detach().cpu().numpy()

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logging.warning("GPU out of memory, falling back to CPU")
                self.device = 'cpu'
                self.model.to(self.device)
                return self.generate(image_rgb, roi)
            else:
                logging.error(f"Error during SAM generation: {str(e)}")
                raise

    def _prepare_inputs(self, roi = None):
        input_points, input_boxes = [], []
        if roi:
            for region in roi:
                if len(region) == 4:
                    input_boxes.append(region)
                else:
                    input_points.append(region)
        return [input_points], [input_boxes]

    def _process_inputs(self, image_rgb, input_points, input_boxes):
        if input_points[0] and input_boxes[0]:
            inputs = self.processor(image_rgb, input_points=input_points, input_boxes=input_boxes, return_tensors="pt")
        elif input_points[0]:
            inputs = self.processor(image_rgb, input_points=input_points, return_tensors="pt")
        elif input_boxes[0]:
            inputs = self.processor(image_rgb, input_boxes=input_boxes, return_tensors="pt")
        else:
            inputs = self.processor(image_rgb, return_tensors="pt")
        
        return {k: v.to(self.device) for k, v in inputs.items()}

    def __del__(self):
        if hasattr(self, 'model'):
            del self.model
        torch.cuda.empty_cache()