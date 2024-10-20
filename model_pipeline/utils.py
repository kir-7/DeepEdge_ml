import matplotlib.pyplot as plt
import numpy as np
import cv2

from io import BytesIO
from PIL import Image

import base64

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

def show_boxes_on_image(raw_image, boxes):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()

def show_points_on_image(raw_image, input_points, input_labels=None):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    plt.axis('on')
    plt.show()

def show_points_and_boxes_on_image(raw_image, boxes, input_points, input_labels=None):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()


def show_points_and_boxes_on_image(raw_image, boxes, input_points, input_labels=None):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_masks_on_image(raw_image, masks, scores, return_pil=False):
    if len(masks.shape) == 4:
      masks = masks.squeeze()
    if scores.shape[0] == 1:
      scores = scores.squeeze()

    nb_predictions = scores.shape[-1]
    fig, axes = plt.subplots(1, nb_predictions, figsize=(15, 15))

    for i, (mask, score) in enumerate(zip(masks, scores)):
      mask = mask.cpu().detach()
      axes[i].imshow(np.array(raw_image))
      show_mask(mask, axes[i])
      axes[i].title.set_text(f"Mask {i+1}, Score: {score.item():.3f}")
      axes[i].axis("off")
    
    if return_pil:
        img_buf = BytesIO()
        plt.savefig(img_buf, format='png')
        im = Image.open(img_buf)
        return im
    
    plt.show()
  

def get_masked_area(image, masks):
    if len(masks.shape) == 4:
        masks = masks.squeeze()
    if isinstance(image, np.ndarray):
      image = Image.fromarray(image)

    images = []

    for mask in masks:
      binary_mask = Image.fromarray((mask * 255).astype(np.uint8))  # Convert boolean to binary image
      result = Image.composite(image, Image.new('RGB', image.size), binary_mask)

      images.append(result)
    
    return images  

'''
def masks_to_polygons(masks, scores, threshold=0.5):

    if len(masks.shape) == 4:
      masks = masks.squeeze()
    if scores.shape[0] == 1:
      scores = scores.squeeze()

    nb_predictions = scores.shape[-1]

    polygons = []
    for mask, iou in zip(masks, scores):
        print(mask.shape, iou)
        # Convert mask to binary image
        binary_mask = (mask.numpy() > 0).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Convert contour to polygon
            polygon = Polygon(contour[:, 0, :])
            
            # Filter based on IoU score
            if iou > threshold:
                polygons.append(polygon)

    return polygons

    '''

def extract_pixels(image_rgb, masks):
    if len(masks.shape) == 4:
      masks = masks.squeeze()

    extracted_pixels = []
    for mask in masks:
        # Convert mask to binary (boolean) format
        binary_mask = (mask.numpy() > 0).astype(bool)
        # Extract the pixels from image_rgb where mask is True
        masked_pixels = image_rgb[binary_mask]

        extracted_pixels.append(masked_pixels)

    return extracted_pixels


def encode_images(images):

  
  def encode(image):
    if isinstance(image, Image):
      buffered = BytesIO()
      image.save(buffered, format="JPEG")
      base_64_encoded_image = base64.b64encode(buffered.getvalue())
      return base_64_encoded_image
    
    else:
      raise Exception("inputto encode_image should a PIL.Image object")

  encoded_images = []

  if isinstance(images, list):
      for i in images:
        encoded_images.append(encode(i))
    
      return encoded_images
  
  elif isinstance(images, Image):
     return encode(images)
  

def decode_image(encoded_image):

  try:
      image = base64.b64decode(encoded_image, validate=True)
      image = BytesIO(image)
      image = Image,open(image)
      return image
  except Exception as e:
    print(e)
