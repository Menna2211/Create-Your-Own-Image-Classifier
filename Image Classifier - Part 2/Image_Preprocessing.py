# Imports here
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import json

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    # Resize the image while maintaining aspect ratio
    image = Image.open(image_path)
    size = 256, 256
    image.thumbnail(size)
    # Crop out the center 224x224 portion of the image
    left_margin = (image.width - 224) / 2
    bottom_margin = (image.height - 224) / 2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    image = image.crop((left_margin, bottom_margin, right_margin, top_margin))
    
    # Convert color channels to floats in the range [0, 1]
    np_image = np.array(image) / 255.0
    # Normalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Reorder dimensions and add batch dimension
    np_image = np_image.transpose((2, 0, 1))
    return np_image
