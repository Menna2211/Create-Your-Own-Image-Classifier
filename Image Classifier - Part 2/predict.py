# Import necessary libraries and modules
import argparse
from PIL import Image
from load_models import load_model
from Image_Preprocessing import process_image 
import json
import sys 
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np

# Define command-line arguments
parser = argparse.ArgumentParser(description='Predicting an image-flower name from an image')
parser.add_argument("--image_path", default="flowers/test/1/image_06743.jpg", help="File path of the image to classify")
parser.add_argument("--top_k", type=int, help="Top matched classes to be returned", default=5)
parser.add_argument("--category_names", help="Path linking categories to real names", default="cat_to_name.json")
parser.add_argument('--gpu', default="gpu", help='Use GPU for training')
parser.add_argument('--checkpoint_path', default="./checkpoint.pth", help='Directory to checkpoints')

# Parse command-line arguments
results = parser.parse_args()
image_path = results.image_path
checkpoint_path = results.checkpoint_path
top_k = results.top_k
category_names = results.category_names
gpu = results.gpu

# Use GPU if available
if torch.cuda.is_available() and gpu == 'gpu':
    device = torch.device("cuda")
    print("CUDA is available")
else:
    device = torch.device("cpu")
    print("CUDA is not available")

# Load the model and move it to the specified device
model = load_model(checkpoint_path)
model.to(device)

# Get the class-to-index mapping from the model
class_to_idx = model.class_to_idx 

def predict(image_path, model, top_k=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Ensure the model is in evaluation mode and using GPU
    model.eval()
    model.to('cuda')

    # Process the image
    img = process_image(image_path)
    
    # Convert the processed image to a PyTorch tensor
    img = torch.from_numpy(np.array([img])).float().to('cuda')

    with torch.no_grad():
        # Forward pass to get log probabilities
        logps = model.forward(img)
        
    # Calculate probabilities
    probabilities = torch.exp(logps)

    # Get the top probabilities and corresponding classes
    top_probs, top_indices = probabilities.topk(top_k)
    
    # Convert indices to class labels
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx.item()] for idx in top_indices[0]]

    # Convert tensors to numpy arrays for easier use
    top_probs = top_probs.cpu().numpy()
    top_classes = np.array(top_classes)

    return top_probs, top_classes

# Perform prediction
probs, classes = predict(image_path, model, top_k=5)

# Display results based on category names or class indices
if not category_names:
    print("Class names for Top 5 are ", class_names)
    print("Top 5 Probabilities are", probs)
else:
    name_classes = []
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)    
    print("The original image is ", cat_to_name[image_path.split('/')[-2]])
    # Map class indices to class names
    class_names = [cat_to_name[str(cls )] for cls in classes]
    print("Class for Top 5 are ", classes)
    print("Class names for Top 5 are ", class_names)
    print("Top 5 Probabilities are", probs)
