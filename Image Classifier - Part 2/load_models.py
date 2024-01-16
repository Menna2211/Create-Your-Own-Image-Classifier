import torch
from torch import nn, optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models

def load_model(filepath):
    # Load checkpoint from the saved file
    checkpoint = torch.load(filepath)
    
    # Get the model architecture specified in the checkpoint
    structure = checkpoint['structure']
    
    # Choose the pre-trained model based on the specified architecture
    if structure == "vgg19":
        model = models.vgg19(pretrained=True)
    elif structure == "alexnet": 
        model = models.alexnet(pretrained=True)
    elif structure == "vgg16":
        model = models.vgg16(pretrained=True) 
    elif structure == "densenet161":
        model = models.densenet161(pretrained=True) 
    else:
        # Use VGG19 as the default model if the architecture is not recognized
        model = models.vgg19(pretrained=True)
    
    # Freeze the pre-trained model's parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Load the classifier, class_to_idx mapping, and model state_dict from the checkpoint
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model
