# Function to build a custom neural network model
def build_models(structure='vgg19', dropout=0.5, hidden_units=2048):
    """
    Build a custom neural network model with a specified architecture and classifier.

    Args:
        structure (str): Name of the pre-trained model architecture to use.
        dropout (float): Dropout probability for regularization in the custom classifier.
        hidden_units (int): Number of hidden units in the custom classifier.

    Returns:
        model: PyTorch model with the specified architecture and a custom classifier.
    """

    # Import necessary libraries
    import torch
    from torch import nn
    from torchvision import models
    from collections import OrderedDict

    # Choose the pre-trained model based on the provided structure parameter
    if structure == "vgg19":
        model = models.vgg19(pretrained=True)
        in_features_model = model.classifier[0].in_features
    elif structure == "alexnet":
        model = models.alexnet(pretrained=True)
        in_features_model = model.classifier[0].in_features
    elif structure == "vgg16":
        model = models.vgg16(pretrained=True)
        in_features_model = model.classifier[0].in_features
    elif structure == "densenet161":
        model = models.densenet161(pretrained=True)
        in_features_model = model.classifier.in_features
    else:
        # Default to VGG19 if an unsupported structure is provided
        model = models.vgg19(pretrained=True)

    # Freeze the pre-trained model's parameters to avoid training
    for param in model.parameters():
        param.requires_grad = False
        
    
    # Define a new classifier for the model
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(in_features_model, hidden_units)),
        ('relu1', nn.ReLU()),
        ('drop1', nn.Dropout(dropout)),
        ('fc2', nn.Linear(hidden_units, 102)),  # Assuming 102 output classes
        ('output', nn.LogSoftmax(dim=1))
    ]))

    # Replace the pre-trained model's classifier with the new custom classifier
    model.classifier = classifier

    return model