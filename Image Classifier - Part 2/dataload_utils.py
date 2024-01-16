# Imports for PyTorch and torchvision
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models

# Function for loading data
def data_loading(root='./flowers'):
    """
    Load and preprocess data for training, validation, and testing.

    Args:
        root (str): Root directory containing the 'train', 'valid', and 'test' subdirectories.

    Returns:
        trainloader, validationloader, testloader: DataLoaders for training, validation, and testing sets.
        train_data: Training dataset.
    """
    data_dir = root
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define transforms for the training, validation, and testing sets
    # Transform for the training set
    train_trans = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Transforms for the validation and testing sets
    validation_test_trans = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_trans)
    test_data = datasets.ImageFolder(test_dir, transform=validation_test_trans)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_test_trans)

    # Define dataloaders using the image datasets and transforms
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=64, shuffle=True)

    return trainloader, validationloader, testloader, train_data