# Imports for PyTorch, torchvision, and other libraries
import argparse
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
import numpy as np
import json
from dataload_utils import data_loading
from build_model import build_models

# Function to train and test the image-flower classification model
def main():
    # Command-line arguments parsing
    parser = argparse.ArgumentParser(description='Train an image-flower classification model')
    # Define command-line arguments
    parser.add_argument('--data_dir', default="./flowers", type=str, help='Path to the dataset directory')
    parser.add_argument('--save_dir', default="./checkpoint.pth", type=str, help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg19', choices=['vgg19', 'vgg16' ,'densenet161' , 'alexnet'], help='Architecture ("vgg16", "vgg19" ,"alexnet" , "densenet161")')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=2048, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--gpu', default="gpu", help='Use GPU for training')
    parser.add_argument('--dropout', type=float, default=0.5 , help='dropout')
    
    # Parse the command-line arguments
    args = parser.parse_args()
    gpu = args.gpu
    epochs = args.epochs
    
    # Load the data
    trainloader, validationloader, testloader, train_data = data_loading(args.data_dir)
    # Build the model
    model = build_models(args.arch, args.dropout ,args.hidden_units)
    # Define the loss function
    criterion = nn.NLLLoss()
    # Define the optimizer
    optimizer = optim.Adam(model.classifier.parameters(), args.learning_rate)
   
    # Use GPU if it's available
    if torch.cuda.is_available() and gpu == 'gpu':
        device = torch.device("cuda")
        print("cuda is available")
    else:
        device = torch.device("cpu")
        print("cuda is not available")
    # Move the model to the specified device
    model.to(device)
    print("model is ", model)
    
    # Train the model
    print_every = 10
    step = 0
    train_loss = 0
    print("--Training is starting--")
    for e in range(epochs):
        for images, labels in trainloader:
            # Move images and labels to the device
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.shape[0], 3, 224, 224)
            step += 1
            # Forward pass
            log_ps = model(images)
            # Calculate the loss
            loss = criterion(log_ps, labels)
            # Clear the gradients
            optimizer.zero_grad()
            # Backward pass
            loss.backward()
            # Update the weights
            optimizer.step()
            train_loss += loss.item()
            if step % print_every == 0:
                # Turn model to eval mode
                model.eval()
                valid_loss = 0
                accuracy = 0
                # Turn on no_grad
                with torch.no_grad():
                    for images, labels in validationloader:
                        # Move images and labels to the device
                        images, labels = images.to(device), labels.to(device)
                        images = images.view(images.shape[0], 3, 224, 224)
                        # Forward pass
                        log_ps = model(images)
                        # Calculate the loss
                        valid_loss += criterion(log_ps, labels)
                        # Calculate validation accuracy
                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()       
                # Print stats
                # Calculate training loss, validation loss, and accuracy  
                avg_train_loss = train_loss / print_every
                avg_valid_loss = valid_loss / len(validationloader)
                val_accuracy = accuracy / len(validationloader) 
                print(f"Epoch {e+1}/{epochs}... "
                      f"Train Loss: {avg_train_loss:.3f}... "
                      f"Validation Loss: {avg_valid_loss:.3f}... "
                      f"Validation Accuracy: {val_accuracy * 100:.3f} %")
                train_loss = 0
                model.train()
                
    # Test the model
    print("--Testing is starting--")
    model.eval()
    test_loss = 0
    test_acc = 0
    # Turn on no_grad
    with torch.no_grad():
        for images, labels in testloader:
            # Move images and labels to the device
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.shape[0], 3, 224, 224)
            # Forward pass
            log_ps = model(images)
            # Calculate the loss
            test_loss += criterion(log_ps, labels)
            # Calculate test accuracy
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_acc += torch.mean(equals.type(torch.FloatTensor)).item()

    # Print stats
    # Calculate testing loss and accuracy  
    avg_test_loss = test_loss / len(testloader)
    test_accuracy = test_acc / len(testloader)
    print("Test Loss: {:.3f}".format(avg_test_loss))
    print("Test Accuracy: {:.3f} %".format(test_accuracy * 100)) 
    
    # Save the checkpoint
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {
        'input_size': 25088,
        'output_size': 102,
        'learning_rate': args.learning_rate,
        'classifier': model.classifier,
        'epochs': epochs,
        'hidden layers': args.hidden_units ,
        'structure' : args.arch ,
        'optimizer': optimizer.state_dict(),
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    torch.save(checkpoint, args.save_dir)
    print("Checkpoint saved")

if __name__ == '__main__':
    main()