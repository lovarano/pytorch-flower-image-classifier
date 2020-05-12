# PROGRAMMER: Lorenzo Varano
# DATE CREATED: 2020.04.10
# PURPOSE: This part of the program contains functions for build, train and save the model.
# NOTE: This file was created mainly with support of the previous task in Part 1 and 1st project - use of a pretrained classifier. Additional support material is referenced above the respective code.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
import numpy as np
import pandas as pd
from workspace_utils import active_session
from PIL import Image
import os, random

import processing_images

# Verify device if GPU or CPU is available
def check_gpu(arg_device, device):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(arg_device) == "cuda" and str(device) == 'cpu':
        print("You chose {}, but system is running on {}."
        "Choose cpu or activate gpu before.".format(arg_device, device))
        exit()    
    elif str(arg_device) == "cpu" and str(device) == "cuda":
        print("You chose {}, but {} is available.".format(arg_device, device))
        reply = str(input("Do you want to switch to GPU (cuda). (y/n): ")).lower().strip()
        if reply[0] == 'y':
            return device
        if reply[0] == 'n':
            return arg_device
        else:
            print("Please enter y or n to proceed")
            check_gpu(arg_device, device)
    else:
        print("You chose {} and your system is running on {}."
        "Everything is fine!".format(arg_device, device))
        return arg_device

    
# Load the model as per arch argument
def load_pretrained_model(arch):
    if arch == 'vgg':
        input_size = 25088
        model = models.vgg16(pretrained=True)
    elif arch == 'alexnet':
        input_size = 9216
        model = models.alexnet(pretrained=True)
    else:
        print("Sorry model architecture note recognized. The terminal will be terminated.")
        exit()
        
    return input_size, model
      

# Train the classifier
def train_classifier(model, optimizer, criterion, trainloader, testloader, device, arg_epochs):

   with active_session():
        
        epochs = arg_epochs
        steps = 0
        print_every = 40
        
        model.to(device);
        
        for epoch in range(epochs):

            model.train()
            
            running_loss = 0
            
            for images, labels in iter(trainloader):
                
                steps +=1
                
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                output = model.forward(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if steps % print_every == 0:

                    model.eval()
                    
                    #Turn off gradients for validation
                    with torch.no_grad():
                        test_loss = 0
                        accuracy = 0
                        
                        for images, labels in iter(testloader):
        
                            images, labels = images.to(device), labels.to(device)
                            output = model.forward(images)
                            batch_loss = criterion(output, labels)
                            test_loss += batch_loss.item()
                
                            #Calculate accuracy
                            ps = torch.exp(output)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                    print("Epoch {}/{}".format(epoch+1, epochs),
                          "Train loss: {:.3f}".format(running_loss/print_every),
                          "Test loss: {:.3f} ".format(test_loss/len(testloader)),
                          "Test accuracy: {:.3f}".format(accuracy/len(testloader)))
                    
                    running_loss = 0
                    model.train()
                    
# Function for measuring network accuracy on test data
def test_accuracy(model, testloader, device):

    # Do validation on the test set
    model.eval()
    model.to(device)

    with torch.no_grad():
        
        accuracy = 0
        
        for images, labels in iter(testloader):
            
            images, labels = images.to(device), labels.to(device)
            output = model.forward(images)
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
        print("Test Accuracy: {}".format(accuracy/len(testloader)))  
        
        
# Function for saving the model checkpoint
def save_checkpoint(model, train_dataset, epochs, save_dir, hidden_layers, learning_rate, arch):

    model.class_to_idx = train_dataset.class_to_idx

    checkpoint = {'arch': arch,
                  'hidden_layer_units': hidden_layers,
                  'learning_rate': learning_rate,
                  'model_state_dict': model.state_dict(),
                  'epochs': epochs,
                  'class_to_idx': model.class_to_idx,}

    torch.save(checkpoint, 'checkpoint.pth')
    
# Function for loading the model checkpoint    
def load_checkpoint(filepath):
    
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    
    
    checkpoint = torch.load(filepath, map_location=map_location)
    
    if checkpoint['arch'] == 'vgg':
        input_size = 25088
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch'] == 'alexnet':
        input_size = 9216
        model = models.alexnet(pretrained=True)
    else:
        print("Sorry model architecture note recognized. The terminal will be terminated.")
        exit()
        
    for param in model.parameters():
        param.requires_grad = False
        print(model)
    
    model.class_to_idx = checkpoint['class_to_idx']
    
    checkpoint['hidden_layers'] = 1024
    # Create the classifier
    model.classifier = nn.Sequential(nn.Linear(input_size,checkpoint['hidden_layers']),
                                 nn.ReLU(),
                                 nn.Dropout(0.4),
                                 nn.Linear(checkpoint['hidden_layers'], 102),
                                 nn.LogSoftmax(dim=1))
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def predict(image_path, model, topk, device, cat_to_name):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    model.to(device)
    
    # Process a single image
    image = processing_images.process_image(image_path)
    
    #Turn image into a torch tensor
    if str(device) == "cuda":
        image = torch.from_numpy(image).type(torch.cuda.FloatTensor)
    else:
        image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    
    #Pass through model and reverse log function
    output = model.forward(image)
    ps = torch.exp(output)
    
    #Predict top classes
    top_p, top_indices = ps.topk(topk)
    top_p = top_p.detach().type(torch.FloatTensor).numpy().tolist()[0]
    top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0]
    
    # Convert indices to classes
    idx_to_class = {value: key for key, value in
                    model.class_to_idx.items()}
    
    #print(idx_to_class)
    
    top_flowers = [idx_to_class[index] for index in top_indices]
    
    #Convert from the class integer encoding to actual flower names
    #top_flower_name = [cat_to_name[i] for i in top_flowers[0]]
    #print(top_flower_name)
    return top_p, top_flowers 