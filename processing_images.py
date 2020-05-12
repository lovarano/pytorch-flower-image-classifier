# PROGRAMMER: Lorenzo Varano
# DATE CREATED: 2020.04.10
# PURPOSE: This part of the program contains functions to process an image in order to fit in the model.
# NOTE: This file was created mainly with support of the previous task in Part 1 and 1st project - use of a pretrained classifier. Additional support material is referenced in Part 1.

#import python modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from PIL import Image
import os, random
import json

# Define transforms for the training, validation, and testing sets
def data_transforms():
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       normalize])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      normalize])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      normalize])
    
    return train_transforms, valid_transforms, test_transforms


# Load the datasets with ImageFolder
def load_datasets(train_dir, train_transforms, valid_dir, valid_transforms, test_dir, test_transforms):
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    return train_data, valid_data, test_data

# Load class_to_name json file 
def load_json(json_file):
    
    with open(json_file, 'r') as f:
        cat_to_name = json.load(f)
        return cat_to_name

# Get a random image from folder ..15 for the arguments
def getrandomimage(folder):
    random_folder = random.choice(os.listdir(str(folder)))
    random_image = random.choice(os.listdir(str(folder) + str(random_folder)))
    random_image_path = folder + random_folder +"/" + random_image
    #print(random_image_path)
    return random_image_path

# Function for processing a PIL image for use in the PyTorch model
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    #Load image
    image = Image.open(image_path)
    
    #Get height and width of image
    width, height = image.size
    
    #Resize image
    image = image.resize((255, int(255*(height/width))) if width < height else (int(255*(width/height)), 255))      
        
    # Get the dimensions of the new image size
    width, height = image.size
    
    #Do center crop
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    image = image.crop((left, top, right, bottom))
    
    # Turn into numpy array
    image = np.array(image)
       
    #Make all values between 0 and 1
    image = image/255
    
    #Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    # Make the color channel dimension first instead of last
    image = image.transpose((2, 0, 1))
    
    return image

# Function to convert a PyTorch tensor and display it
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
    
    
# Function to display an image along with the top classes
def display_image(image_dir, cat_to_name, classes):

    image = process_image(image_dir)

    image = process_image(image_dir)  
    key = image_dir.split('/')[-2]
    flower_title = cat_to_name[key]

    #Convert from the class integer encoding to actual flower names
    flower_names = [cat_to_name[i] for i in classes]
    return flower_names
