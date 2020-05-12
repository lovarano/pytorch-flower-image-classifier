# PROGRAMMER: Lorenzo Varano
# DATE CREATED: 2020.04.10
# PURPOSE: This part of the program trains a model in order to classify flowers and their type.
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
import classify_images
import processing_images
import argparse

# Reference https://pymotw.com/3/argparse/
# Create Parse using ArgumentParser
parser = argparse.ArgumentParser(description='Flower Image Classifier')

# Command line arguments
parser.add_argument('--data_dir', type = str, default = 'flowers/', help = 'path to the folder of images')
parser.add_argument('--train_dir', type = str, default = 'flowers/train', help = 'path to the folder of train dataset')
parser.add_argument('--valid_dir', type = str, default = 'flowers/valid', help = 'path to the folder of validation dataset')
parser.add_argument('--test_dir', type = str, default = 'flowers/test', help = 'path to the folder of testing dataset')
parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth', help = 'Path to save checkpoint')
parser.add_argument('--device', type = str, default = 'cuda', help = 'choose device - cuda or cpu')
parser.add_argument('--arch', type = str, default = 'vgg', help = 'Choose model architecture - vgg or alexent')
parser.add_argument('--hidden_layers', type = int, default = 1024, help = 'Neurons in the Hidden Layer')
parser.add_argument('--epochs', type = int, default = 10, help = 'number of epochs for the training')
parser.add_argument('--learning_rate', type = float, default = 0.002, help = 'Learning Rate')

# Replace None with parser.parse_args() arguments in default
arguments = parser.parse_args()

check_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Verify device if GPU or CPU
device = classify_images.check_gpu(arguments.device, check_device)

# Define training, validation, and testing sets
train_transforms, valid_transforms, test_transforms = processing_images.data_transforms()

# Load the datasets with ImageFolder
train_data, valid_data, test_data = processing_images.load_datasets(arguments.train_dir, train_transforms, arguments.valid_dir, valid_transforms, arguments.test_dir, test_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

# Load the model architecture and print the model
input_size, model = classify_images.load_pretrained_model(arguments.arch)
print("Your pretrained model is loaded: {}".format(model))

# Freeze pretrained model parameters
for parameter in model.parameters():
    parameter.requires_grad = False

# Build a classifier
model.classifier = nn.Sequential(nn.Linear(input_size, arguments.hidden_layers),
                                 nn.ReLU(),
                                 nn.Dropout(0.4),
                                 nn.Linear(arguments.hidden_layers, 102),
                                 nn.LogSoftmax(dim=1))

#Set Criterion
criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=arguments.learning_rate)

# Train the classifier
classify_images.train_classifier(model, optimizer, criterion, trainloader, testloader, device, arguments.epochs)

# Do validation on the test set
classify_images.test_accuracy(model, testloader, device)

# Save the checkpoints
classify_images.save_checkpoint(model, train_data, arguments.epochs, arguments.save_dir, arguments.hidden_layers, arguments.learning_rate, arguments.arch)

# Inform user that model was trained
print("The model has been trained and saved under path {}".format(arguments.save_dir))
