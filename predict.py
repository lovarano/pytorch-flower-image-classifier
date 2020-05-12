# PROGRAMMER: Lorenzo Varano
# DATE CREATED: 2020.04.11
# PURPOSE: This part of the program predict the type of a flower in an image.
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

# Get gpu if available for the arguments
pre_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get a random image from subfolder 15 in test data for the arguments
folder = 'flowers/train/'
pre_image_path = processing_images.getrandomimage(folder)

# Reference https://pymotw.com/3/argparse/
# Create Parse using ArgumentParser
parser = argparse.ArgumentParser(description='Flower Image Classifier')

# Command line arguments
parser.add_argument('--arch', type = str, default = 'vgg', help = 'Choose model architecture - vgg or alexent')
parser.add_argument('--image_path', type = str, default = pre_image_path, help = 'Path to image')
parser.add_argument('--checkpoint', type = str, default = 'checkpoint.pth', help = 'Path to checkpoint')
parser.add_argument('--topk', type = int, default = 5, help = 'Top k classes and probabilities')
parser.add_argument('--device', type = str, default = pre_device, help = 'choose device - cuda or cpu')
parser.add_argument('--json_file', type = str, default = 'cat_to_name.json', help = 'class_to_name json file')


# Replace None with parser.parse_args() arguments in default
arguments = parser.parse_args()

check_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Verify device if GPU or CPU
device = classify_images.check_gpu(arguments.device, check_device)

# Load in a mapping from category label to category name
class_to_name_dict = processing_images.load_json(arguments.json_file)

# Load the pretrained model 
model = classify_images.load_checkpoint(arguments.checkpoint)

#checkpoint = torch.load(arguments.checkpoint, map_location=device)
print("Your model is loaded: {}".format(model))  

# Scales, crops, and normalizes a PIL image for the PyTorch model; returns a Numpy array
image = processing_images.process_image(arguments.image_path)

# Display image
#processing_images.imshow(image)

#Print which file will be processed
print("The following image will be now processed: {}".format(arguments.image_path))

# Highest k probabilities and the indices of those probabilities
probabilities, classes = classify_images.predict(arguments.image_path, model, arguments.topk, device, class_to_name_dict) 

# Display the image along with the top 5 classes
flower_names = processing_images.display_image(arguments.image_path, class_to_name_dict, classes)

#Print the most likely class and the associated probability
print("The most likely class is: {} and the probability is: {:.3f}%.".format(flower_names[0], probabilities[0]*100))

#Print the top classes and the associated probabilities
flower_names = [class_to_name_dict[i] for i in classes]
counter = 0
for probability, flower_name in zip(probabilities, flower_names):
    counter += 1
    print("Top class {}: {} and the probability is {:.3f}%".format(counter, flower_name, probability*100))

