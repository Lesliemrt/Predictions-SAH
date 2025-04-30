# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 10:31:30 2024

@author: scluc
"""
import numpy as np
import pandas as pd
import pydicom
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf


# Constants
HEIGHT = 256
WIDTH = 256
CHANNELS = 3
SHAPE = (HEIGHT, WIDTH, CHANNELS)

# Define functions for image processing and prediction
# DICOM WINDOWING AND DATA GENERATORS

# Function to correct DICOM images
def correct_dcm(dcm):
    x = dcm.pixel_array + 1000
    px_mode = 4096
    x[x >= px_mode] -= px_mode
    dcm.PixelData = x.tobytes()
    dcm.RescaleIntercept = -1000

# Function to adjust the window from DICOM images
def window_image(dcm, window_center, window_width):    
    if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):
        correct_dcm(dcm)
    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    
    # resize the image
    img = cv2.resize(img, SHAPE[:2], interpolation=cv2.INTER_LINEAR)
   
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    return img.astype(np.float32)  

def bsb_window(dcm):
    brain_img = window_image(dcm, 40, 80)
    subdural_img = window_image(dcm, 80, 200)
    soft_img = window_image(dcm, 40, 380)
    brain_img = (brain_img - 0) / 80
    subdural_img = (subdural_img - (-20)) / 200
    soft_img = (soft_img - (-150)) / 380
    bsb_img = np.array([brain_img, subdural_img, soft_img]).transpose(1, 2, 0)
    return bsb_img

# To read and preprocess the images
def _read_and_preprocess_image(path):
    try:
        dcm = pydicom.dcmread(path)
        img = bsb_window(dcm)
    except FileNotFoundError:
        img = np.zeros(SHAPE)
    return img

# Path to the folder containing the images
folder_path = "C:\\Users\\Lesli\\Documents\\Doc administratif\\2024-2025\\Madrid\\Stage\\Projet predictions SAH\\hospital data 1\\raw data\\HSA 1\\DICOM\\ST00001\\SE00003"

# Get a list of all files in the folder
image_files = os.listdir(folder_path)

# Initialize lists to store images and identifiers
images = []
identifiers = []

# Loop through each image file
for file in image_files:
    # Construct the full path to the image
    image_path = os.path.join(folder_path, file)
    
    # Read and preprocess the image
    image = _read_and_preprocess_image(image_path)
    
    # Append the image and its identifier to the lists
    images.append(image)
    identifier = "HSA 1-SE00003-" + file.split(".")[0] # Assuming file names are of the form "IMxxxxx.dcm"
    identifiers.append(identifier)

# Convert lists to numpy arrays
images = np.array(images)

# Load the pre-trained model
model = tf.keras.models.load_model("C:\\Users\\Lesli\\Documents\\Doc administratif\\2024-2025\\Madrid\\Stage\\Projet predictions SAH\\densenet169_model.h5")

# Make predictions for all images
predictions = model.predict(images)

# Define the hemorrhage types
hemorrhage_types = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']

# Create a DataFrame with the probabilities
predictions_df = pd.DataFrame(predictions, columns=hemorrhage_types)

# Add the identifiers as a column to the DataFrame
predictions_df["Identifier"] = identifiers

# Check if the CSV file already exists
predictions_csv_path = "predictions.csv"
if os.path.isfile(predictions_csv_path):
    # If the CSV file already exists, load it and append new predictions
    existing_predictions_df = pd.read_csv(predictions_csv_path)
    predictions_df = pd.concat([existing_predictions_df, predictions_df], ignore_index=True)

# Save the updated predictions to the CSV file
predictions_df.to_csv(predictions_csv_path, index=False)
print("Predictions saved to:", predictions_csv_path)


