import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import os
import matplotlib.pyplot as plt
import pydicom

import configs


# Image windowing
def correct_dcm(dcm):
    x = dcm.pixel_array + 1000
    px_mode = 4096
    x[x>=px_mode] = x[x>=px_mode] - px_mode
    dcm.PixelData = x.tobytes()
    dcm.RescaleIntercept = -1000

def window_image(dcm, window_center, window_width):
    if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):
        correct_dcm(dcm)
    img = dcm.pixel_array.astype(np.float32) * dcm.RescaleSlope + dcm.RescaleIntercept


    # Resize
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # shape [1, 1, H, W]
    img = F.interpolate(img, size=(configs.HEIGHT, configs.WIDTH)[:2], mode='bilinear', align_corners=False)
    img = img.squeeze(0).squeeze(0)  # back to shape [H, W]

    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = torch.clamp(img, min=img_min, max=img_max)

    return img

def bsb_window(dcm):
    brain_img = window_image(dcm, 40, 80)
    subdural_img = window_image(dcm, 80, 200)
    soft_img = window_image(dcm, 40, 380)
    brain_img = (brain_img - 0) / 80
    subdural_img = (subdural_img - (-20)) / 200
    soft_img = (soft_img - (-150)) / 380
    bsb_img = torch.stack([brain_img, subdural_img, soft_img])
    return bsb_img

non_existant_file=[]
def _read(path, SHAPE):
    try:
        dcm = pydicom.dcmread(path)
    except Exception as e:
        # to delete when we will have all hospital data
        last_files=path.split(os.sep)[-5:]
        if last_files not in non_existant_file:
            non_existant_file.append(last_files)
    try:
        img = bsb_window(dcm)
    except Exception as e:
        img = torch.zeros((configs.CHANNELS, configs.HEIGHT, configs.WIDTH))
    return img

# Image Augmentation
def sometimes(prob, transform):
    return transforms.RandomApply([transform], p=prob)

# Ajust the image paths
def ajust_path(identifier):
    paciente, serie, imagen = identifier.split('-')
    path = f"{configs.DATA_DIR}hospital data 1\\raw data\\{paciente}\\DICOM\\ST00001\\{serie}\\{imagen}"
    return path

# Visualize random images from a dataset before training
def visualize(num_images_to_show, train_df):
    for i in range(num_images_to_show):
        # random_index = np.random.randint(0, len(train_df))
        random_index = 56
        
        img_path = train_df['Path'].iloc[random_index]
        label = train_df['ANY_Vasospasm'].iloc[random_index]
        
        img = _read(img_path, (configs.CHANNELS,configs.HEIGHT, configs.WIDTH))
        
        plt.figure(figsize=(10, 5))
        img = img.permute(1, 2, 0).cpu().numpy()
        plt.imshow(img, cmap='gray')
        plt.title(f"Index: {random_index}, Etiqueta: {label}")
        plt.axis('off')
        plt.show()