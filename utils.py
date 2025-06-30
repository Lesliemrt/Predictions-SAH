import torch
import torch.nn.functional as F
from torchvision import transforms
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pydicom
from collections import OrderedDict
from torch.utils.data.dataloader import default_collate

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
    # windowing
    brain_img = window_image(dcm, 40, 80)
    subdural_img = window_image(dcm, 80, 200)
    soft_img = window_image(dcm, 40, 380)
    # normalization
    brain_img = (brain_img - 0) / 80
    subdural_img = (subdural_img - (-20)) / 200
    soft_img = (soft_img - (-150)) / 380
    bsb_img = torch.stack([brain_img, subdural_img, soft_img])
    return bsb_img

def _read(path):
    dcm = pydicom.dcmread(path)
    # print("===================DCM ======================")
    # print(dcm)
    # except Exception as e:
    #     # to delete when we will have all hospital data
    #     path_end=path.split(os.sep)[-5:]
    #     if path_end not in non_existant_file:
    #         non_existant_file.append(path_end)
    try:
        img = bsb_window(dcm)
    except Exception as e:
        # img = torch.zeros((configs.CHANNELS, configs.HEIGHT, configs.WIDTH))
        img = None
    return img

# Image Augmentation
def sometimes(prob, transform):
    return transforms.RandomApply([transform], p=prob)

# Ajust the image paths
def ajust_path(identifier):
    paciente, serie, imagen = identifier.split('-')
    path = f"{configs.DATA_DIR}hospital_data_1/raw data/{paciente}/DICOM/ST00001/{serie}/{imagen}"
    return path

def ajust_path_data2(identifier):
    patient, id1, id2, id3, image = identifier.split('-')
    path = f"{configs.DATA_DIR}hospital_data_2/{patient}/{id1}/{id2}/{id3}/{image}"
    return path

# Visualize random images from a dataset before training
def visualize(num_images_to_show, train_df):
    _ , axs = plt.subplots(1, num_images_to_show, figsize=(20, 5))
    if num_images_to_show == 1:
        axs = [axs]
    for i in range(num_images_to_show):
        random_index = np.random.randint(0, len(train_df))
        
        img_path = train_df['Path'].iloc[random_index]
        label = train_df['ANY_Vasospasm'].iloc[random_index]

        img = _read(img_path)
        
        plt.figure(figsize=(10, 5))
        img = img.permute(1, 2, 0).cpu().numpy()
        # plt.imshow(img, cmap='gray')
        # plt.title(f"Index: {random_index}, Etiqueta: {label}")
        # plt.axis('off')
        # plt.show()
        axs[i].imshow(img,cmap='gray')
        axs[i].set_title(f"Index: {random_index}, Etiqueta: {label}")
        axs[i].axis('off')
    plt.tight_layout()
    plt.savefig(f"{configs.DIR}/results/visualize before training.png") 
    plt.close()

# Remove .module dans state_dict and change features to densenet169 so the weights match
def adapt_name(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '').replace('densenet169.', 'features.')
        # name = k.replace('module.', '')
        new_state_dict[name] = v
    return new_state_dict

# Normalize min max between 0 and 1
def normalize_min_max(x, min, max):
    return (x - min) / (max - min)

def normalize_min_max_inverted(x, min, max):
    return (max - x) / (max - min)

def collate_remove_none(batch):
    # Supprime les items où 'image' est None
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


# Other preprocessing
def get_first_of_dicom_field_as_int(x):
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    return int(x)

def get_id(img_dicom):
    return str(img_dicom.SOPInstanceUID)

def get_metadata_from_dicom(img_dicom):
    metadata = {
        # "window_center": img_dicom.WindowCenter,
        # "window_width": img_dicom.WindowWidth,
        "intercept": img_dicom.RescaleIntercept,
        "slope": img_dicom.RescaleSlope,
    }
    return {k: get_first_of_dicom_field_as_int(v) for k, v in metadata.items()}

def window_image_new(img, window_center, window_width, intercept, slope):
    img = img.astype(np.float32) * slope + intercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    return img 

def normalize_minmax(img):
    mi, ma = img.min(), img.max()
    return (img - mi) / (ma - mi)

def _read_new(img_path):
    img_dicom = pydicom.dcmread(img_path)
    metadata = get_metadata_from_dicom(img_dicom)
    raw_img = img_dicom.pixel_array

    # 3 standard windows for head CT
    windows = [
        {"center": 40, "width": 80},    # brain
        {"center": 80, "width": 200},   # subdural
        {"center": 40, "width": 380}    # soft tissue
    ]

    channels = []
    for win in windows:
        img = window_image_new(raw_img, win["center"], win["width"], **metadata)
        img = normalize_minmax(img) * 255.0
        img_tensor = torch.tensor(img, dtype=torch.float32)
        channels.append(img_tensor)

    img = torch.stack(channels)  # Shape: [3, H, W]
    return img

# to extract ["PatientID", "SOPInstanceUID", "SeriesInstanceUID", "ImagePositionPatient2"]
def extract_dicom_info(dcm_path):
    try:
        dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        patient_id = dcm.PatientID
        sop_uid = dcm.SOPInstanceUID
        series_uid = dcm.SeriesInstanceUID
        ipp2 = float(dcm.ImagePositionPatient[2])  # Z-axis position
        return pd.Series([patient_id, sop_uid, series_uid, ipp2])
    except Exception as e:
        print(f"Erreur lecture {dcm_path}: {e}")
        return pd.Series([None, None, None, None])

def denormalize(img, means, stds):
    img = img.clone()
    for c in range(3):
        img[c] = img[c] * stds[c] + means[c]
    return img