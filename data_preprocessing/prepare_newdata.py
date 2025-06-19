import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.models import densenet169
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import configs
import utils
from dataloader import TestDataset

# To make sah predictions and choose slices with pretrained model on hospital_data_2

# 1. Create an excel with identifier for each image
images = []
identifiers = []
# Loop through each image file
all_patients = os.listdir(f"{configs.DATA_DIR}hospital_data_2")
all_patients.remove('HSA 242')
all_patients.sort()
for patient in all_patients:
    all_id1 = os.listdir(f"{configs.DATA_DIR}hospital_data_2/{patient}")
    files = ['DICOMDIR', 'IHE_PDI', 'INDEX.HTM', 'RUN.CMD', 'JRE', 'XTR_CONT', 'REPORT', 'PLUGINS', 'README.TXT', 'RUN.COMMAND', 'AUTORUN.INF', 'LOG4J.XML', 'HELP', 'CDVIEWER.EXE', 'RUN.SH']
    for file in files:
        if file in all_id1 :
            all_id1.remove(file)
    id1 = all_id1[0]
    all_id2 = os.listdir(f"{configs.DATA_DIR}hospital_data_2/{patient}/{id1}")
    for id2 in all_id2:
        all_id3 = os.listdir(f"{configs.DATA_DIR}hospital_data_2/{patient}/{id1}/{id2}")
        for id3 in all_id3 : 
            all_images= os.listdir(f"{configs.DATA_DIR}hospital_data_2/{patient}/{id1}/{id2}/{id3}")
            all_images.sort() # to order the slices
            for image in all_images:
                images.append(image)
                identifier = f"{patient}-{id1}-{id2}-{id3}-{image}"
                identifiers.append(identifier)

predictions_df = pd.DataFrame()
predictions_df["Identifier"] = identifiers
predictions_df["Path"] = predictions_df["Identifier"].apply(utils.ajust_path_data2)

# 2. Remove image == none
invalid_paths = []
for i in range(len(predictions_df)):
    path = predictions_df['Path'].iloc[i]
    img = utils._read(path)
    if img is None:
        invalid_paths.append(path)
print("len de invalid_paths : ",len(invalid_paths))
predictions_df = predictions_df[predictions_df['Path'].isin(invalid_paths) == False].reset_index(drop=True)

# # Visualize an image with preprocessing
# for k in range(20):
#     image_path = utils.ajust_path_data2(predictions_df['Identifier'][k])
#     print(image_path)
#     image = utils._read(image_path)
#     image = image.permute(1, 2, 0).cpu().numpy()
#     plt.axis('off')
#     plt.title(f'{predictions_df['Identifier'][k]}')
#     plt.imshow(image)
#     plt.savefig(f"{configs.DIR}/results/visualize new data test {k}.png") 
#     plt.close()

# 3. Dataloader for preprocessing
test_dataset = TestDataset(
    dataset=predictions_df,
    batch_size=configs.TEST_BATCH_SIZE,
)

testloader = DataLoader(test_dataset, batch_size=configs.TEST_BATCH_SIZE, shuffle=False, collate_fn=utils.collate_remove_none)

# 4. Load model + weights
# Model transfered from .h5 to pytorch
from keras_to_pytorch.densenet_from_IR import KitModel
my_model = KitModel("/export/usuarios01/lmurat/Datos/Predictions-SAH/keras_to_pytorch/densenet_from_IR_weights.npy")

# 5. Predicts
print("debut predict ...........................")
my_model.to(configs.device)
my_model.eval()

all_probs = []
with torch.no_grad():
    for batch in testloader : 
        images = batch['image']
        images = images.float().to(configs.device)
        outputs = my_model(images.to(configs.device))
        probs = torch.sigmoid(outputs)
        all_probs.append(probs.cpu())

all_probs = torch.cat(all_probs).numpy()
print(all_probs.shape)
hemorrhage_types = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
for i, col in enumerate(hemorrhage_types):
    predictions_df[col] = all_probs[:, i]

print("predictions_df")
print(predictions_df)
predictions_df.to_excel('/export/usuarios01/lmurat/Datos/Predictions-SAH/data_preprocessing/excel_new_data_prepared.xlsx', sheet_name = 'predictions', index=False)

