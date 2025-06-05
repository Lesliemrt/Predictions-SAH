import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import configs
import utils


# Create an excel with identifier for each image
images = []
identifiers = []
# Loop through each image file
all_patients = os.listdir(f"{configs.DATA_DIR}hospital_data_2")
for patient in all_patients:
    all_id1 = os.listdir(f"{configs.DATA_DIR}hospital_data_2/{patient}")
    files = ['DICOMDIR', 'IHE_PDI', 'INDEX.HTM', 'RUN.CMD', 'JRE', 'XTR_CONT', 'REPORT', 'PLUGINS', 'README.TXT', 'RUN.COMMAND', 'AUTORUN.INF', 'LOG4J.XML', 'HELP', 'CDVIEWER.EXE', 'RUN.SH']
    for file in files:
        if file in all_id1 :
            all_id1.remove(file)
    id1 = all_id1[0]
    print(id1)
    all_id2 = os.listdir(f"{configs.DATA_DIR}hospital_data_2/{patient}/{id1}")
    for id2 in all_id2:
        all_id3 = os.listdir(f"{configs.DATA_DIR}hospital_data_2/{patient}/{id1}/{id2}")
        for id3 in all_id3 : 
            all_images= os.listdir(f"{configs.DATA_DIR}hospital_data_2/{patient}/{id1}/{id2}/{id3}")
            for image in all_images:
                images.append(image)
                identifier = f"{patient}-{id1}-{id2}-{id3}-{image}"
                identifiers.append(identifier)

predictions_df = pd.DataFrame()
predictions_df["Identifier"] = identifiers

print(predictions_df.tail(50))

# Visualize an image with preprocessing
image_path = utils.ajust_path_data2(predictions_df['Identifier'][0])
print(image_path)
image = utils._read(image_path)
image = image.permute(1, 2, 0).cpu().numpy()
plt.axis('off')
plt.imshow(image)
plt.savefig(f"{configs.DATA_DIR}/results/visualize new data test.png") 
plt.close()



