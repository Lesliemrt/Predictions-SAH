import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from albumentations import *
import torch
from torch import nn
from torchvision.models import densenet169
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import configs
import utils
from dataloader import TestDataset, RSNADatasetTest
from model import DenseNet169_change_avg, Model_6_classes, CnnModel

# 1. Take the excel with identifier for each images
excel_path = '/export/usuarios01/lmurat/Datos/Predictions-SAH/data_preprocessing/test_predictions.xlsx'
predictions_df = pd.read_excel(excel_path, sheet_name = "identifiers")

print("torch.cuda.is_available()", torch.cuda.is_available())  # Doit retourner True si un GPU est détecté
print("torch.cuda.device_count()", torch.cuda.device_count())  # Nombre de GPUs disponibles
print("torch.cuda.get_device_name(0)", torch.cuda.get_device_name(0))  # Nom du GPU (si disponible)


# # -----------------------------------------Test 1 -----------------------------------------------
# # 2. Create dataloader for predictions

# print("debut data loadiiiinnnnggg ...........")
# test_dataset = TestDataset(
#     dataset=predictions_df,
#     batch_size=configs.TEST_BATCH_SIZE,
# )

# testloader = DataLoader(test_dataset, batch_size=configs.TEST_BATCH_SIZE, shuffle=False, collate_fn=utils.collate_remove_none)

# # 3. Load model + weights
# from keras_to_pytorch.densenet_from_IR import KitModel
# my_model = KitModel("/export/usuarios01/lmurat/Datos/Predictions-SAH/keras_to_pytorch/densenet_from_IR_weights.npy")

# 4. Predicts
# my_model.to(configs.device)
# my_model.eval()

# all_probs = []
# with torch.no_grad():
#     for batch in testloader : 
#         images = batch['image']
#         images = images.float().to(configs.device)
#         outputs = my_model(images.to(configs.device))
#         probs = torch.sigmoid(outputs)
#         all_probs.append(probs.cpu())

# all_probs = torch.cat(all_probs).numpy()
# print(all_probs.shape)
# hemorrhage_types = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
# for i, col in enumerate(hemorrhage_types):
#     predictions_df[col] = all_probs[:, i]

# print("predictions_df")
# print(predictions_df)
# predictions_df.to_excel('test_predictions.xlsx', index=False)

# -----------------------------------------Test 2 ------------------------------------------------

# 2. Create dataloader for predictions
img_size = 512
test_augmentation = Compose([
    CenterCrop(512 - 50, 512 - 50, p=1.0),
    Resize(img_size, img_size, p=1)
])


test_dataset = RSNADatasetTest(predictions_df, img_size, id_colname="SOPInstanceUID",
                            transforms=test_augmentation, black_crop=False, subdural_window=True,
                            n_tta=2)
test_loader = DataLoader(test_dataset, batch_size=configs.TEST_BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)
del test_dataset
# gc.collect()

# 3. Load model + weights
model_path = "/export/usuarios01/lmurat/Datos/Predictions-SAH/Data/exp16_seres_ep5.pth"
model = CnnModel(num_classes=6, encoder="se_resnext50_32x4d", pretrained="imagenet")
model.load_state_dict(torch.load(model_path,  map_location=configs.device))
model.to(configs.device)
model = torch.nn.DataParallel(model, device_ids=[1])

# 4. Predictions
print("debut predict ...................")

def predict(model, test_loader, device, n_tta=1, flip_aug=False):
    model.eval()
    preds_cat = []
    with torch.no_grad():
        for step, imgs in tqdm(enumerate(test_loader), total=len(test_loader), desc="Predicting"):
            features = imgs[0].to(device)
            print(f"Batch {step} size: {features.shape[0]}, img size : {features.shape[2:] }")
            logits = model(features)

            if n_tta >= 2:
                flip_img = imgs[1].to(device)
                # print(f"flip : Batch {step} size: {flip_img.shape[0]}, img size : {flip_img.shape[2:] }")
                logits += model(flip_img)

            logits = logits / n_tta

            logits = torch.sigmoid(logits).float().cpu().numpy()
            preds_cat.append(logits)

        all_preds = np.concatenate(preds_cat, axis=0)
    return all_preds

pred = predict(model, test_loader, configs.device, n_tta=2)
pred = np.clip(pred, 1e-6, 1-1e-6)

print(pred.shape)
hemorrhage_types = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
for i, col in enumerate(hemorrhage_types):
    predictions_df[col] = pred[:, i]

print("predictions_df")
print(predictions_df)

# 5. Save the excel
with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    predictions_df.to_excel(writer, sheet_name='predictions', index=False)