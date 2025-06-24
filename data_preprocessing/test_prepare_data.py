import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from albumentations import *
import torch
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import configs
import utils

# To test sah predictions with pretrained model on hospital_data_1 to compare performances

# Create an excel with identifier for each image
images = []
identifiers = []
# Loop through each image file
all_patients = os.listdir(f"{configs.DATA_DIR}hospital_data_1/raw data")
all_patients.remove('__MACOSX')
all_patients.remove('Data List HSA.xlsx')
all_patients.remove('Data List HSA.numbers')
all_patients.sort()
for patient in all_patients:
    all_id1 = os.listdir(f"{configs.DATA_DIR}hospital_data_1/raw data/{patient}/DICOM/ST00001")
    all_id1 = [f for f in all_id1 if not f.startswith('.')]
    all_id1.sort()
    for id1 in all_id1:
        all_images = os.listdir(f"{configs.DATA_DIR}hospital_data_1/raw data/{patient}/DICOM/ST00001/{id1}")
        all_images = [f for f in all_images if not f.startswith('.')]
        all_images.sort() # to order the slices
        for image in all_images : 
            images.append(image)
            identifier = f"{patient}-{id1}-{image}"
            identifiers.append(identifier)

predictions_df = pd.DataFrame()
predictions_df["Identifier"] = identifiers
predictions_df["Path"] = predictions_df["Identifier"].apply(utils.ajust_path)

#remove image none
invalid_paths = []
for i in range(len(predictions_df)):
    path = predictions_df['Path'].iloc[i]
    img = utils._read(path)
    if img is None:
        invalid_paths.append(path)
print("len de invalid_paths : ",len(invalid_paths))
predictions_df = predictions_df[predictions_df['Path'].isin(invalid_paths) == False].reset_index(drop=True)

# Visualize an image with preprocessing
# image_path = utils.ajust_path_data2(predictions_df['Identifier'][0])
# print(image_path)
# image = utils._read(image_path)
# image = image.permute(1, 2, 0).cpu().numpy()
# plt.axis('off')
# plt.imshow(image)
# plt.savefig(f"{configs.DIR}/results/visualize new data test.png") 
# plt.close()

# 4. Add dicom informations to order the slices
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
predictions_df[["PatientID", "SOPInstanceUID", "SeriesInstanceUID", "ImagePositionPatient2"]] = predictions_df["Path"].apply(extract_dicom_info)
predictions_df = predictions_df.sort_values(["PatientID", "SeriesInstanceUID", "ImagePositionPatient2"]).reset_index(drop=True)
predictions_df["pre1_SOPInstanceUID"] = predictions_df.groupby(["PatientID", "SeriesInstanceUID"])["SOPInstanceUID"].shift(1)
predictions_df["post1_SOPInstanceUID"] = predictions_df.groupby(["PatientID", "SeriesInstanceUID"])["SOPInstanceUID"].shift(-1)

predictions_df.to_excel('/export/usuarios01/lmurat/Datos/Predictions-SAH/data_preprocessing/test_predictions.xlsx', sheet_name="identifiers", index=False)






# print("debut data loadiiiinnnnggg ...........")

# # # -----------------------------------------Test 1 ------------------------------------
# # # Dataloader for predictions
# # test_dataset = TestDataset(
# #     dataset=predictions_df,
# #     batch_size=configs.TEST_BATCH_SIZE,
# # )

# # testloader = DataLoader(test_dataset, batch_size=configs.TEST_BATCH_SIZE, shuffle=False, collate_fn=utils.collate_remove_none)

# # # Load model + weights
# # from keras_to_pytorch.densenet_from_IR import KitModel
# # my_model = KitModel("/export/usuarios01/lmurat/Datos/Predictions-SAH/keras_to_pytorch/densenet_from_IR_weights.npy")

# # -----------------------------------------Test 2 ------------------------------------
# # Dataloader for predictions
# img_size = 512
# test_augmentation = Compose([
#     CenterCrop(512 - 50, 512 - 50, p=1.0),
#     Resize(img_size, img_size, p=1)
# ])


# test_dataset = RSNADatasetTest(predictions_df, img_size, id_colname="SOPInstanceUID",
#                             transforms=test_augmentation, black_crop=False, subdural_window=True,
#                             n_tta=2)
# test_loader = DataLoader(test_dataset, batch_size=configs.TEST_BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)
# del test_dataset
# # gc.collect()

# # Load model + weights
# model_path = "/export/usuarios01/lmurat/Datos/Predictions-SAH/Data/exp16_seres_ep5.pth"
# model = CnnModel(num_classes=6, encoder="se_resnext50_32x4d", pretrained="imagenet")
# model.load_state_dict(torch.load(model_path))
# model = torch.nn.DataParallel(model, device_ids=[configs.device.index])
# model.to(configs.device)

# print("debut predict ...................")

# def predict(model, test_loader, device, n_tta=1, flip_aug=False):
#     model.eval()
#     preds_cat = []
#     with torch.no_grad():
#         for step, imgs in enumerate(test_loader):
#             features = imgs[0].to(device)
#             logits = model(features)

#             if n_tta >= 2:
#                 flip_img = imgs[1].to(device)
#                 logits += model(flip_img)

#             logits = logits / n_tta

#             logits = torch.sigmoid(logits).float().cpu().numpy()
#             preds_cat.append(logits)

#         all_preds = np.concatenate(preds_cat, axis=0)
#     return all_preds

# pred = predict(model, test_loader, configs.device, n_tta=2)
# pred = np.clip(pred, 1e-6, 1-1e-6)

# print(pred.shape)
# hemorrhage_types = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
# for i, col in enumerate(hemorrhage_types):
#     predictions_df[col] = pred[:, i]

# print("predictions_df")
# print(predictions_df)
# predictions_df.to_excel('test_predictions.xlsx', index=False)






# # Predicts
# # my_model.to(configs.device)
# # my_model.eval()

# # all_probs = []
# # with torch.no_grad():
# #     for batch in testloader : 
# #         images = batch['image']
# #         images = images.float().to(configs.device)
# #         outputs = my_model(images.to(configs.device))
# #         probs = torch.sigmoid(outputs)
# #         all_probs.append(probs.cpu())

# # all_probs = torch.cat(all_probs).numpy()
# # print(all_probs.shape)
# # hemorrhage_types = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
# # for i, col in enumerate(hemorrhage_types):
# #     predictions_df[col] = all_probs[:, i]

# # print("predictions_df")
# # print(predictions_df)
# # predictions_df.to_excel('test_predictions.xlsx', index=False)