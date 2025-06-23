import pandas as pd
import matplotlib.pyplot as plt
import pydicom
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils
import configs

# 1. Take the excel with identifier for each images and probability of hemorrage
excel_path = '/export/usuarios01/lmurat/Datos/Predictions-SAH/data_preprocessing/excel_new_data_prepared.xlsx'
df = pd.read_excel(excel_path, sheet_name = "predictions")

# 2. Add a column with patient ID (HSA 1, HSA 2, ...)
df['Patient ID'] = df['Identifier'].apply(lambda x: x.split('-')[0])

# 3. Select 13 rows per patient
selected_rows = []

for id, group_df in df.groupby('Patient ID'):
    # center_idx = len(group_df) // 2
    center_idx = group_df['subarachnoid'].values.argmax() # to not take the middle index but the one with max probability of hemorrage
    start_idx = max(center_idx - 6, 0)
    end_idx = start_idx + 13
    selected = group_df.iloc[start_idx:end_idx]
    selected_rows.append(selected)

df_new_data_prepared = pd.concat(selected_rows).reset_index(drop=True)

# 4. Add dicom informations to order the slices
df_new_data_prepared.drop(["Patient ID"], axis=1)
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
df_new_data_prepared[["PatientID", "SOPInstanceUID", "SeriesInstanceUID", "ImagePositionPatient2"]] = df_new_data_prepared["Path"].apply(extract_dicom_info)
df_new_data_prepared = df_new_data_prepared.sort_values(["PatientID", "SeriesInstanceUID", "ImagePositionPatient2"]).reset_index(drop=True)
df_new_data_prepared["pre1_SOPInstanceUID"] = df_new_data_prepared.groupby(["PatientID", "SeriesInstanceUID"])["SOPInstanceUID"].shift(1)
df_new_data_prepared["post1_SOPInstanceUID"] = df_new_data_prepared.groupby(["PatientID", "SeriesInstanceUID"])["SOPInstanceUID"].shift(-1)

# 5. Visualize an image with preprocessing
for k in range(0, 30):
    image_path = utils.ajust_path_data2(df_new_data_prepared['Identifier'][k])
    image = utils._read(image_path)
    image = image.permute(1, 2, 0).cpu().numpy()
    plt.axis('off')
    identifier = df_new_data_prepared['Identifier'][k]
    plt.title(f'{identifier}')
    plt.imshow(image)
    print(f"saving {k}")
    plt.savefig(f"{configs.DIR}/results/visualize new data test {k}.png") 
    plt.close()


# 6. Save into a new sheet in the excel
df_new_data_prepared = df_new_data_prepared.drop(['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural'], axis = 1)
with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    df_new_data_prepared.to_excel(writer, sheet_name='selected_slices', index=False)
print(df_new_data_prepared)
