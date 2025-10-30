import re
import os

import pandas as pd
from PyPDF2 import PdfReader

DATA_DIR = '/Users/Lesli/Documents/Doc administratif/2024-2025/Madrid/Stage/Predictions-SAH/data/'
def ajust_path_data2(identifier):
    patient = identifier
    path = f"{DATA_DIR}hospital_data_2/{patient}/REPORT/REPORT.pdf"
    return path

df = pd.read_excel(f'{DATA_DIR}excel_predicciones2.xlsx', sheet_name='completa_datos')
df['Path'] = df['HSA'].apply(ajust_path_data2)

# print(df['Path'])

pdf = PdfReader(df['Path'][0])
texte = pdf.pages[0].extract_text()
num_historia = re.search(r"Nº Historia:\s*(\d+)", texte).group(1)
nom_patient = re.search(r"Nombre del Paciente:\s*([A-ZÁÉÍÓÚÑ ,]+)", texte).group(1)
print(f"num : {num_historia},nom du patient : {nom_patient}")


for k, path in enumerate(df['Path']):
    if not os.path.exists(path):
        continue
    pdf = PdfReader(path)
    texte = pdf.pages[0].extract_text()
    nhc= re.search(r"Nº Historia:\s*(\d+)", texte).group(1)
    patient = re.search(r"Nombre del Paciente:\s*([A-ZÁÉÍÓÚÑ ,]+)", texte).group(1)
    df['NHC'][k] = nhc
    df['Nombre'][k] = patient

with pd.ExcelWriter(f'{DATA_DIR}excel_predicciones2.xlsx', mode = 'a', if_sheet_exists='replace') as writer:  
    df.to_excel(writer, sheet_name='completa_datos', index = False)