import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import pandas as pd
import numpy as np
import pydicom
from albumentations import Compose, Resize, CenterCrop
import cv2
import csv

import configs
import utils

# to understand the data we have for multi classes
df = pd.read_excel(f'{configs.DATA_DIR}excel_predicciones.xlsx', sheet_name='completa_datos')

def histo(label):
    data=df[label]
    serie = pd.Series(data)

    q1 = serie.quantile(1/3)
    q2 = serie.quantile(2/3)

    # Affichage des bornes
    print(f"Groupe 1 : <= {q1:.2f}")
    print(f"Groupe 2 : > {q1:.2f} et <= {q2:.2f}")
    print(f"Groupe 3 : > {q2:.2f}")

    # Construction de l'histogramme
    plt.hist(serie, bins=10, edgecolor='black')
    plt.axvline(q1, color='r', linestyle='dashed', linewidth=1.5, label=f'1er tiers ({q1:.2f})')
    plt.axvline(q2, color='g', linestyle='dashed', linewidth=1.5, label=f'2e tiers ({q2:.2f})')
    plt.title("Histogramme des durées")
    plt.xlabel("Durée (jours)")
    plt.ylabel("Fréquence")
    plt.legend()
    plt.grid(True)
    plt.show()

histo('DiasVM')
