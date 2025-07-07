import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import configs
import utils

# to understand the data we have for multi classes
df = pd.read_excel(f'{configs.DATA_DIR}excel_predicciones.xlsx', sheet_name='datos hospital')

def histo(label):
    data=df[label]
    serie = pd.Series(data)

    q1 = serie.quantile(1/3)
    q2 = serie.quantile(2/3)

    # Affichage des bornes
    print(f'Label : {label}')
    print(f"Group 1 : <= {q1:.2f}")
    print(f"Group 2 : > {q1:.2f} and <= {q2:.2f}")
    print(f"Group 3 : > {q2:.2f}")

    # Construction de l'histogramme
    plt.hist(serie, bins=10, edgecolor='black')
    plt.axvline(q1, color='r', linestyle='dashed', linewidth=1.5, label=f'1st tertile ({q1:.2f})')
    plt.axvline(q2, color='g', linestyle='dashed', linewidth=1.5, label=f'2nd tertile ({q2:.2f})')
    plt.title(f"Histogram - {label}")
    plt.xlabel("Time (days)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{configs.DIR}/results/histogram - {label}.png") 
    plt.close()

histo('DiasVM')
histo('DiasUCI')
