# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 12:34:19 2024

@author: scluc
"""
import pandas as pd

# Leer el archivo original
archivo_excel = 'excel_predicciones.xlsx'
nombre_hoja = 'subaracnoidea_predicciones'
df = pd.read_excel(archivo_excel, sheet_name=nombre_hoja)

# Paso 1: Encontrar el valor máximo de 'subarachnoid' para cada grupo
max_values = df.groupby('Group').agg({'subarachnoid': 'max'}).reset_index()

# Paso 2: Obtener el 'Identifier', 'subarachnoid' y 'group' correspondiente al valor máximo
result_df = df.merge(max_values, on=['Group', 'subarachnoid'], how='inner')
result_df = result_df[['Identifier', 'subarachnoid', 'Group']]

# Mostrar el DataFrame resultante
print(result_df)

# Inicializar una lista para guardar los resultados finales
resultados_finales = []

# Iterar sobre cada fila en result_df
for _, row in result_df.iterrows():
    identifier = row['Identifier']
    group = row['Group']
    max_subarachnoid = row['subarachnoid']

    # Encontrar la fila correspondiente en el DataFrame original
    fila = df[df['Identifier'] == identifier].index.tolist()[0]

    # Encontrar las filas adyacentes dentro del mismo grupo
    start_index = max(0, fila - 6)  # Mínimo 6 filas antes
    end_index = min(len(df), fila + 7)  # Máximo 6 filas después
    group_df = df[(df['Group'] == group) & (df.index >= start_index) & (df.index < end_index)]

    # Agregar estas filas al resultado final
    resultados_finales.append(group_df)

# Concatenar todos los resultados en un nuevo DataFrame
resultado_final_df = pd.concat(resultados_finales)

# Guardar el resultado en un nuevo archivo Excel
archivo_nuevo_excel = 'choosed_slices.xlsx'
nombre_hoja_nueva = 'results'
resultado_final_df.to_excel(archivo_nuevo_excel, sheet_name=nombre_hoja_nueva, index=False)

# Mostrar mensaje de finalización
print("Process complete - Results are saved in choosed_slices", archivo_nuevo_excel)


