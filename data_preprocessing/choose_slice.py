import pandas as pd

# 1. Take the excel with identifier for each images and probability of hemorrage
excel_path = '/export/usuarios01/lmurat/Datos/Predictions-SAH/data_preprocessing/excel_new_data_prepared.xlsx'
df = pd.read_excel(excel_path)

# 2. Add a column with patient ID (HSA 1, HSA 2, ...)
df['Patient ID'] = df['Identifier'].apply(lambda x: x.split('-')[0])

# 3. Select 13 rows per patient
selected_rows = []

for id, group_df in df.groupby('Patient ID'):
    center_idx = len(group_df) // 2
    # center_idx = group_df['subarachoid'].idxmax() # to not take the middle index but the one with max probability of hemorrage
    start_idx = max(center_idx - 6, 0)
    end_idx = start_idx + 13
    selected = group_df.iloc[start_idx:end_idx]
    selected_rows.append(selected)

df_new_data_prepared = pd.concat(selected_rows).reset_index(drop=True)

# 4. Save into a new sheet in the excel
df_new_data_prepared = df_new_data_prepared.drop(['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural'], axis = 1)
with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    df_new_data_prepared.to_excel(writer, sheet_name='selected_slices', index=False)
print(df_new_data_prepared)




# # ____________________________

# # 3. Take the value of the middle slice (or slice with the maximum subarachnoid probability) per group (per patient)
# max_values = df.groupby('Group').agg({'subarachnoid': 'max'}).reset_index() #max


# # Paso 2: Obtener el 'Identifier', 'subarachnoid' y 'group' correspondiente al valor máximo
# result_df = df.merge(max_values, on=['Group', 'subarachnoid'], how='inner')
# result_df = result_df[['Identifier', 'subarachnoid', 'Group']]

# # Inicializar una lista para guardar los resultados finales
# resultados_finales = []

# # Iterar sobre cada fila en result_df
# for _, row in result_df.iterrows():
#     identifier = row['Identifier']
#     group = row['Group']
#     max_subarachnoid = row['subarachnoid']

#     # Encontrar la fila correspondiente en el DataFrame original
#     fila = df[df['Identifier'] == identifier].index.tolist()[0]

#     # Encontrar las filas adyacentes dentro del mismo grupo
#     start_index = max(0, fila - 6)  # Mínimo 6 filas antes
#     end_index = min(len(df), fila + 7)  # Máximo 6 filas después
#     group_df = df[(df['Group'] == group) & (df.index >= start_index) & (df.index < end_index)]

#     # Agregar estas filas al resultado final
#     resultados_finales.append(group_df)

# # Concatenar todos los resultados en un nuevo DataFrame
# resultado_final_df = pd.concat(resultados_finales)

# # Guardar el resultado en un nuevo archivo Excel
# archivo_nuevo_excel = '6cortesenteros.xlsx'
# nombre_hoja_nueva = 'resultados'
# resultado_final_df.to_excel(archivo_nuevo_excel, sheet_name=nombre_hoja_nueva, index=False)

# # Mostrar mensaje de finalización
# print("Proceso completado. Los resultados han sido guardados en", archivo_nuevo_excel)