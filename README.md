# Installation
environment.yml (conda)  
Pytorch, torchvision, numpy, matplotlib, os, pandas, iterstrat.ml_stratifiers (pip install iterative-stratification), pydicom, gdcm (pip install python-gdcm), opencv, openpyxl
installation of pytorch first for dependencies  
  
# Files
Project  
|-hospital_data_1  
|-excel_predicciones.xlsx --> "selected_cortes" sheet for labels    
|-main.py --> import model or model_many_layers  
|-main_iterations --> aucroc curve for different seed  
|-configs.py (constants + directory)  
|-utils.py  
|-dataloader.py  
|-model.py  
|-model_many_layers.py  
|-train.py  
|-environment.yml  
  
Test (data from hospital 1, Sofia's code) --> Tensorflow  
|-hospital data 1  
|-todaspredicciones.py  
|-densenet169_model.h5 --> pretrained model weights  
|-cortes6.py --> to create choosed_slices  
|-choosed_slices.xlsx  
  
07/05/25  

