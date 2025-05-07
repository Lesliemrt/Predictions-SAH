# Installation
environment.yml with conda
Pytorch, torchvision, numpy, matplotlib, os, pandas, iterstrat.ml_stratifiers (pip install iterative-stratification), pydicom, gdcm (pip install python-gdcm), opencv, openpyxl
install pytorch first for dependencies

# Files
Project 
|-hospital data 1  
|-excel_predicciones.xlsx  
|-main.py --> import model or model_many_layers  
|-main_40_iterations --> test on different seed  
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
  
05/05  

