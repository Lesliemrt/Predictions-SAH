# Installation
environment.yml with conda to replicate uc3m_env
requirements for core project: Pytorch, torchvision, numpy, matplotlib, os, pandas, iterstrat.ml_stratifiers (pip install iterative-stratification), pydicom, gdcm (pip install python-gdcm), opencv, openpyxl
install pytorch first for dependencies

# Files
Pre-project (data from RSNA 2019) --> uc3m_env
|-stage_2_test
|-stage_2_train
|-stage_2_sample_submission
|-stage_2_train.csv
|-pytorch_prepared_model_(densenet169)leslie
|-dataloader_RSNA

Project --> uc3m_env
|-hospital data 1
  |-raw data
|-excel_predicciones.xlsx
|-main.py --> import model or model_many_layers
|-main_40_iterations
|-configs.py (constants + directory)
|-utils.py
|-dataloader.py
|-model.py
|-model_many_layers.py
|-train.py
|-environment.yml

Test (data from hospital 1, Sofia's code) --> Tensorflow --> uc3m_tf_env
|-hospital data 1 --> the data
|-todaspredicciones.py
|-densenet169_model.h5 --> pretrained model weights
|-cortes6.py --> to create choosed_slices
|-choosed_slices.xlsx

05/05

