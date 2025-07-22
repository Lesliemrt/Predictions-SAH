# Installation
environment.yml (conda)  
Pytorch, torchvision, numpy, matplotlib, os, pandas, iterstrat.ml_stratifiers (pip install iterative-stratification), pydicom, gdcm (pip install python-gdcm), opencv, openpyxl
installation of pytorch first for dependencies  
  
# Files
```bash
├── README.md
├── requirement.txt
├── configs.py : select target_output and num_classes
├── utils.py
├── dataloader.py
├── model.py : get_model function, classifier class, model class
├── train.py : Model_extended class with trainloop
├── main.py : to train, get results, loss graph, accuracy, saliency maps, reliability diagramms, auc roc curves
├── main_iterations.py : to train x times, get the 5 best models based on validation auc scores and take the mean/ max predictions of the 5 best models
├── RESULTS : excel file with all the plots
├── data/
│   ├── hospital_data_1/ : 1st cohort of data
│   ├── hospital_data_2/ : 2nd cohort of data (used for tests)
│   ├── excel_predicciones.xlsx : labels for 1st cohort of data
│   └── exp16_seres_ep5.pth : weights of pretrained model (from https://github.com/okotaku/kaggle_rsna2019_3rd_solution)
├── data_preprocessing/ : only for 2nd cohort of data
│   ├── prepare_new_data.py
│   ├── predictions.py : with pretrained model : predict proba of hemorrage
│   ├── choose_slice.py : 13 slices per patient based on max probability of subacharoid hemorrage
│   ├── excel_new_data_prepared.xlsx : results of prepare_new_data.py, predictions.py and choose_slice.py
│   └── multiclasses.py : histograms for multiclass outputs
├── checkpoints/
├── results/ : automatics plots and images
├── saliency_maps_exemples/ : saliency maps from a model with good performances train on the output "Exitus"
└── keras_to_pytorch/ : used to transpose .h5 files to .pth (not used anymore)
```

# References
