import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import configs
import utils

image_path = f"{configs.DATA_DIR}OneDrive_1_02_06_2025/HSA 266/10003A4B/10003A4C/10003A4D/10003A6E"
image = utils._read(image_path)
image = image.permute(1, 2, 0).cpu().numpy()
plt.axis('off')
plt.imshow(image)
plt.savefig(f"{configs.DATA_DIR}/results/visualize new data test.png") 
plt.close()

# (0028,1050) Window Center                       DS: '40'
# (0028,1051) Window Width                        DS: '90'
# (0028,1052) Rescale Intercept                   DS: '0'
# (0028,1053) Rescale Slope                       DS: '1'