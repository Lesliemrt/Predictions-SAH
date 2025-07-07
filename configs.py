# Constants
TEST_SIZE = 0.3
CHANNELS = 3
HEIGHT = 256
WIDTH = 256
SEED = 12345 #for reproductability

split_train = 0.6
split_valid = 0.3
split_test = 0.1

TRAIN_BATCH_SIZE = 32 
VALID_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32

patient = 9 #index of {patiente} in the path (hospital_data_1)
patient_data2 = 8

# Folders
DATA_DIR = '/export/usuarios01/lmurat/Datos/Predictions-SAH/Data/'
DIR = '/export/usuarios01/lmurat/Datos/Predictions-SAH/'

# Output : 
# 0-1 : 'Rebleeding, 'VasoespasmoA', 'ANY Vasoespasm ', 'Hydrocephalus', 'Infarction', 'Exitus', 'Epileptic seizure'
# 0-6 : 'mRSalta', 'mRS1año'
# 0-38+ : 'DiasVM'
# 0-108+ : 'DiasUCI'
target_output = 'mRSalta'

# Ro run on gpu if available
import torch
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") #to run on gpu if available




# only for test with RSNA images
TEST_IMAGES_DIR = DATA_DIR + 'stage_2_test/'
TRAIN_IMAGES_DIR = DATA_DIR + 'stage_2_train/'