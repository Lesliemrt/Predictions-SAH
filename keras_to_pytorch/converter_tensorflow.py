# To use with environment with tensorflow keras
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras import backend as K

from mmdnn.conversion._script.convertToIR import convertToIR


# 1. Create keras model and load weights
HEIGHT, WIDTH, CHANNELS = 256, 256, 3
SHAPE = (HEIGHT, WIDTH, CHANNELS)
def create_model():
    K.clear_session()
    
    base_model =  DenseNet169(weights = 'imagenet', include_top = False, input_shape = SHAPE)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.15)(x)
    y_pred = Dense(6, activation = 'sigmoid')(x)

    return Model(inputs = base_model.input, outputs = y_pred)

keras_model = create_model()
weights_path = f'/export/usuarios01/lmurat/Datos/Predictions-SAH/keras_to_pytorch/densenet169_model.h5'
keras_model.load_weights(weights_path)   

# # 2.1 Saving keras model weights layer per layer
# keras_weights = {}
# for layer in keras_model.layers:
#     if layer.get_weights():  # Ignore layers without weights
#         keras_weights[layer.name] = layer.get_weights()

# path = '/export/usuarios01/lmurat/Datos/Predictions-SAH/keras_to_pytorch/keras_weights.npy'
# np.save(path, keras_weights)

# 2.2 Oteher method : Saving keras model using mmdnn
keras_model.save('/export/usuarios01/lmurat/Datos/Predictions-SAH/keras_to_pytorch/densenet169_fullmodel.h5')
model_path = '/export/usuarios01/lmurat/Datos/Predictions-SAH/keras_to_pytorch/densenet169_fullmodel.h5'
destination_name = 'densenet169_IR'

convertToIR._main([
    '-f', 'keras', # framework source
    '-d', destination_name, # destination name
    '-n', model_path # .h5 file (structure + weights)
])