import os
from tensorflow.keras.models import load_model
from model import model

model.save(os.path.join('models', 'happysadmodel.h5'))