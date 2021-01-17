import logging
import mtcnn
from utils.load_models import *

logging.basicConfig(level=logging.DEBUG)

model = load_keras_model('models/facenet_keras.h5')

