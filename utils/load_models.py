from keras.models import load_model
import logging

def load_keras_model(model):
    logging.info("Loading keras model")
    lm = load_model(model)
    return lm
