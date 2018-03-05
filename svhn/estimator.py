"""Implements an Estimator to run Keras model in Google Cloud ML
"""

import keras
import tensorflow as tf
from keras.models import load_model

def convert_to_estimator(model_file='svhn_model.h5'):
    estimator_model = tf.keras.estimator.model_to_estimator(keras_model_path=model_file)
    print estimator_model



convert_to_estimator()


