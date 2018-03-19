"""Implements an Estimator to run Keras model in Google Cloud ML
See https://cloud.google.com/blog/big-data/2017/12/new-in-tensorflow-14-converting-a-keras-model-to-a-tensorflow-estimator
"""

import tensorflow as tf
import numpy as np
from predict import load_images
from model import INPUT_SHAPE

# Input tensor name matching our input layer
# @todo give this a custom name rather than having Keras name this
INPUT_TENSOR_NAME = 'image'
EXPORT_ESTIMATOR_DIR = 'svhn/export'

def convert_to_estimator(model_file):
    """Convert saved Keras model to TF Estimator
    """
    estimator_model = tf.keras.estimator.model_to_estimator(keras_model_path=model_file)
    return estimator_model

def input_function(features, labels=None, shuffle=False):
    """Returns an input function that can be used for prediction within an estimator"""
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={INPUT_TENSOR_NAME: features},
        y=labels,
        shuffle=shuffle
    )
    return input_fn

def serving_input_function():
    """Returns a serving input function used for exporting a model"""
    # raw_byte_strings = tf.placeholder(dtype=tf.string, shape=[None], name='source')
    # decode = lambda raw_byte_str: tf.decode_raw(raw_byte_str, tf.float32)
    # input_images = tf.map_fn(decode, raw_byte_strings, dtype=tf.float32)
    feature_spec = {INPUT_TENSOR_NAME: tf.placeholder(dtype=tf.float32, shape=(1, 64, 64, 3))}
    # feature_spec = {'image_bytes': input_images}
    return tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)()

def infer(model_file):
    """Run the inference and print the results to stdout."""
    # Initialize the estimator and run the prediction
    estimator = convert_to_estimator(model_file)

    examples = load_images()
    # examples = examples[:1]

    predictions = list(estimator.predict(input_fn=input_function(examples)))
    for p in predictions:
        predicted_label = np.argmax(p.values(), axis=-1)
        print predicted_label

def export(model_file):
    """Export estimator for use in Google Cloud ML"""
    estimator = convert_to_estimator(model_file)
    export = estimator.export_savedmodel(export_dir_base=EXPORT_ESTIMATOR_DIR, serving_input_receiver_fn=serving_input_function)
    print export

## Run script
if __name__ == "__main__":
    # infer('svhn_model.h5')
    # infer('svhn_model.0.3.h5')
    # export('svhn_model.h5')
    export('svhn_model.v2.h5')
