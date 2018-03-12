"""Implements an Estimator to run Keras model in Google Cloud ML
See https://cloud.google.com/blog/big-data/2017/12/new-in-tensorflow-14-converting-a-keras-model-to-a-tensorflow-estimator
"""

import tensorflow as tf
import numpy as np
from predict import load_images
from model import INPUT_SHAPE
from tensorflow.python.estimator.export.export import build_raw_serving_input_receiver_fn

# Input tensor name matching our input layer
# @todo give this a custom name rather than having Keras name this
INPUT_TENSOR_NAME = "input_1"

def convert_to_estimator(model_file):
    """Convert saved Keras model to TF Estimator
    """
    estimator_model = tf.keras.estimator.model_to_estimator(keras_model_path=model_file)
    return estimator_model

def input_function(features, labels=None, shuffle=False):
    """Returns an input function that can be used for prediction within an estimator"""
    # @todo rename input_1
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={INPUT_TENSOR_NAME: features},
        y=labels,
        shuffle=shuffle
    )
    return input_fn

def serving_input_function():
    """Returns a serving input function used for exporting a model"""
    feature_spec = {INPUT_TENSOR_NAME: tf.FixedLenFeature(dtype=tf.float32, shape=INPUT_SHAPE)}
    return tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)()

def infer(argv=None):
    """Run the inference and print the results to stdout."""
    # Initialize the estimator and run the prediction
    estimator = convert_to_estimator('svhn_model.h5')

    examples = load_images()
    # examples = examples[:1]

    predictions = list(estimator.predict(input_fn=input_function(examples)))
    for p in predictions:
        predicted_label = np.argmax(p.values(), axis=-1)
        print predicted_label

def export():
    """Export estimator for use in Google Cloud ML"""
    estimator = convert_to_estimator('svhn_model.h5')
    export = estimator.export_savedmodel(export_dir_base='.', serving_input_receiver_fn=serving_input_function)
    print export

## Run script
if __name__ == "__main__":
    # infer()
    export()
