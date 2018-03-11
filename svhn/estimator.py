"""Implements an Estimator to run Keras model in Google Cloud ML
See https://cloud.google.com/blog/big-data/2017/12/new-in-tensorflow-14-converting-a-keras-model-to-a-tensorflow-estimator
"""

import tensorflow as tf
import numpy as np
from predict import load_images

def convert_to_estimator(model_file):
    """Convert saved Keras model to TF Estimator
    """
    estimator_model = tf.keras.estimator.model_to_estimator(keras_model_path=model_file)
    return estimator_model

def input_function(features, labels=None, shuffle=False):
    # @todo rename input_1
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"input_1": features},
        y=labels,
        shuffle=shuffle
    )
    return input_fn

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

## Run script
if __name__ == "__main__":
    infer()
