"""Implements an Estimator to run Keras model in Google Cloud ML
"""

import tensorflow as tf
import skimage.io
import numpy as np
import keras
from svhn_data import load_data
from predict import load_images


def convert_to_estimator(model_file='svhn_model.h5'):
    """Convert saved Keras model to TF Estimator
    """
    # model = keras.models.load_model(model_file)
    # print model.input_names[0]
    estimator_model = tf.keras.estimator.model_to_estimator(keras_model_path=model_file)
    return estimator_model

# def test_inputs():
#     """Returns training set as Operations.
#     Returns:
#         (features, ) Operations that iterate over the test set.
#     """
#     with tf.name_scope('Test_data'):
#         images = tf.constant(load_images(), dtype=np.float32)
#         dataset = tf.data.Dataset.from_tensor_slices((images,))
#         # Return as iteration in batches of 1
#         # f = dataset.batch(1).make_one_shot_iterator().get_next()
#         # print f
#         features = dataset.batch(1).make_one_shot_iterator().get_next()
#         print features
#         return {'input_1': features}
#         # return tf.estimator.inputs.numpy_input_fn(
#         #     x={'input_1': features},
#         #     num_epochs=1,
#         #     shuffle=False);

def input_function(features, labels=None, shuffle=False):

    print features[0].dtype
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"input_1": features},
        y=labels,
        shuffle=shuffle
    )
    return input_fn

def infer(argv=None):
    """Run the inference and print the results to stdout."""
    # Initialize the estimator and run the prediction
    estimator = convert_to_estimator()
    print estimator

    result = list(estimator.predict(input_fn=input_function(load_images())))
    # result = estimator.evaluate(input_fn=test_inputs)
    # print result
    # for r in result:
    #     print "Z"
    #     # print(r)

# def input_function(features,labels=None,shuffle=False):
#     input_fn = tf.estimator.inputs.numpy_input_fn(
#         x={"input_1": features},
#         y=labels,
#         shuffle=shuffle
#     )
#     return input_fn


## Load our data
# (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_data("full", verbose=1)
# print x_test[0]

# estimator_model = convert_to_estimator()
# estimator_model.train(input_fn=input_function(x_train, y_train, True))

## Run script
if __name__ == "__main__":
    infer()



