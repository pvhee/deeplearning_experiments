"""Implements an Estimator to run Keras model in Google Cloud ML
"""

import tensorflow as tf
import skimage.io
import numpy as np
import keras
from svhn_data import load_data
from keras.datasets import cifar10


# MNIST sample images
IMAGE_URLS = [
    'data_test/34.jpg',  # 0
    'data_test/1984_failed.jpg',  # 0
    'data_test/1984.jpg',  # 0
    'data_test/170.jpg',  # 0
    'data_test/10.jpg',  # 0
    # 'https://i.imgur.com/SdYYBDt.png',  # 0
    # 'https://i.imgur.com/Wy7mad6.png',  # 1
    # 'https://i.imgur.com/nhBZndj.png',  # 2
    # 'https://i.imgur.com/V6XeoWZ.png',  # 3
    # 'https://i.imgur.com/EdxBM1B.png',  # 4
    # 'https://i.imgur.com/zWSDIuV.png',  # 5
    # 'https://i.imgur.com/Y28rZho.png',  # 6
    # 'https://i.imgur.com/6qsCz2W.png',  # 7
    # 'https://i.imgur.com/BVorzCP.png',  # 8
    # 'https://i.imgur.com/vt5Edjb.png',  # 9
]

def convert_to_estimator(model_file='svhn_model.h5'):
    estimator_model = tf.keras.estimator.model_to_estimator(keras_model_path=model_file)
    return estimator_model

def input_function(features,labels=None,shuffle=False):
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"input_1": features},
        y=labels,
        shuffle=shuffle
    )
    return input_fn

def load_images():
    """Load sample images from the web and return them in an array.
    Returns:
        Numpy array of size (1, 64, 64, 3) with sample images.
    """
    images = np.zeros((len(IMAGE_URLS), 64, 64, 3))
    # images = np.zeros((1, 28, 28, 1))
    for idx, url in enumerate(IMAGE_URLS):
        img = skimage.io.imread(url)
        img_normalized = tf.Session().run(tf.image.per_image_standardization(img))
        images[idx, :, :, :] = img_normalized
    return images


def predict(images, model_file):
    model = keras.models.load_model(model_file)
    model.summary()
    # random_batch = examples[np.random.choice(examples.shape[0], size=num_examples, replace=False), :]
    probas = model.predict(images, verbose=1)
    probas = probas.argmax(axis=-1)
    # print(random_batch)
    print(images.shape)
    print(probas)
    # visualize_batch(random_batch, probas)


## Load our data
# (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_data("full", verbose=1)
# print x_test[0]

# estimator_model = convert_to_estimator()
# estimator_model.train(input_fn=input_function(x_train, y_train, True))


x_wild = load_images()
predict(x_wild, 'svhn_model.11.h5')
