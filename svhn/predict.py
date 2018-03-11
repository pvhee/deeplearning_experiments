"""Implements an Estimator to run Keras model in Google Cloud ML
"""

import tensorflow as tf
import skimage.io
import numpy as np
import keras

# MNIST sample images
IMAGE_URLS = [
    'data_test/34.jpg',  # 34
    'data_test/1984_failed.jpg',  # 1984 (drawn)
    'data_test/1984.jpg',  # 1984 (drawn again)
    'data_test/170.jpg',  # 170
    'data_test/10.jpg',  # 10
]

def load_images():
    """Load sample images from the web and return them in an array.
    Returns:
        Numpy array with sample images.
    """
    images = np.zeros((len(IMAGE_URLS), 64, 64, 3), dtype=np.float32)
    for idx, url in enumerate(IMAGE_URLS):
        img = skimage.io.imread(url)
        img_normalized = tf.Session().run(tf.cast(tf.image.per_image_standardization(img), tf.float32))
        images[idx, :, :, :] = img_normalized
    return images

def predict(images, model_file):
    model = keras.models.load_model(model_file)
    model.summary()
    probas = model.predict(images, verbose=1)
    probas = probas.argmax(axis=-1)
    print(images.shape)
    print(probas)

## Run script
if __name__ == "__main__":
    x_wild = load_images()
    predict(x_wild, 'svhn_model.h5')
