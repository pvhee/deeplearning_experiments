"""implements a Convoluted NN to recognise number sequences from the SVHN dataset
"""

import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation, Reshape, BatchNormalization
from keras.models import Model
from keras.models import load_model
from svhn_data import load_data, visualize_batch, visualize
import numpy as np
import os
import random

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Training settings
BATCH_SIZE = 128
EPOCHS = 10

# Various data properties, probably we should extract them from our data source
NUM_LABELS = 11
# Our first number in the sequence is the length of the number, followed by 5 numbers 0-10 (with 10 meaning N/A)
SEQUENCE_LENGTH = 6
INPUT_SHAPE = (64, 64, 3)

# Saved model, turn the flag on or off to do evaluating or training
VERSION = 'v2'
MODEL_FILE = 'svhn_model.' + VERSION + '.h5'
LOAD_MODEL_FLAG = 0

def create_conv(x, filters, input_shape=False):
    kwargs = {}
    if input_shape:
        kwargs['input_shape'] = input_shape
    x = Conv2D(filters, kernel_size=5, strides=1, activation='relu', padding='same', **kwargs)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=2)(x)
    # Apply dropout only during training
    if not LOAD_MODEL_FLAG:
        x = Dropout(0.5)(x)
    return x

def create_network(img_input):
    x = create_conv(img_input, 16, input_shape=INPUT_SHAPE)
    x = create_conv(x, 32)
    x = create_conv(x, 64)
    x = create_conv(x, 128)
    x = create_conv(x, 256)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    # x = Flatten()(x)
    x = Dense(6*11)(x)
    # We need to reshape our output layer so we can apply a softmax to each number independently.
    # This adds an extra dimension for the sequence length
    x = Reshape((6, 11))(x)
    return Activation('softmax', name='labels')(x)

def predict(model_file, x_test):
    model = load_model(model_file)
    model.summary()
    examples = x_test
    num_examples = 8
    random_batch = examples[np.random.choice(examples.shape[0], size=num_examples, replace=False), :]
    probas = model.predict(random_batch, verbose=1)
    probas = probas.argmax(axis=-1)
    # print(random_batch)
    # print(random_batch.shape)
    # print(probas)
    visualize_batch(random_batch, probas)

def train(model_file, x_train, y_train, x_valid, y_valid, x_test, y_test):
    # Note the name of our input layer needs to follow the naming convention *_bytes
    # so we can map base64 encoded images on this input layer
    img_input = Input(shape=INPUT_SHAPE, name="image")
    network = create_network(img_input)
    model = Model(img_input, network)

    model.summary()

    ## Train the network
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              verbose=1,
              validation_data=(x_valid, y_valid))

    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save(model_file)

# # Load some data to test on
# test_data, test_labels = load_svhn_data("test", "full")
# print(test_data.shape)
# print(test_labels.shape)
# img = test_data[19]
# label = test_labels[19]

## Run script
if __name__ == "__main__":
    # Load our data
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_data("full", verbose=1)

    if (LOAD_MODEL_FLAG):
        print("Evaluating previously saved model")
        predict(MODEL_FILE, x_test)
    else:
        print("Training new model")
        train(MODEL_FILE, x_train, y_train, x_valid, y_valid, x_test, y_test)
