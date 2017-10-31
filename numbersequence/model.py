"""implements a Convoluted NN to recognise number sequences
"""

import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation, Reshape, BatchNormalization
from keras.models import Model
from keras.models import load_model
from notmnist_sequence import load_data_or_generate, SEQUENCE_LENGTH
from notmnist import notMNIST
import numpy as np
import os
import random

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Training settings
BATCH_SIZE = 16
EPOCHS = 1

# Saved model, turn the flag on or off to do evaluating or training
MODEL_FILE = 'numbersequence_model.5.h5'
LOAD_MODEL_FLAG = 0

# Load our data
(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_data_or_generate(verbose=1, pickle_seq_file='numbersequence/notMNIST.sequence.pickle')
input_shape = (notMNIST.IMAGE_ROWS, notMNIST.IMAGE_COLS * SEQUENCE_LENGTH, notMNIST.NUM_CHANNELS)

def create_conv(x, filters, input_shape=False):
    kwargs = {}
    if input_shape:
        kwargs['input_shape'] = input_shape
    x = Conv2D(filters, kernel_size=3, strides=1, activation='relu', padding='same', **kwargs)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=2)(x)
    # x = Dropout(0.7)(x)
    return x

def create_network(img_input):
    x = create_conv(img_input, 16, input_shape=input_shape)
    x = create_conv(x, 32)
    x = create_conv(x, 64)
    x = create_conv(x, 128)
    # x = create_conv(x, 256)
    # x = create_conv(x, 192)
    # x = create_conv(x, 192)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(SEQUENCE_LENGTH * notMNIST.NUM_LABELS)(x)
    # We need to reshape our output layer so we can apply a softmax to each number independently.
    # This adds an extra dimension for the sequence length
    x = Reshape((SEQUENCE_LENGTH, notMNIST.NUM_LABELS))(x)
    return Activation('softmax')(x)

if(LOAD_MODEL_FLAG):
    model = load_model(MODEL_FILE)
    model.summary()
    examples = x_test
    num_examples = 8
    random_batch = examples[np.random.choice(examples.shape[0], size=num_examples, replace=False), :]
    probas = model.predict(random_batch, verbose=1)
    probas = probas.argmax(axis=-1)
    notMNIST.visualize_batch(random_batch, probas)

else:
    img_input = Input(shape=input_shape)
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

    model.save(MODEL_FILE)

    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])