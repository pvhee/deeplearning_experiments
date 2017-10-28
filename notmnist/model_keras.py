"""implements the Convoluted NN from model.py using Keras Functional API.
"""

import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Model
from keras.models import load_model
from notMNIST import load_data, IMAGE_ROWS, IMAGE_COLS, NUM_LABELS, NUM_CHANNELS, pretty_labeler
import matplotlib.pyplot as plt
import numpy as np
import os
import random

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Training settings
BATCH_SIZE = 128
EPOCHS = 1

# Saved model, turn the flag on or off to do evaluating or training
MODEL_FILE = 'notmnist_model.h5'
LOAD_MODEL_FLAG = 1

# Load our data
(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_data(verbose=1)
input_shape = (IMAGE_ROWS, IMAGE_COLS, NUM_CHANNELS)

def visualize(x):
    '''Visualize a number from our dataset'''
    # Get rid of the last dimension (our number of channels, as can't have this for the plot)
    x = np.squeeze(x, axis=2)
    plt.figure()
    plt.axis('off')
    plt.imshow(x, interpolation='none', cmap='gray')
    plt.show()

def visualize_batch(x_batch, y_batch):
    '''Visualize a batch of images'''
    plt.figure()
    plt.suptitle(np.array_str(pretty_labeler(y_batch)))
    for i, x in enumerate(x_batch):
        x = np.squeeze(x, axis=2)
        plt.subplot(2, 4, i+1)
        plt.axis('off')
        plt.imshow(x)
    plt.show()

def visualize_number(model, number):
    '''Visualize a number after passing through our model. This gives a visual into the architecture of our network '''
    # Keras expects batches of images, so we have to add a dimension to trick it into being nice
    number_batch = np.expand_dims(number,axis=0)
    conv_letter = model.predict(number_batch)
    conv_letter = np.squeeze(conv_letter, axis=0)
    print conv_letter.shape
    visualize(conv_letter)

def create_network(img_input):
    x = Conv2D(64, kernel_size=3, strides=(1,1), input_shape=input_shape, activation='relu', padding='same')(img_input)
    x = MaxPooling2D(pool_size=2)(x)
    x = Conv2D(32, kernel_size=3, strides=(1,1), input_shape=input_shape, activation='relu', padding='same')(img_input)
    x = MaxPooling2D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    return Dense(NUM_LABELS, activation='softmax')(x)

## Try visualising network parts
## Note: also uncomment the parts in create_network and return early (i.e. before we flatten)

# example = x_train[20]
# visualize(example)
# visualize_number(model, example)
# exit(1)

if(LOAD_MODEL_FLAG):
    model = load_model(MODEL_FILE)
    examples = x_test
    num_examples = 8
    random_batch = examples[np.random.choice(examples.shape[0], size=num_examples, replace=False), :]
    probas = model.predict(random_batch, verbose=1)
    probas = probas.argmax(axis=-1)
    visualize_batch(random_batch, probas)

else:
    img_input = Input(shape=input_shape)
    network = create_network(img_input)
    model = Model(img_input, network)

    ## Train the network
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
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
