"""implements the Convoluted NN from model.py using Keras Functional API.
"""

import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Model
from notMNIST import load_data, IMAGE_ROWS, IMAGE_COLS, NUM_LABELS, NUM_CHANNELS

# Training settings
BATCH_SIZE = 128
EPOCHS = 10

# Load our data
(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_data(verbose=1)
input_shape = (IMAGE_ROWS, IMAGE_COLS, NUM_CHANNELS)


def create_network(img_input):
    x = Conv2D(64, kernel_size=(3,3), strides=(1,1), input_shape=input_shape, activation='relu', padding='same')(img_input)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(21, kernel_size=(3,3), strides=(1,1), input_shape=input_shape, activation='relu', padding='same')(img_input)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    return Dense(NUM_LABELS, activation='softmax')(x)

img_input = Input(shape=input_shape)
network = create_network(img_input)
model = Model(img_input, network)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=1,
          validation_data=(x_valid, y_valid))
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
