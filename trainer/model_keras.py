"""implements the Convoluted NN from model.py using Keras Functional API.
"""

import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Model, Sequential

from task import input_fn

# Data properties
IMAGE_ROWS = 28
IMAGE_COLS = 28
NUM_LABELS = 10
NUM_CHANNELS = 1  # grayscale

# Training settings
BATCH_SIZE = 128
EPOCHS = 1

# Load our data
x_train, y_train, x_valid, y_valid, x_test, y_test = input_fn()
input_shape = (IMAGE_ROWS, IMAGE_COLS, NUM_CHANNELS)

# Set up our model
model = Sequential()

# Conv network with 24 depth and 4x4 convolutions
model.add(Conv2D(16, kernel_size=(4,4), strides=(1,1), input_shape=input_shape, activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Conv2D(48, kernel_size=(4,4), strides=(1,1), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
# Let's reshape to our conv layer output fits our fully connected layer input
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
# Our output layer
model.add(Dense(NUM_LABELS, activation='softmax'))

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





# model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(NUM_LABELS, activation='softmax'))








