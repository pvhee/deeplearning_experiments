"""implements the Convoluted NN from model.py using Keras Sequential API.
"""

import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Sequential
from notMNIST import load_data, IMAGE_ROWS, IMAGE_COLS, NUM_LABELS, NUM_CHANNELS

# Training settings
BATCH_SIZE = 128
EPOCHS = 1

# Load our data
(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_data()
input_shape = (IMAGE_ROWS, IMAGE_COLS, NUM_CHANNELS)

# Set up our model
model = Sequential()

# Conv network with 24 depth and 4x4 convolutions
model.add(Conv2D(32, kernel_size=(4,4), strides=(1,1), input_shape=input_shape, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, kernel_size=(4,4), strides=(1,1), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
# Let's reshape to our conv layer output fits our fully connected layer input
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.8))
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
