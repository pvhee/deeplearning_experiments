"""implements a Convoluted NN using TensorFlow Core APIs from Udacity's tutorial.
   It is meant as an exercise to package up TF code, run and deploy it on Google ML Cloud.
   To understand the TF code, refer to 4_convolutions.ipynb.
"""

from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

IMAGE_SIZE = 28
NUM_LABELS = 10
NUM_CHANNELS = 1  # grayscale

PATCH_SIZE = 4
DEPTH = 24
NUM_HIDDEN = 256
NUM_HIDDEN2 = 128

BATCH_SIZE = 32

TRAIN, EVAL, PREDICT = 'TRAIN', 'EVAL', 'PREDICT'


def model_fn(mode):
    """Create a Recurrent DNN (from Problem 2 in Assignment 4 - Convolutions)

    Args:
      mode (string): Mode running training, evaluation or prediction
      features (dict): Dictionary of input feature Tensors
      labels (Tensor): Class label Tensor
      hidden_units (list): Hidden units
      learning_rate (float): Learning rate for the SGD

    Returns:
      Depending on the mode returns Tuple or Dict
    """
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = input_fn()

    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    tf_train_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal(
        [PATCH_SIZE, PATCH_SIZE, NUM_CHANNELS, DEPTH], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([DEPTH]))
    layer2_weights = tf.Variable(tf.truncated_normal(
        [PATCH_SIZE, PATCH_SIZE, DEPTH, DEPTH], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[DEPTH]))
    layer3_weights = tf.Variable(tf.truncated_normal(
        [IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * DEPTH, NUM_HIDDEN], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[NUM_HIDDEN]))
    layer4_weights = tf.Variable(tf.truncated_normal(
        [NUM_HIDDEN, NUM_LABELS], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]))

    # Model.
    def model(data):
        # conv2d computes a a 2D convolution given
        # - a 4D input tensor (our data)
        # - a 4D filter tensor (our weights)
        # - strides: 1D tensor of length 4
        # - padding
        # This is flattened to a 2D matrix of shape
        # [filter_height * filter_width * in_channels, output_channels]
        conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        return tf.matmul(hidden, layer4_weights) + layer4_biases

    # Training computation.
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

    if mode in (TRAIN, EVAL):
        # global_step is necessary in eval to correctly load the step
        # of the checkpoint we are evaluating
        global_step = tf.contrib.framework.get_or_create_global_step()

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss, global_step=global_step)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))

    if mode == TRAIN:
        return tf_train_dataset, tf_train_labels, optimizer, loss, train_prediction, valid_prediction, test_prediction


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


def reformat(dataset, labels):
    dataset = dataset.reshape(
        (-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)).astype(np.float32)
    labels = (np.arange(NUM_LABELS) == labels[:, None]).astype(np.float32)
    return dataset, labels
