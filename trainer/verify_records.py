"""Verify a given TFRecords file."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import os
import sys
import math

import matplotlib.pyplot as plt
import tensorflow as tf

FLAGS = None

IMAGE_SIZE = 28

PRETTY_LABELS = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J'}

# Set warning level lower
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def plot(images, labels):
    plt.figure()
    n = math.sqrt(images.shape[0])
    width = images.shape[1]
    height = images.shape[2]
    for i, image in enumerate(images):
        image_reshaped = np.reshape(image, [width, height])
        plt.subplot(n, n, i+1)
        plt.axis('off')
        plt.title(PRETTY_LABELS[labels[i]])
        plt.imshow(image_reshaped)
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.8, wspace=0.2)
    plt.show()

def main(unused_argv):
    batch_size = FLAGS.number
    images, labels = inputs(FLAGS.read_file, batch_size)

    # We need to start a session, to see something!
    sess = tf.Session()

    # Required. See below for explanation
    init = tf.global_variables_initializer()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    # Let's plot some examples
    label_val, image_val = sess.run([labels, images])

    # Show all figures in a grid
    plot(image_val, label_val)

def read_from_tfrecord(filename_queue):
    # Get a reader
    reader = tf.TFRecordReader()

    # Read in a single example
    _, tfrecord_serialized = reader.read(filename_queue)

    # label and image are stored as bytes but could be stored as
    # int64 or float64 values in a serialized tf.Example protobuf.
    features = tf.parse_single_example(tfrecord_serialized,
                                                features={
                                                    'label': tf.FixedLenFeature([], tf.int64),
                                                    'image_raw': tf.FixedLenFeature([], tf.string),
                                                }, name='features')

    # Decode from a scalar string tensor into a flattened float32 vector
    image = tf.decode_raw(features['image_raw'], tf.float32)

    # We need to reshape this tensor, for TF to know about it in later ops
    # See also https://stackoverflow.com/questions/35691102/valueerror-all-shapes-must-be-fully-defined-issue-due-to-commenting-out-tf-ran
    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 1])

    # OPTIONAL: Could reshape into a 28x28 image and apply distortions
    # here.  Since we are not applying any distortions in this
    # example, and the next step expects the image to be flattened
    # into a vector, we don't bother.

    # If you wish to apply distortions, have a look at these examples:
    # https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_input.py

    label = features['label']
    return label, image

def inputs(filename, batch_size):
    # Create a queue that produces the filenames to read.
    # This allows you to break up the the dataset in multiple files to keep size down
    # Here, though, we only have one file, so let's wrap into a list
    filename_queue = tf.train.string_input_producer([filename])

    # Read examples from files in the filename queue.
    label, image = read_from_tfrecord(filename_queue)

    num_preprocess_threads = 16
    min_queue_examples = 10

    # Generate a batch of images and labels by building up a random queue
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)

    return images, tf.reshape(label_batch, [batch_size])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--read-file',
        type=str,
        required=True,
        help='Verify a given tfrecords file'
    )
    parser.add_argument(
        '--number',
        type=int,
        default=64,
        help="Number of (random)examples to print out"
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
