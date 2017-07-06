"""Converts not MNIST data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import cPickle as pickle
from six.moves import range

import numpy as np
import argparse
import os
import sys

import tensorflow as tf

FLAGS = None

IMAGE_SIZE = 28
NUM_CHANNELS = 1  # grayscale

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(images, labels, directory, name):
    """Converts a dataset to tfrecords."""
    num_examples = images.shape[0]
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    filename = os.path.join(directory, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()

def reformat(dataset):
    dataset = dataset.reshape(
        (-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)).astype(np.float32)
    return dataset

def read_pickle(filename):
    with open(filename, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save  # hint to help gc free up memory

        train_dataset = reformat(train_dataset)
        valid_dataset = reformat(valid_dataset)
        test_dataset = reformat(test_dataset)
    return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels

def main(unused_argv):
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = read_pickle(FLAGS.input_file)

    # Convert to Examples and write the result to TFRecords.
    convert_to(train_dataset, train_labels, FLAGS.directory, 'train')
    convert_to(valid_dataset, valid_labels, FLAGS.directory, 'validation')
    convert_to(test_dataset, test_labels, FLAGS.directory, 'test')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-file',
        type=str,
        default='notMNIST.pickle',
        help='Pickle file with not MNIST data'
    )
    parser.add_argument(
        '--directory',
        type=str,
        default='/tmp/data/',
        help='Directory to write the converted result'
    )
    # parser.add_argument(
    #     '--validation_size',
    #     type=int,
    #     default=5000,
    #     help="""\
    #     Number of examples to separate from the training data for the validation
    #     set.\
    #     """
    # )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
