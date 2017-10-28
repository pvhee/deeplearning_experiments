"""Converts notMNIST data into sequences of letters, saved as TFRecords"""
import convert_to_records
import verify_records
import argparse
import sys
import tensorflow as tf
import numpy as np

from sklearn.utils import shuffle

import notMNIST

def create_sequence(data, labels, limit=0):
    i = 0
    j = 0

    data, labels = shuffle(data, labels)

    # If no limit is given, then give back a data set equal in size as the given data set
    if limit == 0:
        limit = len(data)

    sequence = []
    sequence_labels = []

    while i<limit:
        # check whether we need to restart our loop on a reshuffled data object
        if j + notMNIST.SEQUENCE_LENGTH >= len(data):
            np.random.shuffle(data)
            j=0

        # Get a slice into data capturing SEQUENCE_LENGTH images, then concatenate them horizontally
        # Todo: should we add in a separator?
        # Todo we need to add in empty characters too!
        data_slice = data[j:j+notMNIST.SEQUENCE_LENGTH,]
        labels_slice = labels[j:j+notMNIST.SEQUENCE_LENGTH,]
        img = np.concatenate(data_slice, axis=1)

        sequence.append(img)
        sequence_labels.append(labels_slice)

        # Increase our both counters
        i += 1
        j += notMNIST.SEQUENCE_LENGTH

    # Convert back to nparrays, which are easier to manage
    sequence = np.asarray(sequence)
    sequence_labels = np.asarray(sequence_labels)

    return sequence, sequence_labels


def main(unused_argv):
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = convert_to_records.read_pickle(FLAGS.input_file)

    _test_dataset, _test_labels = create_sequence(test_dataset, test_labels, 16)
    convert_to_records.convert_to(_test_dataset, _test_labels, FLAGS.directory, 'test.sequence')


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

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
