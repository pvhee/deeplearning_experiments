"""Converts notMNIST data into sequences of letters"""
import argparse
import sys
import numpy as np
from sklearn.utils import shuffle
from notmnist import notMNIST
from six.moves import cPickle as pickle
import os

SEQUENCE_LENGTH = 5

def create_sequence(data, labels, limit=10):
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
        if j + SEQUENCE_LENGTH >= len(data):
            np.random.shuffle(data)
            j=0

        # Get a slice into data capturing SEQUENCE_LENGTH images, then concatenate them horizontally
        # Todo: should we add in a separator?
        # Todo we need to add in empty characters too!
        data_slice = data[j:j+SEQUENCE_LENGTH,]
        labels_slice = labels[j:j+SEQUENCE_LENGTH,]
        img = np.concatenate(data_slice, axis=1)

        sequence.append(img)
        sequence_labels.append(labels_slice)

        # Increase our both counters
        i += 1
        j += SEQUENCE_LENGTH

    # Convert back to nparrays, which are easier to manage
    sequence = np.asarray(sequence)
    sequence_labels = np.asarray(sequence_labels)

    return sequence, sequence_labels

def load_data_or_generate(pickle_seq_file='numbersequence/notMNIST.sequence.pickle', pickle_file='notmnist/notMNIST.pickle', verbose=0, overwrite=0):
    '''Load data or generate new data from notMNIST if data cannot be found'''
    if not os.path.exists(pickle_seq_file) or overwrite:
        print('Generating new sequence pickle from ', pickle_file, ' and saving to ', pickle_seq_file)
        (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = notMNIST.load_data(pickle_file, verbose=0)
        x2_train, y2_train = create_sequence(x_train, y_train)
        x2_valid, y2_valid = create_sequence(x_valid, y_valid)
        x2_test, y2_test = create_sequence(x_test, y_test)
        try:
            with open(pickle_seq_file, 'wb') as f:
                pickle.dump(x2_train, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', pickle_seq_file, ':', e)
    else:
        print('Loading existing sequence pickle from ', pickle_seq_file)

    if verbose:
        print('Training set', x2_train.shape, y2_train.shape)
        print('Validation set', x2_valid.shape, y2_valid.shape)
        print('Test set', x2_test.shape, y2_test.shape)


    return (x2_train, y2_train), (x2_valid, y2_valid), (x2_test, y2_test)

# print(labels)

# print(labels.argmax(axis=-1))


# Visualise some examples
# notMNIST.visualize_batch(seq[0:8:], labels[0:8:].argmax(axis=-1))


load_data_or_generate()

# Build a maybe construct to build the sequence from scratch if not found...


# def maybe_download(filename, expected_bytes):
#   """Download a file if not present, and make sure it's the right size."""
#   if not os.path.exists(filename):
#     filename, _ = urllib.request.urlretrieve(url + filename, filename)
#   statinfo = os.stat(filename)
#   if statinfo.st_size == expected_bytes:
#     print('Found and verified', filename)
#   else:
#     print(statinfo.st_size)
#     raise Exception(
#         'Failed to verify ' + filename + '. Can you get to it with a browser?')
#   return filename




# def main(unused_argv):
#     train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = convert_to_records.read_pickle(FLAGS.input_file)
#
#     _test_dataset, _test_labels = create_sequence(test_dataset, test_labels, 16)
#     convert_to_records.convert_to(_test_dataset, _test_labels, FLAGS.directory, 'test.sequence')
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         '--input-file',
#         type=str,
#         default='notMNIST.pickle',
#         help='Pickle file with not MNIST data'
#     )
#     parser.add_argument(
#         '--directory',
#         type=str,
#         default='/tmp/data/',
#         help='Directory to write the converted result'
#     )
#
#     FLAGS, unparsed = parser.parse_known_args()
#     tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
