"""Converts notMNIST data into sequences of numbers"""
import numpy as np
from sklearn.utils import shuffle
from notmnist import notMNIST
from six.moves import cPickle as pickle
import os

# Defaults to sequence of 5 numbers
SEQUENCE_LENGTH = 5

def create_sequence(data, labels, sequence_length=SEQUENCE_LENGTH, limit=0):
    '''Create a sequence of numbers from a given number data set'''
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
        if j + sequence_length >= len(data):
            np.random.shuffle(data)
            j=0

        # Get a slice into data capturing SEQUENCE_LENGTH images, then concatenate them horizontally
        # Todo: should we add in a separator?
        # Todo we need to add in empty characters too!
        data_slice = data[j:j+sequence_length,]
        labels_slice = labels[j:j+sequence_length,]
        img = np.concatenate(data_slice, axis=1)

        sequence.append(img)
        sequence_labels.append(labels_slice)

        # Increase our both counters
        i += 1
        j += sequence_length

    # Convert back to nparrays, which are easier to manage
    sequence = np.asarray(sequence)
    sequence_labels = np.asarray(sequence_labels)

    return sequence, sequence_labels

def load_data_or_generate(pickle_seq_file='numbersequence/notMNIST.sequence.pickle', pickle_file='notmnist/notMNIST.pickle', verbose=0, overwrite=0):
    '''Load data or generate new data from notMNIST if data cannot be found'''
    if not os.path.exists(pickle_seq_file) or overwrite:
        print('Generating new sequence pickle from ', pickle_file, ' and saving to ', pickle_seq_file)
        (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = notMNIST.load_data(pickle_file, verbose=0)
        x2_train, y2_train = create_sequence(x_train, y_train, limit=10000)
        x2_valid, y2_valid = create_sequence(x_valid, y_valid, limit=1000)
        x2_test, y2_test = create_sequence(x_test, y_test, limit=1000)

        print('Generated training set', x2_train.shape, y2_train.shape)
        print('Generated validation set', x2_valid.shape, y2_valid.shape)
        print('Generated test set', x2_test.shape, y2_test.shape)

        try:
            with open(pickle_seq_file, 'wb') as f:
                save = {
                    'train_dataset': x2_train,
                    'train_labels': y2_train,
                    'valid_dataset': x2_valid,
                    'valid_labels': y2_valid,
                    'test_dataset': x2_test,
                    'test_labels': y2_test,
                }
                pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
                f.close()
        except Exception as e:
            print('Unable to save data to', pickle_seq_file, ':', e)
            return
    else:
        print('Loading existing sequence pickle from ', pickle_seq_file)
        try:
            with open(pickle_seq_file, 'rb') as f:
                save = pickle.load(f)
                x2_train = save['train_dataset']
                y2_train = save['train_labels']
                x2_valid = save['valid_dataset']
                y2_valid = save['valid_labels']
                x2_test = save['test_dataset']
                y2_test = save['test_labels']
                del save  # hint to help gc free up memory
        except Exception as e:
            print('Unable to read data from', pickle_seq_file, ':', e)
            return

    if verbose:
        print('Training set', x2_train.shape, y2_train.shape)
        print('Validation set', x2_valid.shape, y2_valid.shape)
        print('Test set', x2_test.shape, y2_test.shape)

    return (x2_train, y2_train), (x2_valid, y2_valid), (x2_test, y2_test)

## Visualise some examples
# (x2_train, y2_train), (x2_valid, y2_valid), (x2_test, y2_test) = load_data_or_generate(pickle_seq_file='numbersequence/notMNIST.sequence.pickle', verbose=1, overwrite=1)
# notMNIST.visualize_batch(x2_train[0:8:], y2_train[0:8:].argmax(axis=-1))
