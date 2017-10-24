from six.moves import cPickle as pickle
import numpy as np

# Data properties
IMAGE_ROWS, IMAGE_COLS, NUM_CHANNELS = 28, 28, 1
NUM_LABELS = 10
PRETTY_LABELS = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J'}


def load_data(pickle_file='notMNIST.pickle', verbose=0):
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save  # hint to help gc free up memory

        train_dataset, train_labels = reformat(train_dataset, train_labels)
        valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
        test_dataset, test_labels = reformat(test_dataset, test_labels)

        if verbose:
            print('Training set', train_dataset.shape, train_labels.shape)
            print('Validation set', valid_dataset.shape, valid_labels.shape)
            print('Test set', test_dataset.shape, test_labels.shape)

        return (train_dataset, train_labels), (valid_dataset, valid_labels), (test_dataset, test_labels)

def reformat(dataset, labels):
    dataset = dataset.reshape(
        (-1, IMAGE_ROWS, IMAGE_COLS, NUM_CHANNELS)).astype(np.float32)
    labels = (np.arange(NUM_LABELS) == labels[:, None]).astype(np.float32)
    return dataset, labels

