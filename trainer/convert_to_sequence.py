"""Converts notMNIST data into sequences of up to 5 numbers, saved as TFRecords"""
import convert_to_records
import verify_records
import argparse
import sys
import tensorflow as tf
import numpy as np

SEQUENCE_LENGTH = 2

def create_sequence(data, labels):
    i = 0
    for j in range(SEQUENCE_LENGTH):
        data[i]
        i += 1
        print(data[i].shape)

    combi = np.concatenate([data[0], np.full([28,28,1], -0.5), data[1]], axis=1)
    print(combi.shape)

    combi_reshaped = np.reshape(combi, [1, 28, 28*3, 1])
    print(combi_reshaped.shape)

    verify_records.plot(combi_reshaped, labels)

def main(unused_argv):
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = convert_to_records.read_pickle(FLAGS.input_file)

    # print(train_dataset.shape)
    create_sequence(test_dataset, test_labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-file',
        type=str,
        default='notMNIST.pickle',
        help='Pickle file with not MNIST data'
    )
    # parser.add_argument(
    #     '--directory',
    #     type=str,
    #     default='/tmp/data/',
    #     help='Directory to write the converted result'
    # )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
