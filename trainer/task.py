"""This code sets up a task to implement a DNN model for notMNIST number recognition.
"""

from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

import argparse
import json
import os
import time
import threading


# import model

NUM_STEPS = 1001
# NUM_STEPS = 9001

IMAGE_SIZE = 28
NUM_LABELS = 10
NUM_CHANNELS = 1 # grayscale

PATCH_SIZE = 4
DEPTH = 24
NUM_HIDDEN = 256
NUM_HIDDEN2 = 128

BATCH_SIZE = 32

tf.logging.set_verbosity(tf.logging.INFO)

## Todo: model after
## https://github.com/tensorflow/models/blob/master/tutorials/image/mnist/convolutional.py

def input_fn(pickle_file='notMNIST.pickle'):
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

    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

    return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)).astype(np.float32)
  labels = (np.arange(NUM_LABELS) == labels[:,None]).astype(np.float32)
  return dataset, labels

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

def main(_):
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

  # if mode in (TRAIN, EVAL):
    # global_step is necessary in eval to correctly load the step
    # of the checkpoint we are evaluating
    # global_step = tf.contrib.framework.get_or_create_global_step()

  # Optimizer.
  # optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss, global_step = global_step)
  optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  test_prediction = tf.nn.softmax(model(tf_test_dataset))

  # Create a local session to run the training.
  start_time = time.time()

  with tf.Session() as session:

    tf.global_variables_initializer().run()
    print('Initialized')

    for step in range(NUM_STEPS):
      offset = (step * BATCH_SIZE) % (train_labels.shape[0] - BATCH_SIZE)
      batch_data = train_dataset[offset:(offset + BATCH_SIZE), :, :, :]
      batch_labels = train_labels[offset:(offset + BATCH_SIZE), :]
      feed_dict = {tf_train_dataset : batch_data,
                  tf_train_labels : batch_labels}
      _, l, predictions = session.run([optimizer, loss, train_prediction],
                                      feed_dict=feed_dict)
      if (step % 50 == 0):
        print('\tMinibatch loss at step %d: %f' % (step, l))
        print('\tMinibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
        print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
        print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

  end_time = time.time()
  print('Duration: %f seconds', end_time-start_time)

# def run():

#   """Run the training and evaluation graph.

#   Args:
#     target (string): Tensorflow server target
#     is_chief (bool): Boolean flag to specify a chief server
#     train_steps (int): Maximum number of training steps
#     eval_steps (int): Number of steps to run evaluation for at each checkpoint
#     job_dir (string): Output dir for checkpoint and summary
#     train_files (string): List of CSV files to read train data
#     eval_files (string): List of CSV files to read eval data
#     train_batch_size (int): Batch size for training
#     eval_batch_size (int): Batch size for evaluation
#   """

#   graph = tf.Graph()

#   # Create a new graph and specify that as default
#   with tf.Graph().as_default():
#     # Placement of ops on devices using replica device setter
#     # which automatically places the parameters on the `ps` server
#     # and the `ops` on the workers
#     #
#     # See https://www.tensorflow.org/api_docs/python/tf/train/replica_device_setter
#     with tf.device(tf.train.replica_device_setter()):

#       # Features and labels

#       # Returns the training graph and global step tensor
#       tf_train_dataset, tf_train_labels, optimizer, loss, train_prediction, valid_prediction, test_prediction = model.model_fn(
#           model.TRAIN
#       )


#     # Creates a MonitoredSession for training
#     # MonitoredSession is a Session-like object that handles
#     # initialization, recovery and hooks
#     # https://www.tensorflow.org/api_docs/python/tf/train/MonitoredTrainingSession
#     # with tf.train.MonitoredTrainingSession(master=target,
#     #                                        is_chief=is_chief,
#     #                                        checkpoint_dir=job_dir,
#     #                                        hooks=hooks,
#     #                                        save_checkpoint_secs=20,
#     #                                        save_summaries_steps=50) as session:
#     with tf.Session(graph=graph) as session:

#       # Tuple of exceptions that should cause a clean stop of the coordinator
#       # https://www.tensorflow.org/api_guides/python/train#Coordinator_and_QueueRunner
#       # coord = tf.train.Coordinator(clean_stop_exception_types=(
#           # tf.errors.CancelledError, tf.errors.OutOfRangeError))

#       # Important to start all queue runners so that data is available
#       # for reading
#       # tf.train.start_queue_runners(coord=coord, sess=session)

#       # Global step to keep track of global number of steps particularly in
#       # distributed setting
#       # step = global_step_tensor.eval(session=session)

#       tf.global_variables_initializer().run()
#       print('Initialized')

#       for step in range(NUM_STEPS):
#         offset = (step * BATCH_SIZE) % (train_labels.shape[0] - BATCH_SIZE)
#         batch_data = train_dataset[offset:(offset + BATCH_SIZE), :, :, :]
#         batch_labels = train_labels[offset:(offset + BATCH_SIZE), :]
#         feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
#         _, l, predictions = session.run(
#           [optimizer, loss, train_prediction], feed_dict=feed_dict)
#         if (step % 50 == 0):
#           print('Minibatch loss at step %d: %f' % (step, l))
#           print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
#           print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
#           print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

#       # Run the training graph which returns the step number as tracked by
#       # the global step tensor.
#       # When train epochs is reached, coord.should_stop() will be true.
#       # with coord.stop_on_exception():
#       #   while (train_steps is None or step < train_steps) and not coord.should_stop():
#       #     step, _ = session.run([global_step_tensor, train_op])

#     # # Find the filename of the latest saved checkpoint file
#     # latest_checkpoint = tf.train.latest_checkpoint(job_dir)

#     # # Only perform this if chief
#     # if is_chief:
#     #   build_and_run_exports(latest_checkpoint,
#     #                         job_dir,
#     #                         'CSV',
#     #                         model.csv_serving_input_fn,
#     #                         hidden_units)
#     #   build_and_run_exports(latest_checkpoint,
#     #                         job_dir,
#     #                         'JSON',
#     #                         model.json_serving_input_fn,
#     #                         hidden_units)
#     #   build_and_run_exports(latest_checkpoint,
#     #                         job_dir,
#     #                         'EXAMPLE',
#     #                         model.example_serving_input_fn,
#     #                         hidden_units)

def dispatch(*args, **kwargs):
  """Parse TF_CONFIG to cluster_spec and call run() method
  TF_CONFIG environment variable is available when running using
  gcloud either locally or on cloud. It has all the information required
  to create a ClusterSpec which is important for running distributed code.
  """
  tf.app.run(main=main)

  # tf_config = os.environ.get('TF_CONFIG')

  # # If TF_CONFIG is not available run local
  # if not tf_config:
  #   return run('', True, *args, **kwargs)

  # tf_config_json = json.loads(tf_config)

  # cluster = tf_config_json.get('cluster')
  # job_name = tf_config_json.get('task', {}).get('type')
  # task_index = tf_config_json.get('task', {}).get('index')

  # # If cluster information is empty run local
  # if job_name is None or task_index is None:
  #   return run()

  # cluster_spec = tf.train.ClusterSpec(cluster)
  # server = tf.train.Server(cluster_spec,
  #                          job_name=job_name,
  #                          task_index=task_index)

  # Wait for incoming connections forever
  # Worker ships the graph to the ps server
  # The ps server manages the parameters of the model.
  #
  # See a detailed video on distributed TensorFlow
  # https://www.youtube.com/watch?v=la_M6bCV91M
  # if job_name == 'ps':
  #   server.join()
  #   return
  # elif job_name in ['master', 'worker']:
  #   # return run(server.target, job_name == 'master', *args, **kwargs)
  #   return run()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  FLAGS, unknown = parser.parse_known_args()
  tf.app.run(main=main)
