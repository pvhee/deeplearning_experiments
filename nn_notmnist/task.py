"""This code sets up a task to implement a DNN model for notMNIST number recognition.
"""

import argparse
import json
import os
import threading

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def run(target,
        is_chief,
        train_steps,
        eval_steps,
        job_dir,
        train_files,
        eval_files,
        train_batch_size,
        eval_batch_size):

  """Run the training and evaluation graph.

  Args:
    target (string): Tensorflow server target
    is_chief (bool): Boolean flag to specify a chief server
    train_steps (int): Maximum number of training steps
    eval_steps (int): Number of steps to run evaluation for at each checkpoint
    job_dir (string): Output dir for checkpoint and summary
    train_files (string): List of CSV files to read train data
    eval_files (string): List of CSV files to read eval data
    train_batch_size (int): Batch size for training
    eval_batch_size (int): Batch size for evaluation
  """

  # Calculate the number of hidden units
  # hidden_units=[
  #     max(2, int(first_layer_size * scale_factor**i))
  #     for i in range(num_layers)
  # ]

  # If the server is chief which is `master`
  # In between graph replication Chief is one node in
  # the cluster with extra responsibility and by default
  # is worker task zero. We have assigned master as the chief.
  #
  # See https://youtu.be/la_M6bCV91M?t=1203 for details on
  # distributed TensorFlow and motivation about chief.
  if is_chief:
    tf.logging.info("Created DNN hidden units {}".format(hidden_units))
    evaluation_graph = tf.Graph()
    with evaluation_graph.as_default():

      # Features and label tensors
      features, labels = model.input_fn(
          eval_files,
          num_epochs=eval_num_epochs,
          batch_size=eval_batch_size,
          shuffle=False
      )
      # Accuracy and AUROC metrics
      # model.model_fn returns the dict when EVAL mode
      metric_dict = model.model_fn(
          model.EVAL,
          features,
          labels,
          hidden_units=hidden_units,
          learning_rate=learning_rate
      )

    hooks = [EvaluationRunHook(
        job_dir,
        metric_dict,
        evaluation_graph,
        eval_frequency,
        eval_steps=eval_steps,
    )]
  else:
    hooks = []

  # Create a new graph and specify that as default
  with tf.Graph().as_default():
    # Placement of ops on devices using replica device setter
    # which automatically places the parameters on the `ps` server
    # and the `ops` on the workers
    #
    # See https://www.tensorflow.org/api_docs/python/tf/train/replica_device_setter
    with tf.device(tf.train.replica_device_setter()):

      # Returns the training graph and global step tensor
      train_op, global_step, train_prediction, valid_prediction, test_prediction = model.model_fn(
          model.TRAIN
      )


    # Creates a MonitoredSession for training
    # MonitoredSession is a Session-like object that handles
    # initialization, recovery and hooks
    # https://www.tensorflow.org/api_docs/python/tf/train/MonitoredTrainingSession
    with tf.train.MonitoredTrainingSession(master=target,
                                           is_chief=is_chief,
                                           checkpoint_dir=job_dir,
                                           hooks=hooks,
                                           save_checkpoint_secs=20,
                                           save_summaries_steps=50) as session:

      # Tuple of exceptions that should cause a clean stop of the coordinator
      # https://www.tensorflow.org/api_guides/python/train#Coordinator_and_QueueRunner
      coord = tf.train.Coordinator(clean_stop_exception_types=(
          tf.errors.CancelledError, tf.errors.OutOfRangeError))

      # Important to start all queue runners so that data is available
      # for reading
      tf.train.start_queue_runners(coord=coord, sess=session)

      # Global step to keep track of global number of steps particularly in
      # distributed setting
      step = global_step_tensor.eval(session=session)

      # Run the training graph which returns the step number as tracked by
      # the global step tensor.
      # When train epochs is reached, coord.should_stop() will be true.
      with coord.stop_on_exception():
        while (train_steps is None or step < train_steps) and not coord.should_stop():
          step, _ = session.run([global_step_tensor, train_op])

    # Find the filename of the latest saved checkpoint file
    latest_checkpoint = tf.train.latest_checkpoint(job_dir)

    # Only perform this if chief
    if is_chief:
      build_and_run_exports(latest_checkpoint,
                            job_dir,
                            'CSV',
                            model.csv_serving_input_fn,
                            hidden_units)
      build_and_run_exports(latest_checkpoint,
                            job_dir,
                            'JSON',
                            model.json_serving_input_fn,
                            hidden_units)
      build_and_run_exports(latest_checkpoint,
                            job_dir,
                            'EXAMPLE',
                            model.example_serving_input_fn,
                            hidden_units)

def dispatch(*args, **kwargs):
  """Parse TF_CONFIG to cluster_spec and call run() method
  TF_CONFIG environment variable is available when running using
  gcloud either locally or on cloud. It has all the information required
  to create a ClusterSpec which is important for running distributed code.
  """

  tf_config = os.environ.get('TF_CONFIG')

  # If TF_CONFIG is not available run local
  if not tf_config:
    return run('', True, *args, **kwargs)

  tf_config_json = json.loads(tf_config)

  cluster = tf_config_json.get('cluster')
  job_name = tf_config_json.get('task', {}).get('type')
  task_index = tf_config_json.get('task', {}).get('index')

  # If cluster information is empty run local
  if job_name is None or task_index is None:
    return run('', True, *args, **kwargs)

  cluster_spec = tf.train.ClusterSpec(cluster)
  server = tf.train.Server(cluster_spec,
                           job_name=job_name,
                           task_index=task_index)

  # Wait for incoming connections forever
  # Worker ships the graph to the ps server
  # The ps server manages the parameters of the model.
  #
  # See a detailed video on distributed TensorFlow
  # https://www.youtube.com/watch?v=la_M6bCV91M
  if job_name == 'ps':
    server.join()
    return
  elif job_name in ['master', 'worker']:
    return run(server.target, job_name == 'master', *args, **kwargs)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--train-files',
                      required=True,
                      type=str,
                      help='Training files local or GCS', nargs='+')
  parser.add_argument('--eval-files',
                      required=True,
                      type=str,
                      help='Evaluation files local or GCS', nargs='+')
  parser.add_argument('--job-dir',
                      required=True,
                      type=str,
                      help='GCS or local dir to write checkpoints and export model')
  parser.add_argument('--train-steps',
                      type=int,
                      help="""\
                       Maximum number of training steps to perform
                       Training steps are in the units of training-batch-size.
                       So if train-steps is 500 and train-batch-size if 100 then
                       at most 500 * 100 training instances will be used to train.
                      """)
  parser.add_argument('--eval-steps',
                      help='Number of steps to run evalution for at each checkpoint',
                      default=100,
                      type=int)
  parser.add_argument('--train-batch-size',
                      type=int,
                      default=40,
                      help='Batch size for training steps')
  parser.add_argument('--eval-batch-size',
                      type=int,
                      default=40,
                      help='Batch size for evaluation steps')
  # parser.add_argument('--learning-rate',
  #                     type=float,
  #                     default=0.003,
  #                     help='Learning rate for SGD')
  # parser.add_argument('--eval-frequency',
  #                     default=50,
  #                     help='Perform one evaluation per n steps')
  # parser.add_argument('--first-layer-size',
  #                    type=int,
  #                    default=256,
  #                    help='Number of nodes in the first layer of DNN')
  # parser.add_argument('--num-layers',
  #                    type=int,
  #                    default=2,
  #                    help='Number of layers in DNN')
  # parser.add_argument('--scale-factor',
  #                    type=float,
  #                    default=0.25,
  #                    help="""\
  #                     Rate of decay size of layer for Deep Neural Net.
  #                     max(2, int(first_layer_size * scale_factor**i)) \
  #                     """)
  # parser.add_argument('--eval-num-epochs',
  #                    type=int,
  #                    default=1,
  #                    help='Number of epochs during evaluation')
  # parser.add_argument('--num-epochs',
  #                     type=int,
  #                     help='Maximum number of epochs on which to train')

  parse_args, unknown = parser.parse_known_args()

  tf.logging.warn('Unknown arguments: {}'.format(unknown))

  print(parse_args)
  dispatch(**parse_args.__dict__)
