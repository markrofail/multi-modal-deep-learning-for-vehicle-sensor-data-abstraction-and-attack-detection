from pathlib import Path

import numpy as np
import tensorflow as tf

from src.helpers import paths
from src.regnet import regnet

PRETRIAN_MODEL_PATH = paths.checkpoints.regnet().parent.with_name(
    'training.ckpt')
WEITGHTS_PATH = paths.models.regnet_tf()

config = paths.config.read(paths.config.regnet())
BETA1 = float(config['HYPERPARAMETERS']['BETA1'])
BETA2 = float(config['HYPERPARAMETERS']['BETA2'])
EPSILON = float(config['HYPERPARAMETERS']['EPSILON'])
LEARNING_RATE = float(config['HYPERPARAMETERS']['LEARNING_RATE'])


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
  with tf.gfile.GFile(str(frozen_graph_filename), "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

  # Then, we import the graph_def into a new Graph and returns it
  with tf.Graph().as_default() as graph:
    # The name var will prefix every op/nodes in your graph
    # Since we load everything in a new graph, this is not needed
    tf.import_graph_def(graph_def=graph_def, name="")
  return graph


def load_tensorflow_weights(model, checkpoint_path):
  graph = load_graph(WEITGHTS_PATH.joinpath('regnet_frozen.pb'))
  sess_regnet = tf.Session(graph=graph)

  ckpt_vars = [var for op in graph.get_operations() for var in op.values()]
  ckpt_vars = [
      var for var in ckpt_vars
      if 'weights:0' in var.name or 'biases:0' in var.name
  ]

  ckpt_vars = np.array(
      list(filter(lambda x: not x.name.startswith('train_opt'), ckpt_vars)))
  ckpt_vars = zip_weight_bias(ckpt_vars)

  for weights, biases in ckpt_vars:
    assign_layer_var(model, checkpoint_path, weights, biases, sess_regnet)

  assert_weights(model, ckpt_vars, checkpoint_path, sess_regnet)


def assert_weights(model, ckpt_vars, checkpoint_path, sess_regnet):
  for weights, biases in ckpt_vars:
    layer_name = Path(weights.name).parent.name
    model_weights, model_biases = model.get_layer(layer_name).get_weights()

    ckpt_weights = sess_regnet.run(weights)
    ckpt_biases = sess_regnet.run(biases)

    assert np.array_equal(model_weights, ckpt_weights)
    assert np.array_equal(model_biases, ckpt_biases)


def zip_weight_bias(array):
  weights = np.array(list(filter(lambda x: 'weights' in x.name, array)))
  weights = np.array(weights)

  biases = np.array(list(filter(lambda x: 'biases' in x.name, array)))
  biases = np.array(biases)

  zipped_array = np.column_stack((weights, biases))
  for weights, biases in zipped_array:
    weights_layer_name = Path(weights.name).parent
    biases_layer_name = Path(biases.name).parent
    assert weights_layer_name == biases_layer_name

  return zipped_array


def assign_layer_var(model, checkpoint_path, weight, biases, sess_regnet):
  layer_name = Path(weight.name).parent.name

  weights_var = sess_regnet.run(weight)
  biases_var = sess_regnet.run(biases)

  model.get_layer(layer_name).set_weights([weights_var, biases_var])


def get_variable_by_name(checkpoint_path, name):
  return tf.train.load_variable(str(checkpoint_path), name)


if __name__ == '__main__':
  net = regnet.Regnet(LEARNING_RATE, BETA1, BETA2, EPSILON)
  net.model.compile(
      optimizer=net.train_opt, loss=net.model_loss, metrics=net.metrics)
  load_tensorflow_weights(net.model, PRETRIAN_MODEL_PATH)
  net.model.save('training.h5')  # creates a HDF5 file 'my_model.h5'
