from pathlib import Path

import numpy as np
import tensorflow as tf

from src.helpers import paths
from src.regnet import regnet

PRETRIAN_MODEL_PATH = paths.checkpoints.regnet().with_name('training.ckpt')

config = paths.config.read(paths.config.regnet())
BETA1 = float(config['HYPERPARAMETERS']['BETA1'])
BETA2 = float(config['HYPERPARAMETERS']['BETA2'])
EPSILON = float(config['HYPERPARAMETERS']['EPSILON'])
LEARNING_RATE = float(config['HYPERPARAMETERS']['LEARNING_RATE'])


def load_tensorflow_weights(model, checkpoint_path):
  ckpt_vars = tf.train.list_variables(str(checkpoint_path))
  ckpt_vars = [name for name, _ in ckpt_vars]

  ckpt_vars = np.array(
      list(filter(lambda x: not x.startswith('train_opt'), ckpt_vars)))
  ckpt_vars = zip_weight_bias(ckpt_vars)

  for weights, biases in ckpt_vars:
    assign_layer_var(model, checkpoint_path, weights, biases)

  assert_weights(model, ckpt_vars, checkpoint_path)


def assert_weights(model, ckpt_vars, checkpoint_path):
  for weights, biases in ckpt_vars:
    layer_name = Path(weights).parent.name
    model_weights, model_biases = model.get_layer(layer_name).get_weights()

    ckpt_weights = get_variable_by_name(checkpoint_path, weights)
    ckpt_biases = get_variable_by_name(checkpoint_path, biases)

    assert np.array_equal(model_weights, ckpt_weights)
    assert np.array_equal(model_biases, ckpt_biases)


def zip_weight_bias(array):
  weights = np.array(list(filter(lambda x: 'weights' in x, array)))
  weights = np.array(weights)
  weights = np.sort(weights)

  biases = np.array(list(filter(lambda x: 'biases' in x, array)))
  biases = np.array(biases)
  weights = np.sort(weights)

  zipped_array = np.column_stack((weights, biases))
  for weights, biases in zipped_array:
    weights_layer_name = Path(weights).parent
    biases_layer_name = Path(biases).parent
    assert weights_layer_name == biases_layer_name

  return zipped_array


def assign_layer_var(model,
                     checkpoint_path,
                     weights_name,
                     biases_name,
                     debug=False):
  layer_name = Path(weights_name).parent.name

  weights_var = get_variable_by_name(checkpoint_path, weights_name)
  biases_var = get_variable_by_name(checkpoint_path, biases_name)

  model.get_layer(layer_name).set_weights([weights_var, biases_var])


def get_variable_by_name(checkpoint_path, name):
  return tf.train.load_variable(str(checkpoint_path), name)


if __name__ == '__main__':
  net = regnet.Regnet(LEARNING_RATE, BETA1, BETA2, EPSILON)
  net.model.compile(
      optimizer=net.train_opt, loss=net.model_loss, metrics=net.metrics)
  load_tensorflow_weights(net.model, PRETRIAN_MODEL_PATH)
  net.model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
