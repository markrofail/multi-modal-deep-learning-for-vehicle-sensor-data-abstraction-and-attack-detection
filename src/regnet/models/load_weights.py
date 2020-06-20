import numpy as np

from src.helpers import paths

WEITGHTS_PATH = paths.models.imagenet().joinpath('nin_imagenet_weights')


def imagenet_weights(model, debug=False):
  layers_files = [('rfe-n1-conv1', 'conv1'), ('rfe-n1-cccp1', 'cccp1'), ('rfe-n1-cccp2', 'cccp2'),
                  ('rfe-n2-conv1', 'conv2'), ('rfe-n2-cccp1', 'cccp3'), ('rfe-n2-cccp2', 'cccp4'),
                  ('rfe-n3-conv1', 'conv3'), ('rfe-n3-cccp1', 'cccp5'), ('rfe-n3-cccp2', 'cccp6')]

  for layer_name, files_name in layers_files:
    assign_layer_var(layer_name, files_name, model, debug=debug)


def assign_layer_var(layer_name, file_name, model, debug=False):
  weights = get_weights(file_name)
  biases = get_bias(file_name)
  model.get_layer(layer_name).set_weights([weights.T, biases.T])

  if debug:
    print(layer_name + " success")


def get_weights(name):
  return np.load(WEITGHTS_PATH.joinpath(name + '_0.npy'), encoding='latin1')


def get_bias(name):
  return np.load(WEITGHTS_PATH.joinpath(name + '_1.npy'), encoding='latin1')
