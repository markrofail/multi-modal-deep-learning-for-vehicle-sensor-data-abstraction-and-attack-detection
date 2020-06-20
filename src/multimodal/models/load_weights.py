from tensorflow.python import keras

from src.helpers import paths, keras as kh
from src.regnet import regnet

WEITGHTS_PATH = paths.models.regnet_tf()
CUSTOM_OBJECTS = {'euclidean_loss_layer': kh.metrics.euclidean_loss}


def regnet_weights(model, debug=False):
  model_path = paths.checkpoints.regnet()  # ./checkpoints/regnet/train
  regnet_model = keras.models.load_model(
      str(model_path), custom_objects=CUSTOM_OBJECTS, compile=False)

  ignore_layers = list()

  input_layers = ['depth_in', 'rgb_in']
  ignore_layers.extend(input_layers)

  global_regression = ['gr-fc1', 'gr-fc2', 'gr-fc3']
  ignore_layers.extend(global_regression)

  for x in regnet_model.layers:
    if x.name not in ignore_layers:
      model.get_layer(x.name).set_weights(x.get_weights())
      # print('regnet/{}'.format(x.name))

      if debug:
        print('multimodal/{}'.format(model.get_layer(x.name).name))
