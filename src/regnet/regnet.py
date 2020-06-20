import numpy as np
import tensorflow as tf
from tensorflow.python import keras

import src.helpers.keras as kh
from src.helpers import paths

config = paths.config.read(paths.config.regnet())
IMAGE_WIDTH = int(config['DATA_INFORMATION']['IMAGE_WIDTH'])
IMAGE_HEIGHT = int(config['DATA_INFORMATION']['IMAGE_HEIGHT'])
DUAL_QUATERNIONS = config['DATA_INFORMATION']['DUAL_QUATERNIONS']

INPUT_RGB_SHAPE = [IMAGE_HEIGHT, IMAGE_WIDTH, 3]
INPUT_DEPTH_SHAPE = [IMAGE_HEIGHT, IMAGE_WIDTH, 1]

if DUAL_QUATERNIONS:
  LABEL_CALIB_SHAPE = [8]
else:
  LABEL_CALIB_SHAPE = [4, 4]
LABEL_CALIB_SHAPE_PROD = np.prod(LABEL_CALIB_SHAPE)


class Regnet:
  def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08,
               hasPipeline=False, pipeline_inputs=None, pipeline_labels=None):
    # Build placeholders for the inputs and labels
    if hasPipeline:
      self.input_rgb = kh.layers.pipeline(pipeline_inputs[0], name='rgb_in')
      self.input_depth = kh.layers.pipeline(pipeline_inputs[1], name='depth_in')
      self.label = kh.layers.pipeline(pipeline_labels, name='decalib_in')
    else:
      self.input_rgb = kh.layers.placeholder(shape=INPUT_RGB_SHAPE,
                                             name='rgb_in')

      self.input_depth = kh.layers.placeholder(shape=INPUT_DEPTH_SHAPE,
                                               name='depth_in')

      self.label = kh.layers.placeholder(shape=[LABEL_CALIB_SHAPE_PROD],
                                         name='decalib_in')

    # Create the network
    self.logits = self.init_network(self.input_rgb, self.input_depth)

    inputs = [self.input_rgb, self.input_depth]
    self.model = keras.models.Model(inputs=inputs, outputs=self.logits)

    # Define Loss (Euclidean)
    with tf.variable_scope("model_loss"):
      self.model_loss = kh.metrics.euclidean_loss
      # self.model_loss = tf.losses.huber_loss(labels=self.label, predictions=self.logits)

    # Define Optimizations (Adam)
    with tf.variable_scope("train_opt"):
      self.train_opt = keras.optimizers.Adam(lr=learning_rate,
                                             beta_1=beta1,
                                             beta_2=beta2,
                                             epsilon=epsilon)

    # Define Accuracy metric
    with tf.variable_scope("accuracy"):
      self.metrics = ['mae', kh.metrics.r2_score]

    graph_path = paths.models.regnet_tf().joinpath('model.png')
    graph_path.parent.mkdir(exist_ok=True, parents=True)
    keras.utils.plot_model(self.model, to_file=str(graph_path))

  def init_network(self, rgb_intput, depth_input):
    """
    Construct the Regnet network as per description in the Regnet paper

    :param rgb_intput: Tensor input to the network
    :param depth_input: Tensor input to the network
    :return: Output tensor
    """

    with tf.variable_scope("regnet"):
      # the Feature Extraction Module (RGB and LiDar)
      rgb_features, depth_features = self.feature_extraction(rgb_intput, depth_input)

      '''
      here we concatenate the features extracted from the rgb module
      and the features extracted from the depth module in order to
      pass them to the next block (feature matching)
      '''
      # concat features
      stacked_features = kh.layers.concatenate(values=(rgb_features, depth_features), name='concat')

      # the Feature Match Module
      matched_features = self.feature_matching(stacked_features)

      # the Global Regression Module
      calibration_parameters = self.global_regression(matched_features)

      return calibration_parameters

  def rgb_feature_extraction(self, rgb_input):
    """
    Construct the RGB Feature Extraction block as per description in the Regnet paper

    :param inputs: Tensor input to the rgb feature extraction block (camera)
    :return: Output tensor
    """

    with tf.variable_scope("rgb_feature_extraction"):
      # first NiN block
      nin1 = nin_block(inputs=rgb_input,
                       num_outputs=96,
                       kernel_size=11,
                       stride=4,
                       activation_fn='relu',
                       name="rfe-n1")

      # second NiN block
      nin2 = nin_block(inputs=nin1,
                       num_outputs=256,
                       kernel_size=5,
                       stride=1,
                       activation_fn='relu',
                       name="rfe-n2")

      # third NiN block
      nin3 = nin_block(inputs=nin2,
                       num_outputs=384,
                       kernel_size=3,
                       stride=1,
                       activation_fn='relu',
                       name="rfe-n3")

    return nin3

  def depth_feature_extraction(self, depth_input):
    """
    Construct the Depth Feature Extraction block as per description in the Regnet paper

    :param inputs: Tensor input to the depth feature extraction block (Lidar)
    :return: Output tensor
    """
    with tf.variable_scope("depth_feature_extraction"):
      # first pooling layer
      pool1 = kh.layers.max_pool2d(inputs=depth_input,
                                   kernel_size=3,
                                   stride=1,
                                   padding='same',
                                   scope='dfe-pool1')

      # second pooling layer
      pool2 = kh.layers.max_pool2d(inputs=pool1,
                                   kernel_size=3,
                                   stride=1,
                                   padding='same',
                                   scope='dfe-pool2')

      # first NiN block
      nin1 = nin_block(inputs=pool2,
                       num_outputs=48,
                       kernel_size=11,
                       stride=4,
                       activation_fn='relu',
                       name="dfe-n1")

      # second NiN block
      nin2 = nin_block(inputs=nin1,
                       num_outputs=128,
                       kernel_size=5,
                       stride=1,
                       activation_fn='relu',
                       name="dfe-n2")

      # third NiN block
      nin3 = nin_block(inputs=nin2,
                       num_outputs=192,
                       kernel_size=3,
                       stride=1,
                       activation_fn='relu',
                       name="dfe-n3")

    return nin3

  def feature_extraction(self, rgb_intput, depth_input):
    """
    Construct the Feature Extraction block as per description in the Regnet paper

    :param rgb_intput: Tensor input to the rgb feature extraction block
    :param depth_input: Tensor input to the depth feature extraction block
    :return: Output tensor
    """

    '''
    We have two *independent* feature extraction modules
    since it is better for each modality to train on its
    own first then concatinated the found features after
    '''

    with tf.variable_scope("feature_extraction"):
      # the RGB Feature Extraction Module (Camera)
      rgb_features = self.rgb_feature_extraction(rgb_intput)

      # the Depth Feature Extraction Module (LiDar)
      depth_features = self.depth_feature_extraction(depth_input)

    return rgb_features, depth_features

  def feature_matching(self, stacked_input):
    """
    Construct the Feature Matching block as per description in the Regnet paper

    :param stacked_input: Tensor input to the feature matching block
        which a concatinated stack of the extracted rgb and depth features
    :return: Output tensor
    """

    with tf.variable_scope("feature_matching"):
      # first NiN block
      nin1 = nin_block(inputs=stacked_input,
                       num_outputs=512,
                       kernel_size=5,
                       stride=1,
                       activation_fn='relu',
                       name="fm-n1")

      # second NiN block
      nin2 = nin_block(inputs=nin1,
                       num_outputs=512,
                       kernel_size=3,
                       stride=1,
                       activation_fn='relu',
                       name="fm-n2")
    return nin2

  def global_regression(self, matched_input):
    """
    Construct the Global Regression block as per description in the Regnet paper

    :param matched_input: Tensor input to the global regression block
        which is a unified representation for the inputs
    :return: Output tensor
    """

    with tf.variable_scope("global_regression"):
      # first fully connected layer
      fc1 = kh.layers.flatten(matched_input, name='flatten')
      fc1 = kh.layers.fully_connected(inputs=fc1,
                                      num_outputs=512,
                                      activation_fn='relu',
                                      scope='gr-fc1')

      # second fully connected layer
      fc2 = kh.layers.fully_connected(inputs=fc1,
                                      num_outputs=256,
                                      activation_fn='relu',
                                      scope='gr-fc2')

      # third fully connected layer
      fc3 = kh.layers.fully_connected(inputs=fc2,
                                      num_outputs=LABEL_CALIB_SHAPE_PROD,
                                      activation_fn=None,
                                      scope='gr-fc3')
    return fc3


def nin_block(inputs, num_outputs, kernel_size, stride=1, padding='same', activation_fn=None,
              name=None):
  """
  Construct a Network in Network block as per description in the Regnet paper

  :param inputs: Tensor input to the NiN block
  :param filters: the number of feature channels. Integer, the
      dimensionality of the output space.
  :param kernel_size: the kernel size `k`. An integer, specifying the height
      and width of the 2D convolution window.
  :param padding: the padding algorithm to be used One of "VALID" or "SAME".
      (case-insensitive).
  :activation: Activation function. The default value is a ReLU function.
      Explicitly set it to None to skip it and maintain a linear activation.
  :name: the name of the layer.
  :return: Output tensor
  """
  with tf.variable_scope(name):
    # first convolution layer
    conv1 = kh.layers.convolution2d(inputs=inputs,
                                    num_outputs=num_outputs,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    activation_fn=activation_fn,
                                    scope=name + '-conv1')

    # the mlpconv kh.layers
    cccp1 = kh.layers.convolution2d(inputs=conv1,
                                    num_outputs=num_outputs,
                                    kernel_size=1,
                                    stride=1,
                                    padding=padding,
                                    activation_fn=activation_fn,
                                    scope=name + '-cccp1')

    cccp2 = kh.layers.convolution2d(inputs=cccp1,
                                    num_outputs=num_outputs,
                                    kernel_size=1,
                                    stride=1,
                                    padding=padding,
                                    activation_fn=activation_fn,
                                    scope=name + '-cccp2')

    # final pooling layer
    maxpool = kh.layers.max_pool2d(inputs=cccp2,
                                    kernel_size=3,
                                    stride=2,
                                    padding='SAME',
                                    scope=name + '-max')
  return maxpool
