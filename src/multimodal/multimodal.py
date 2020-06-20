import os

import tensorflow as tf
from tensorflow.python import keras

import src.helpers.keras as kh
from src.helpers import paths
from src.regnet import regnet


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#################################################################################
# Data Description                                                              #
#################################################################################
config = paths.config.read(paths.config.multimodal())
IMAGE_WIDTH = int(config['DATA_INFORMATION']['IMAGE_WIDTH'])
IMAGE_HEIGHT = int(config['DATA_INFORMATION']['IMAGE_HEIGHT'])

#################################################################################
# Network Architecture                                                          #
#################################################################################
LAYER_1_SIZE = config['NETWORK_ARCHITECTURE']['LAYER_1_SIZE']
LAYER_2_SIZE = config['NETWORK_ARCHITECTURE']['LAYER_2_SIZE']
LAYER_3_STATUS = config['NETWORK_ARCHITECTURE']['LAYER_3_STATUS']
LAYER_3_SIZE = config['NETWORK_ARCHITECTURE']['LAYER_3_SIZE']
NUM_CLASSES = 2

INPUT_RGB_SHAPE = [IMAGE_HEIGHT, IMAGE_WIDTH, 3]
INPUT_DEPTH_SHAPE = [IMAGE_HEIGHT, IMAGE_WIDTH, 1]
LABEL_SHAPE = [NUM_CLASSES]


class Multimodal(regnet.Regnet):

  def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08,
               hasPipeline=False, pipeline_inputs=None, pipeline_labels=None):
        # Build placeholders for the inputs and labels
    if hasPipeline:
      self.input_rgb = kh.layers.pipeline(pipeline_inputs[0], name='rgb_in')
      self.input_depth = kh.layers.pipeline(pipeline_inputs[1], name='depth_in')
      self.label = kh.layers.pipeline(pipeline_labels, name='label_in')
    else:
      self.input_rgb = kh.layers.placeholder(shape=INPUT_RGB_SHAPE, name='rgb_in')

      self.input_depth = kh.layers.placeholder(shape=INPUT_DEPTH_SHAPE, name='depth_in')

      self.label = kh.layers.placeholder(shape=LABEL_SHAPE, name='label_in')

    # Create the network
    self.logits = self.init_network(self.input_rgb, self.input_depth)

    inputs = [self.input_rgb, self.input_depth]
    self.model = keras.models.Model(inputs=inputs, outputs=self.logits)

    # freeze all kh.layers except the classifier
    for layer in self.model.layers[:-4]:
      layer.trainable = False

    # Define Loss (Sigmoid Cross Entropy)
    with tf.variable_scope("model_loss"):
      self.model_loss = 'binary_crossentropy'

    # Define Optimizations (Adam)
    with tf.variable_scope("train_opt"):
      self.train_opt = keras.optimizers.Adam(lr=learning_rate,
                                             beta_1=beta1,
                                             beta_2=beta2,
                                             epsilon=epsilon)

    # Define Accuracy metric
    with tf.variable_scope("accuracy"):
      self.metrics = ['binary_accuracy', kh.metrics.precision, kh.metrics.recall,
                      kh.metrics.fmeasure, kh.metrics.auc_roc]

    self.eval_metrics = [kh.metrics.false_negatives, kh.metrics.true_negatives,
                         kh.metrics.false_positives, kh.metrics.true_positives]
    self.eval_metrics.extend(self.metrics)

    graph_path = paths.models.multimodal().joinpath('model.png')
    graph_path.parent.mkdir(exist_ok=True, parents=True)
    keras.utils.plot_model(self.model, to_file=str(graph_path))

  def init_network(self, rgb_intput, depth_input):
    """
    Construct the Regnet network as per description in the Regnet paper

    :param rgb_intput: Tensor input to the network
    :param depth_input: Tensor input to the network
    :return: Output tensor
    """
    with tf.variable_scope("multimodal"):
      # the Feature Extraction Module (RGB and LiDar)
      rgb_features, depth_features = self.feature_extraction(rgb_intput, depth_input)

      '''
      here we concatenate the features extracted from the rgb module
      and the features extracted from the depth module in order to
      pass them to the next block (feature matching)
      '''
      # concat features
      stacked_features = kh.layers.concatenate(
          values=(rgb_features, depth_features), name='concat')

      # the Feature Match Module
      matched_features = self.feature_matching(stacked_features)

      # the Classifier Module
      output = self.classifier(matched_features)
    return output

  def classifier(self, multimodal_input):
    """
    Construct the Classifier block

    :param multimodal_input: Tensor input to the global regression block
        which is a unified representation for the inputs
    :return: Output tensor
    """

    '''
    2048 was inspired by the AlexNet architecture
    '''
    with tf.variable_scope("classifier"):
      # first fully connected layer
      fc1 = kh.layers.flatten(multimodal_input, name='flatten')
      fc1 = kh.layers.fully_connected(inputs=fc1,
                                      num_outputs=LAYER_1_SIZE,
                                      activation_fn='relu',
                                      scope='c-fc1')

      # second fully connected layer
      fc2 = kh.layers.fully_connected(inputs=fc1,
                                      num_outputs=LAYER_2_SIZE,
                                      activation_fn='relu',
                                      scope='c-fc2')
      if LAYER_3_STATUS:
        # third fully connected layer
        fc2 = kh.layers.fully_connected(inputs=fc2,
                                        num_outputs=LAYER_3_SIZE,
                                        activation_fn='relu',
                                        scope='c-fc21')

      '''
      We chose sigmoid instead of softmax since we only have two classes
      '''
      # last fully connected layer
      fc3 = kh.layers.fully_connected(inputs=fc2,
                                      num_outputs=NUM_CLASSES,
                                      activation_fn='sigmoid',
                                      scope='c-fc3')
    return fc3
