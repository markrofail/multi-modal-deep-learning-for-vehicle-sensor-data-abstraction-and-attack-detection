import tensorflow as tf

import src.regnet.regnet as regnet

NETWORK_INPUT_SHAPE = [1, 1216, 352, 3]
NETWORK_TARGET_SHAPE = [1, 16]


def test_regnet():
  """
  feed forward test for the entire Regnet
  """
  with tf.Graph().as_default():
    test_rgb_input = tf.placeholder(tf.float32, NETWORK_INPUT_SHAPE)
    test_depth_input = tf.placeholder(tf.float32, NETWORK_INPUT_SHAPE)
    output = regnet.init_network(test_rgb_input, test_depth_input)

    shape = output.shape.as_list()
    print("final output shape: {}".format(shape))

    assert output.shape.as_list() == NETWORK_TARGET_SHAPE, \
        'required shape: {}, found shape: {}'.format(NETWORK_TARGET_SHAPE,
                                                     output.shape.as_list())


REGRESSION_INPUT_SHAPE = [1, 9, 2, 512]
REGRESSION_TARGET_SHAPE = [1, 16]


def test_global_regression():
  """
  feed forward test for the regression module
  """
  with tf.Graph().as_default():
    test_input = tf.placeholder(tf.float32, REGRESSION_INPUT_SHAPE)
    output = regnet.global_regression(test_input)

    assert output.shape.as_list() == REGRESSION_TARGET_SHAPE, \
        'required shape: {}, found shape: {}'.format(REGRESSION_TARGET_SHAPE,
                                                     output.shape.as_list())


FEATURE_MATCHING_INPUT_SHAPE = [1, 38, 11, 576]
FEATURE_MATCHING_TARGET_SHAPE = [1, 10, 3, 512]
# FEATURE_MATCHING_TARGET_SHAPE = [1, 9, 2, 512]


def test_feature_matching():
  """
  feed forward test for the feature matching module
  """
  with tf.Graph().as_default():
    test_input = tf.placeholder(tf.float32, FEATURE_MATCHING_INPUT_SHAPE)
    output = regnet.feature_matching(test_input)

    assert output.shape.as_list() == FEATURE_MATCHING_TARGET_SHAPE, \
        'required shape: {}, found shape: {}'.format(FEATURE_MATCHING_TARGET_SHAPE,
                                                     output.shape.as_list())


FEATURE_EXTRACTION_INPUT_SHAPE = [1, 1216, 352, 3]
FEATURE_EXTRACTION_TARGET_SHAPE = [1, 38, 11, 576]


def test_feature_extraction():
  """
  feed forward test for the feature extraction module
  """
  with tf.Graph().as_default():
    test_rgb_input = tf.placeholder(tf.float32, FEATURE_EXTRACTION_INPUT_SHAPE)
    test_depth_input = tf.placeholder(tf.float32, FEATURE_EXTRACTION_INPUT_SHAPE)
    output = regnet.feature_extraction(test_rgb_input, test_depth_input)

    output = tf.concat(output, axis=3)

    assert output.shape.as_list() == FEATURE_EXTRACTION_TARGET_SHAPE, \
        'required shape: {}, found shape: {}'.format(FEATURE_EXTRACTION_TARGET_SHAPE,
                                                     output.shape.as_list())


RGB_FEATURE_EXTRACTION_INPUT_SHAPE = [1, 1216, 352, 3]
RGB_FEATURE_EXTRACTION_TARGET_SHAPE = [1, 38, 11, 384]


def test_rgb_feature_extraction():
  """
  feed forward test for the rgb feature extraction module
  """
  with tf.Graph().as_default():
    test_input = tf.placeholder(tf.float32, RGB_FEATURE_EXTRACTION_INPUT_SHAPE)
    output = regnet.rgb_feature_extraction(test_input)

    assert output.shape.as_list() == RGB_FEATURE_EXTRACTION_TARGET_SHAPE, \
        'required shape: {}, found shape: {}'.format(RGB_FEATURE_EXTRACTION_TARGET_SHAPE,
                                                     output.shape.as_list())


DEPTH_FEATURE_EXTRACTION_INPUT_SHAPE = [1, 1216, 352, 3]
DEPTH_FEATURE_EXTRACTION_TARGET_SHAPE = [1, 38, 11, 192]


def test_depth_feature_extraction():
  """
  feed forward test for the depth feature extraction module
  """

  with tf.Graph().as_default():
    test_input = tf.placeholder(tf.float32, DEPTH_FEATURE_EXTRACTION_INPUT_SHAPE)
    output = regnet.depth_feature_extraction(test_input)

    assert output.shape.as_list() == DEPTH_FEATURE_EXTRACTION_TARGET_SHAPE, \
        'required shape: {}, found shape: {}'.format(DEPTH_FEATURE_EXTRACTION_TARGET_SHAPE,
                                                     output.shape.as_list())


NIN_TEST_INPUT_SHAPE = [1, 1216, 352, 3]
NIN_TEST_KERNEL_SIZE = 11
NIN_TEST_FILTERS = 94
NIN_TEST_STRIDE = 2


def test_layers_nin():
  """
  a test method for the NiN block by comparing its output tensor
  to that of a normal CNN output tensor
  """

  with tf.Graph().as_default():
    test_input = tf.placeholder(tf.float32, NIN_TEST_INPUT_SHAPE)

    cnn_output = tf.contrib.layers.conv2d(inputs=test_input,
                                          kernel_size=NIN_TEST_KERNEL_SIZE,
                                          num_outputs=NIN_TEST_FILTERS,
                                          stride=NIN_TEST_STRIDE * 2,
                                          activation_fn=tf.nn.relu)

    nin_output = regnet.layers_nin(inputs=test_input,
                                   kernel_size=NIN_TEST_KERNEL_SIZE,
                                   num_outputs=NIN_TEST_FILTERS,
                                   stride=NIN_TEST_STRIDE,
                                   activation_fn=tf.nn.relu)

  # Check that NiN doesn't change output shape
  assert cnn_output.shape.as_list() == nin_output.shape.as_list(),\
      'CNN shape {}. NiN shape {}'.format(cnn_output.shape, nin_output.shape)
