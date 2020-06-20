import glob

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K

from src.helpers import paths


###############################################################################
# DATA PARAMETERS                                                             #
###############################################################################
config = paths.config.read(paths.config.regnet())
IMAGE_WIDTH = int(config['DATA_INFORMATION']['IMAGE_WIDTH'])
IMAGE_HEIGHT = int(config['DATA_INFORMATION']['IMAGE_HEIGHT'])
INPUT_RGB_SHAPE = [IMAGE_HEIGHT, IMAGE_WIDTH, 3]
INPUT_RGB_SHAPE_PROD = [np.prod(INPUT_RGB_SHAPE)]

INPUT_DEPTH_SHAPE = [IMAGE_HEIGHT, IMAGE_WIDTH, 1]
INPUT_DEPTH_SHAPE_PROD = [np.prod(INPUT_DEPTH_SHAPE)]

DUAL_QUATERNIONS = config['DATA_INFORMATION']['DUAL_QUATERNIONS']
if DUAL_QUATERNIONS:
  LABEL_CALIB_SHAPE = [8]
else:
  LABEL_CALIB_SHAPE = [4, 4]
LABEL_CALIB_SHAPE_PROD = [np.prod(LABEL_CALIB_SHAPE)]


def get_train_batches(**kwargs):
  """Return train batch generators (rgb, depth)"""
  return get_batches(dataset='train', **kwargs)


def get_valid_batches(**kwargs):
  """Return valid batch generators (rgb, depth)"""
  return get_batches(dataset='valid', **kwargs)


def get_test_batches(**kwargs):
  """Return test batch generators (rgb, depth)"""
  return get_batches(dataset='test', **kwargs)


def get_batches(dataset, batch_size=1, infinite=True):
  """
  Create a generator that returns batches of tuples
  rgb, depth and calibration
  :param date: date of the drive
  :param drives: array of the drive_numbers within the drive date
  :return: batch generator
  """

  filenames = get_dataset_tensors(dataset)
  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.map(input_parser)  # Parse the record into tensors.
  if infinite:
    dataset = dataset.repeat()  # Repeat the input indefinitely.
  dataset = dataset.batch(batch_size)

  return dataset


def make_iterator(dataset):
  iterator = dataset.make_one_shot_iterator()
  next_val = iterator.get_next()

  with K.get_session().as_default() as sess:
    while True:
      *inputs, labels = sess.run(next_val)
      yield inputs, labels


def get_dataset_tensors(dataset):
  path = paths.DATA_PROCESSED_PATH.joinpath('KITTI')
  path = path.joinpath(path, dataset)
  if not path.exists():
    raise Exception('the {} dataset was not found'.format(dataset.upper()))

  pattern = path.joinpath('*.tfr')
  filenames = glob.glob(str(pattern))

  filenames = np.array(filenames)
  filenames = np.sort(filenames)
  return filenames


def input_parser(example_proto):
  features = {'data_rgb': tf.FixedLenFeature(INPUT_RGB_SHAPE_PROD, tf.float32),
              'data_depth': tf.FixedLenFeature(INPUT_DEPTH_SHAPE_PROD, tf.float32),
              'data_decalib': tf.FixedLenFeature(LABEL_CALIB_SHAPE_PROD, tf.float32)}
  parsed_features = tf.parse_single_example(example_proto, features)

  data_rgb = parsed_features['data_rgb']
  data_rgb = tf.cast(data_rgb, tf.float32)
  data_rgb.set_shape(INPUT_RGB_SHAPE_PROD)
  data_rgb = tf.reshape(data_rgb, INPUT_RGB_SHAPE)

  data_depth = parsed_features['data_depth']
  data_depth = tf.cast(data_depth, tf.float32)
  data_depth.set_shape(INPUT_DEPTH_SHAPE_PROD)
  data_depth = tf.reshape(data_depth, INPUT_DEPTH_SHAPE)

  data_decalib = parsed_features['data_decalib']
  data_decalib = tf.cast(data_decalib, tf.float32)
  data_decalib.set_shape(LABEL_CALIB_SHAPE_PROD)

  return data_rgb, data_depth, data_decalib
