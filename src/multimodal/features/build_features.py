import glob

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K

from src.helpers import paths


###############################################################################
# DATA PARAMETERS                                                             #
###############################################################################
config = paths.config.read(paths.config.multimodal())
IMAGE_WIDTH = int(config['DATA_INFORMATION']['IMAGE_WIDTH'])
IMAGE_HEIGHT = int(config['DATA_INFORMATION']['IMAGE_HEIGHT'])

INPUT_RGB_SHAPE = [IMAGE_HEIGHT, IMAGE_WIDTH, 3]
INPUT_RGB_SHAPE_PROD = [np.prod(INPUT_RGB_SHAPE)]

INPUT_DEPTH_SHAPE = [IMAGE_HEIGHT, IMAGE_WIDTH, 1]
INPUT_DEPTH_SHAPE_PROD = [np.prod(INPUT_DEPTH_SHAPE)]

LABEL_SHAPE = [2]


def get_train_batches(**kwargs):
  """Return train batch generators (rgb, depth)"""
  return get_batches(dataset='train', **kwargs)


def get_valid_batches(**kwargs):
  """Return valid batch generators (rgb, depth)"""
  return get_batches(dataset='valid', **kwargs)


def get_test_batches(**kwargs):
  """Return test batch generators (rgb, depth)"""
  return get_batches(dataset='test', **kwargs)


def get_batches(dataset, batch_size=1, infinite=True, attack=True, normal=True, inpaint=True, translate=True):
  """
  Create a generator that returns batches of tuples
  rgb, depth and calibration
  :param date: date of the drive
  :param drives: array of the drive_numbers within the drive date
  :return: batch generator
  """

  all_filenames = get_dataset_tensors(dataset)

  filenames = list()
  if inpaint:
    inp_frames = [x for x in all_filenames if 'inp' in x.lower()]
    if attack:
      attack_frames = [x for x in inp_frames if 'atk' in x.lower()]
      filenames.extend(attack_frames)
    if normal:
      normal_frames = [x for x in inp_frames if 'nrm' in x.lower()]
      filenames.extend(normal_frames)

  if translate:
    tra_frames = [x for x in all_filenames if 'tra' in x.lower()]
    if attack:
      attack_frames = [x for x in tra_frames if 'atk' in x.lower()]
      filenames.extend(attack_frames)
    if normal:
      normal_frames = [x for x in tra_frames if 'nrm' in x.lower()]
      filenames.extend(normal_frames)

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
              'data_label': tf.FixedLenFeature(LABEL_SHAPE, tf.float32)}
  parsed_features = tf.parse_single_example(example_proto, features)

  data_rgb = parsed_features['data_rgb']
  data_rgb = tf.cast(data_rgb, tf.float32)
  data_rgb.set_shape(INPUT_RGB_SHAPE_PROD)
  data_rgb = tf.reshape(data_rgb, INPUT_RGB_SHAPE)

  data_depth = parsed_features['data_depth']
  data_depth = tf.cast(data_depth, tf.float32)
  data_depth.set_shape(INPUT_DEPTH_SHAPE_PROD)
  data_depth = tf.reshape(data_depth, INPUT_DEPTH_SHAPE)

  data_label = parsed_features['data_label']
  data_label = tf.cast(data_label, tf.float32)
  data_label.set_shape(LABEL_SHAPE)

  return data_rgb, data_depth, data_label
