import tensorflow as tf


def _floats_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value.flatten()))


def write_record(output_path, features):
  feature = {}
  for key, value, data_type in features:
    if data_type == 'float':
      feature[key] = _floats_feature(value)
    elif data_type == 'int':
      feature[key] = _int64_feature(value)

  sample = tf.train.Example(features=tf.train.Features(feature=feature))
  with tf.python_io.TFRecordWriter(output_path) as writer:
    writer.write(sample.SerializeToString())
