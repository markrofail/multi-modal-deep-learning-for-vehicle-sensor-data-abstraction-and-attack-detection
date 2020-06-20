import tensorflow as tf


__all__ = ['placeholder', 'convolution2d', 'max_pool2d', 'flatten', 'fully_connected',
           'concatenate', 'pipeline']


def placeholder(shape, name=None):
  return tf.keras.layers.Input(shape=shape, name=name)


def pipeline(tensor, name=None):
  return tf.keras.layers.Input(tensor=tensor, name=name)


def fully_connected(inputs, num_outputs, activation_fn=None, scope=None):
  return tf.keras.layers.Dense(units=num_outputs,
                               activation=activation_fn,
                               use_bias=True,
                               kernel_initializer='glorot_uniform',
                               bias_initializer='zeros',
                               name=scope)(inputs)


def convolution2d(inputs, num_outputs, kernel_size, stride=1,
                  padding='valid', activation_fn=None, scope=None):
  return tf.keras.layers.Conv2D(filters=num_outputs,
                                kernel_size=kernel_size,
                                strides=(stride, stride),
                                padding=padding,
                                activation=activation_fn,
                                use_bias=True,
                                kernel_initializer='glorot_uniform',
                                bias_initializer='zeros',
                                name=scope)(inputs)


def max_pool2d(inputs, kernel_size, stride=2, padding='valid', scope=None):
  return tf.keras.layers.MaxPooling2D(pool_size=(kernel_size, kernel_size),
                                      strides=stride,
                                      padding=padding,
                                      name=scope)(inputs)


def concatenate(values, axis=-1, name=None):
  return tf.keras.layers.Concatenate(axis=axis, name=name)(list(values))


def flatten(inputs, name=None):
  return tf.keras.layers.Flatten(name=name)(inputs)


def reshape(inputs, shape, name=None):
  return tf.keras.layers.Reshape(target_shape=shape, name=name)(inputs)
