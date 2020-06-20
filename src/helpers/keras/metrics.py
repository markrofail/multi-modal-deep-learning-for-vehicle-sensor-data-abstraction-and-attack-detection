import functools

import tensorflow as tf
from tensorflow.python.keras import backend as K


def as_keras_metric(method):
  @functools.wraps(method)
  def wrapper(self, args, **kwargs):
    """ Wrapper for turning tensorflow metrics into keras metrics """
    value, update_op = method(self, args, **kwargs)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([update_op]):
      value = tf.identity(value)
    return value
  return wrapper


def euclidean_loss(y_true, y_pred):
  """
  Adds a Euclidean loss to the training procedure.
  :param y_true: The ground truth output tensorr, same dimensions as 'labels'
  :param y_pred: The output from the neural network
  :return: Weighted loss float Tensor
  """

  '''
  Used Caffe docs as refrence and the problem was already refrenced in a
  stackoverflow question

  http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1EuclideanLossLayer.html
  https://stackoverflow.com/questions/45028222/caffe-euclideanloss-reproduce-in-tensorflow
  '''
  return K.mean(K.square(y_pred - y_true), axis=-1, keepdims=True) / 2


def r2_score(y_true, y_pred):
  """
  Adds coefficient of determination metric
  :param y_true: The ground truth output tensorr, same dimensions as 'labels'
  :param y_pred: The output from the neural network
  :return: Weighted loss float Tensor
  """

  '''
  https://jmlb.github.io/ml/2017/03/20/CoeffDetermination_CustomMetric4Keras/
  '''
  SS_res = K.sum(K.square(y_true-y_pred))
  SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
  return 1 - SS_res/(SS_tot + K.epsilon())


@as_keras_metric
def precision(y_true, y_pred, thresholds=[0.5]):
  '''Calculates the precision, a metric for multi-label classification of
  how many selected items are relevant.
  '''
  return tf.metrics.precision_at_thresholds(y_true, y_pred, thresholds)

  # '''
  # https://github.com/keras-team/keras/blob/2b51317be82d4420169d2cc79dc4443028417911/keras/metrics.py
  # '''
  # true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  # predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
  # precision = true_positives / (predicted_positives + K.epsilon())
  # return precision


@as_keras_metric
def recall(y_true, y_pred, thresholds=[0.5]):
  '''Calculates the recall, a metric for multi-label classification of
  how many relevant items are selected.
  '''
  return tf.metrics.recall_at_thresholds(y_true, y_pred, thresholds)

  # '''
  # https://github.com/keras-team/keras/blob/2b51317be82d4420169d2cc79dc4443028417911/keras/metrics.py
  # '''
  # true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  # possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  # recall = true_positives / (possible_positives + K.epsilon())
  # return recall


def fbeta_score(y_true, y_pred, beta):
  '''Calculates the F score, the weighted harmonic mean of precision and recall.
  This is useful for multi-label classification, where input samples can be
  classified as sets of labels. By only using accuracy (precision) a model
  would achieve a perfect score by simply assigning every class to every
  input. In order to avoid this, a metric should penalize incorrect class
  assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
  computes this, as a weighted mean of the proportion of correct class
  assignments vs. the proportion of incorrect class assignments.
  With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
  correct classesas_keras_metric becomes more important, and with beta > 1 the metric is
  instead weighted towards penalizing incorrect class assignments.
  '''

  '''
  https://github.com/keras-team/keras/blob/2b51317be82d4420169d2cc79dc4443028417911/keras/metrics.py
  '''
  if beta < 0:
    raise ValueError('The lowest choosable beta is zero (only precision).')

  # If there are no true positives, fix the F score at 0 like sklearn.
  if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
    return 0

  p = precision(y_true, y_pred)
  r = recall(y_true, y_pred)
  bb = beta ** 2
  fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
  return fbeta_score


def fmeasure(y_true, y_pred):
  '''Calculates the f-measure, the harmonic mean of precision and recall.
  '''

  '''
  https://github.com/keras-team/keras/blob/2b51317be82d4420169d2cc79dc4443028417911/keras/metrics.py
  '''
  return fbeta_score(y_true, y_pred, beta=1)


@as_keras_metric
def auc_roc(y_true, y_pred):
  return tf.metrics.auc(y_true, y_pred)


@as_keras_metric
def true_positives(y_true, y_pred, thresholds=[0.50]):
  return tf.metrics.true_positives_at_thresholds(y_true, y_pred, thresholds)


@as_keras_metric
def true_negatives(y_true, y_pred, thresholds=[0.50]):
  return tf.metrics.true_negatives_at_thresholds(y_true, y_pred, thresholds)


@as_keras_metric
def false_positives(y_true, y_pred, thresholds=[0.50]):
  return tf.metrics.false_positives_at_thresholds(y_true, y_pred, thresholds)


@as_keras_metric
def false_negatives(y_true, y_pred, thresholds=[0.50]):
  return tf.metrics.false_negatives_at_thresholds(y_true, y_pred, thresholds)
