import warnings

import numpy as np
from tensorflow.python.keras.callbacks import Callback


class EarlyStopping(Callback):
  """Stop training when a monitored quantity has stopped improving.
  # Arguments
      monitor: quantity to be monitored.
      min_delta: minimum change in the monitored quantity
          to qualify as an improvement, i.e. an absolute
          change of less than min_delta, will count as no
          improvement.
      patience: number of epochs with no improvement
          after which training will be stopped.
      verbose: verbosity mode.
      mode: one of {auto, min, max}. In `min` mode,
          training will stop when the quantity
          monitored has stopped decreasing; in `max`
          mode it will stop when the quantity
          monitored has stopped increasing; in `auto`
          mode, the direction is automatically inferred
          from the name of the monitored quantity.
      baseline: Baseline value for the monitored quantity to reach.
          Training will stop if the model doesn't show improvement
          over the baseline.
  """

  def __init__(self,
               monitor='val_loss',
               min_delta=0,
               patience=0,
               verbose=0,
               mode='auto',
               baseline=None):
    super(EarlyStopping, self).__init__()

    self.monitor = monitor
    self.baseline = baseline
    self.patience = patience
    self.verbose = verbose
    self.min_delta = min_delta
    self.wait = 0
    self.stopped_epoch = 0

    if mode not in ['auto', 'min', 'max']:
      warnings.warn('EarlyStopping mode %s is unknown, '
                    'fallback to auto mode.' % mode,
                    RuntimeWarning)
      mode = 'auto'

    if mode == 'min':
      self.monitor_op = np.less
    elif mode == 'max':
      self.monitor_op = np.greater
    else:
      if 'acc' in self.monitor:
        self.monitor_op = np.greater
      else:
        self.monitor_op = np.less

    if self.monitor_op == np.greater:
      self.min_delta *= 1
    else:
      self.min_delta *= -1

  def on_train_begin(self, logs=None):
    # Allow instances to be re-used
    self.wait = 0
    self.stopped_epoch = 0
    if self.baseline is not None:
      self.best = self.baseline
    else:
      self.best = np.Inf if self.monitor_op == np.less else -np.Inf

  def on_epoch_end(self, epoch, logs=None):
    current = logs.get(self.monitor)
    if current is None:
      warnings.warn(
          'Early stopping conditioned on metric `%s` '
          'which is not available. Available metrics are: %s' %
          (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
      )
      return
    if self.monitor_op(current - self.min_delta, self.best):
      self.best = current
      self.wait = 0
    else:
      self.wait += 1
      if self.wait >= self.patience:
        self.stopped_epoch = epoch
        self.model.stop_training = True

  def on_train_end(self, logs=None):
    if self.stopped_epoch > 0 and self.verbose > 0:
      print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
