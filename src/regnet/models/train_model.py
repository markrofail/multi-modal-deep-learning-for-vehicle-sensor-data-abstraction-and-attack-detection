import os
import pickle

import click
from tensorflow.python import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks

import src.helpers.keras as kh
from src.helpers import paths, timeit
from src.helpers.keras import early_stopping
from src.regnet import regnet
from src.regnet.features import build_features
from src.regnet.models import load_weights

###############################################################################
# HYPER PARAMETERS
###############################################################################
config = paths.config.read(paths.config.regnet())
BETA1 = float(config['HYPERPARAMETERS']['BETA1'])
BETA2 = float(config['HYPERPARAMETERS']['BETA2'])
EPOCHS = int(float(config['HYPERPARAMETERS']['EPOCHS']))
EPSILON = float(config['HYPERPARAMETERS']['EPSILON'])
PATIENCE = config['HYPERPARAMETERS']['PATIENCE']
BASE_LINE = config['HYPERPARAMETERS']['BASE_LINE']
BATCH_SIZE = int(config['HYPERPARAMETERS']['BATCH_SIZE'])
LEARNING_RATE = float(config['HYPERPARAMETERS']['LEARNING_RATE'])

# obtain frames
one_frame = paths.tfrecord.dataset_frame(dataset='train', drive_date='', drive_number=0, frame=0)
NUM_BATCHES = len(paths.similar_files(path=one_frame))

###############################################################################
# DATA PARAMETERS
###############################################################################
ITERATIVE_REFINEMENT = config['DECALIBRATIONS']['ITERATIVE_REFINEMENT']
ROT = config['DECALIBRATIONS']['RANGE_ROTATION'] if ITERATIVE_REFINEMENT else None
DISP = config['DECALIBRATIONS']['RANGE_DISPLACEMENT'] if ITERATIVE_REFINEMENT else None

###############################################################################
# ENVIROMENTAL VARIABLES
###############################################################################
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
VERBOSE = config['ENVIROMENT_CONFIG']['VERBOSE']
SAVE_EVERY = int(config['ENVIROMENT_CONFIG']['SAVE_EVERY'])

CUSTOM_LAYERS = dict([
    ('euclidean_loss', kh.metrics.euclidean_loss),
    ('r2_score', kh.metrics.r2_score),
    # ('y_true', kh.metrics.y_true),
    # ('y_pred', kh.metrics.y_pred),
])


def run(epochs, num_batches, batch_size=1, learning_rate=0.001, beta1=0.9, beta2=0.999,
        epsilon=1e-08, save_every=10, patience=5, baseline=2e-5, resume=False):
    # Destroy old graph
  K.clear_session()

  # Initialize batch generators
  batch_train = build_features.get_train_batches(batch_size=batch_size)
  batch_valid = build_features.get_valid_batches(batch_size=batch_size)

  # Create TensorFlow Iterator object
  itr_train = build_features.make_iterator(batch_train)
  itr_valid = build_features.make_iterator(batch_valid)

  # Init callbacks
  cbs = list()

  # EarlyStopping callback: stops whenever loss doesn't imporve
  # cbs.append(early_stopping.EarlyStopping(monitor='val_loss', mode='min', patience=patience,
  #                                         verbose=1, baseline=baseline))

  # ModelCheckpoint callback: saves model every SAVE_EVERY
  save_path = paths.checkpoints.regnet(rot=ROT, disp=DISP)  # ./checkpoints/regnet/train
  save_path.parent.mkdir(exist_ok=True, parents=True)
  if save_path.exists() and not resume:
    save_path.unlink()  # deletes file before training
  cbs.append(callbacks.ModelCheckpoint(str(save_path), save_best_only=True, period=save_every))

  # TensorBoard callback: saves logs for tensorboard
  log_path = str(paths.logs.regnet())  # ./logs/regnet/train
  cbs.append(callbacks.TensorBoard(log_dir=log_path, batch_size=batch_size, write_graph=True))

  # History callback: saves all losses
  cbs.append(callbacks.CSVLogger(save_path.with_suffix('.csv'), append=True, separator=','))

  # Create the network
  net = regnet.Regnet(learning_rate, beta1, beta2, epsilon)

  # Configures the model for training
  net.model.compile(optimizer=net.train_opt, loss=net.model_loss, metrics=net.metrics)

  # Load the pretrained imagenet weights
  load_weights.imagenet_weights(net.model)

  if resume:
    net.model = keras.models.load_model(save_path, custom_objects=CUSTOM_LAYERS, compile=True)

  # Train network
  net.model.fit_generator(
      generator=itr_train, validation_data=itr_valid, validation_steps=batch_size,
      epochs=epochs, steps_per_epoch=num_batches, callbacks=cbs, verbose=1, workers=0)


def evaluate(batch_size=1, dataset='valid'):
  assert dataset in ['valid', 'test'], 'dataset has to be one of \'valid\' or \'test\''

  # Destroy old graph
  K.clear_session()

  # Initialize validation batch generator
  if dataset == 'valid':
    batch_dataset = build_features.get_valid_batches(batch_size=batch_size, infinite=False)
  elif dataset == 'test':
    batch_dataset = build_features.get_test_batches(batch_size=batch_size, infinite=False)
  itr_dataset = batch_dataset.make_one_shot_iterator()

  # Create the network
  *inputs, labels = itr_dataset.get_next()
  net = regnet.Regnet(hasPipeline=True, pipeline_inputs=inputs, pipeline_labels=labels)

  # Load pretrained model
  model_path = paths.checkpoints.regnet(rot=ROT, disp=DISP)  # ./checkpoints/regnet/train
  net.model.load_weights(str(model_path))

  # Configures the model for evaluation
  net.model.compile(optimizer=net.train_opt, loss=net.model_loss, target_tensors=[net.label])

  # Evaluate
  valid_files = build_features.get_dataset_tensors(dataset)
  print('feed forward on {} {} frames'.format(len(valid_files), dataset))

  loss = net.model.evaluate(steps=len(valid_files))
  loss = zip(net.model.metrics_names, loss)

  for label, value in loss:
    value = np.array([value])
    print('{} {}: {}'.format(dataset, label, value))


@click.command()
@click.option(
    '--epochs',
    type=int,
    default=EPOCHS,
    help='number of epochs to train'
)
@click.option(
    '--resume',
    is_flag=True,
    help='resume training',
)
@timeit.logtime
def main(epochs=EPOCHS, resume=False):
  if NUM_BATCHES > BATCH_SIZE:
    num_batches = NUM_BATCHES // BATCH_SIZE
  else:
    num_batches = NUM_BATCHES

  print('\n# Training for {} epoch(s)...'.format(epochs))
  run(
      beta1=BETA1,
      beta2=BETA2,
      epsilon=EPSILON,
      patience=PATIENCE,
      baseline=BASE_LINE,
      batch_size=BATCH_SIZE,
      save_every=SAVE_EVERY,
      learning_rate=LEARNING_RATE,
      resume=resume,
      epochs=epochs,
      num_batches=num_batches,
  )

  print('\n\n# Running evaluation algorithm...')
  evaluate(dataset='valid')
  evaluate(dataset='test')


if __name__ == '__main__':
  main()
