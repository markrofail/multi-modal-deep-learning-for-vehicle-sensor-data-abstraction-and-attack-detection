import csv
from shutil import copyfile

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.helpers import paths
from src.helpers.flags import AttackModes, Verbose
from src.multimodal import multimodal
from src.multimodal.data import make_dataset
from src.multimodal.features import build_features
from src.regnet.data.make_dataset import cleanUp
from src.multimodal.models import train_model

@click.command()
def main():
  model_checkpoints = [
      '1024_1024_50epochs.hdf5',
      '2048_1024_50epochs.hdf5',
      '2048_1024_512_50epochs.hdf5',
      '2048_2048_50epochs.hdf5'
  ]

  root = paths.ROOT_PATH.parent.joinpath('MultimodalCheckpoints')
  dst = paths.checkpoints.multimodal()

  make_data()

  for checkpoint in model_checkpoints:
    src = root.joinpath(checkpoint)
    copyfile(src, dst)

    print(checkpoint)
    run_pred()
    run_eval()


def make_data():
  print('# Getting all frames...')
  frames_info = get_all_frames('2011_09_30', 28)

  for frame in tqdm(frames_info, desc='Generating Data', ascii=True):
    make_dataset.make_data([frame], name='test', attack_type=AttackModes.INPAINT,
                           verbose=Verbose.SILENT, keep=False)
    make_dataset.make_data([frame], name='test', attack_type=AttackModes.TRANSLATE,
                           verbose=Verbose.SILENT, keep=False)


def run_pred():
  print('# Initiating logs...')
  output_path = paths.DATA_PROCESSED_PATH.joinpath('logs')
  output_path.mkdir(exist_ok=True, parents=True)  # ensure directory exists

  inp_log = output_path.joinpath('inp.csv')
  tra_log = output_path.joinpath('tra.csv')
  nrm_log = output_path.joinpath('nrm.csv')

  print('# Loading model...')
  model = load_pretrained_model()

  frames_info = get_all_frames('2011_09_30', 28)

  print('# Predicting on Normal data...')
  predict_and_log(model, frames_info, log_file=nrm_log, attack=False)

  print('# Predicting on Inpainting Attacks...')
  predict_and_log(model, frames_info, log_file=tra_log, attack=True, attack_type=True)

  print('# Predicting on Translation Attacks...')
  predict_and_log(model, frames_info, log_file=inp_log, attack=True, attack_type=False)


def predict_and_log(model, batch, log_file, attack=False, attack_type=False):
  # Load Data
  batch_test = build_features.get_test_batches(batch_size=1, infinite=False,
                                               attack=attack, normal=not attack,
                                               inpaint=attack_type, translate= not attack_type)
  itr_test = build_features.make_iterator(batch_test)

  # Predict
  pred = model.predict_generator(generator=itr_test, steps=len(batch), workers=0, verbose=1)

  # Log Restults
  batch = np.array(batch).T

  drive_dates = batch[0]
  drives = np.array(batch[1]).astype(int)
  frames = np.array(batch[2]).astype(int)

  pred = np.argmax(pred, axis=1)
  labels = np.ones_like(pred) if attack else np.zeros_like(pred)

  csv_data = [['drive_date', 'drive', 'frame', 'label', 'prediction']]
  csv_data.extend(zip(drive_dates, drives, frames, labels, pred))
  with open(log_file, 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(csv_data)


def get_all_frames(drive_date, drive):
  path = paths.depth.external_frame(drive_date, drive, 0)
  assert path.parents[3].exists(), 'Drive not found'

  frames = paths.similar_files(path, as_int=True)
  frames = np.sort(frames)

  frame_info = list()
  for frame in frames:
    frame_info.append((drive_date, int(drive), int(frame)))

  return frame_info


def load_pretrained_model():
  from tensorflow.python.keras.models import load_model
  model_path = str(paths.checkpoints.multimodal())
  return load_model(model_path, custom_objects=train_model.CUSTOM_LAYERS)


def run_eval():

  filenames = ['nrm', 'inp', 'tra']
  root_path = paths.DATA_PROCESSED_PATH.joinpath('logs')

  for filename in filenames:
    log_file = root_path.joinpath('{}.csv'.format(filename))
    max_0, max_1 = evaluate(log_file)

    print('{}: (max 0s: {}, max 1s: {})'.format(filename, max_0, max_1))

def evaluate(log_file):
  df = pd.read_csv(log_file)

  predicate = df.prediction.gt(0)
  counts = df.groupby([predicate, (predicate != predicate.shift()).cumsum()])
  counts = counts.size().rename_axis(['>0', 'grp'])

  max_running_0 = counts.loc[False].max()
  max_running_1 = counts.loc[True].max()

  return max_running_0, max_running_1


if __name__ == '__main__':
  main()
