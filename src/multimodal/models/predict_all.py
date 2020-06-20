import csv

import click
import numpy as np
from tqdm import tqdm

from src.helpers import paths
from src.helpers.flags import AttackModes, Verbose
from src.multimodal import multimodal
from src.multimodal.data import make_dataset
from src.multimodal.features import build_features
from src.regnet.data.make_dataset import cleanUp

INP_NRM_LOG = None
TRA_NRM_LOG = None
INP_ATK_LOG = None
TRA_ATK_LOG = None

###############################################################################
# HYPERPARAMETERS
###############################################################################
config = paths.config.read(paths.config.multimodal())
BATCH_SIZE = config['HYPERPARAMETERS']['BATCH_SIZE']

# @click.command()
# def main():
#   train_model.evaluate(dataset='valid')
#   print()
#   train_model.evaluate(dataset='test')

def get_all_dates():
  path = paths.DATA_EXTERNAL_PATH.joinpath('KITTI')
  assert path.exists(), 'No Dataset Found'

  dates = list()

  for current_file in path.iterdir():
    if current_file.is_dir():
      filename = current_file.name.split('_')
      if len(filename) >= 3:
        dates.append(current_file.name)
  return np.sort(dates)

def get_all_drives(drive_date):
  path = paths.depth.external_frame(drive_date, 0, 0).parents[3]
  if not path.exists():
    return list()

  drives = list()
  for current_file in path.iterdir():
    if current_file.is_dir():
      drive = current_file.name.split('_')

      if len(drive) < 4:
        continue

      drive = int(drive[4])
      drives.append(drive)

  drives = np.sort(drives)
  drive_info = list()
  for drive in drives:
    drive_info.append((drive_date, drive))

  return drive_info

def get_all_frames(drive_date, drive):
  path = paths.depth.external_frame(drive_date, drive, 0)
  path_dir = path.parents[3]
  if not path_dir.exists():
    return list()

  frames = paths.similar_files(path, as_int=True)
  frames = np.sort(frames)

  frame_info = list()
  for frame in frames:
    frame_info.append((drive_date, int(drive), int(frame)))

  return frame_info


def load_pretrained_model():
  # Create the network
  net = multimodal.Multimodal()

  # Load pretrained model
  model_path = str(paths.checkpoints.multimodal())  # ./checkpoints/multimodal/train
  net.model.load_weights(model_path)

  return net.model


def init_logs():
  output_path = paths.DATA_PROCESSED_PATH.joinpath('logs')
  output_path.mkdir(exist_ok=True, parents=True)  # ensure directory exists

  global INP_NRM_LOG, TRA_NRM_LOG
  global INP_ATK_LOG, TRA_ATK_LOG

  INP_NRM_LOG = output_path.joinpath('inp_nrm.csv')
  TRA_NRM_LOG = output_path.joinpath('tra_nrm.csv')
  INP_ATK_LOG = output_path.joinpath('inp_atk.csv')
  TRA_ATK_LOG = output_path.joinpath('tra_atk.csv')

  logs = [
      INP_NRM_LOG,
      TRA_NRM_LOG,
      INP_ATK_LOG,
      TRA_ATK_LOG,
  ]

  csv_data = [['drive_date', 'drive', 'frame', 'result']]
  for log in logs:
    with open(log, 'w') as csv_file:
      writer = csv.writer(csv_file)
      writer.writerows(csv_data)


@click.command()
def main(batch_size=BATCH_SIZE):
  print('# Initiating logs...')
  init_logs()

  print('# Getting all frames...')
  drive_dates = get_all_dates()

  drives_info = list()
  for drive_date in drive_dates:
    drives_info.extend(get_all_drives(drive_date))

  frames_info = list()
  for drive_date, drive in drives_info:
    frames_info.extend(get_all_frames(drive_date, drive))

  print('# Loading model...')
  model = load_pretrained_model()

  num_batches = int(len(frames_info)//batch_size)

  for i in tqdm(range(num_batches), desc='# Testing all frames', ascii=True):
    batch = frames_info[i*batch_size: min((i+1)*batch_size, len(frames_info))]

    feed_forward(model, batch, attack_type=AttackModes.INPAINT)
    feed_forward(model, batch, attack_type=AttackModes.TRANSLATE)


def feed_forward(model, batch, attack_type, batch_size=BATCH_SIZE):
  # Generate Data
  make_dataset.make_data(batch, name='test', attack_type=attack_type,
                         verbose=Verbose.SILENT, keep=False)

  # Load Data
  batch_nrm = build_features.get_test_batches(batch_size=batch_size, infinite=False, attack=False)
  itr_nrm = build_features.make_iterator(batch_nrm)

  batch_atk = build_features.get_test_batches(batch_size=batch_size, infinite=False, normal=False)
  itr_atk = build_features.make_iterator(batch_atk)

  # Predict
  pred_nrm = model.predict_generator(generator=itr_nrm, steps=1, workers=0)
  pred_nrm = np.argmax(pred_nrm, axis=1)

  pred_atk = model.predict_generator(generator=itr_atk, steps=1, workers=0)
  pred_atk = np.argmax(pred_atk, axis=1)

  # Log Restults
  if attack_type == AttackModes.INPAINT:
    log_atk = INP_ATK_LOG
    log_nrm = INP_NRM_LOG
  elif attack_type == AttackModes.TRANSLATE:
    log_atk = TRA_ATK_LOG
    log_nrm = TRA_NRM_LOG

  batch = np.array(batch)
  batch = batch.T

  data_atk = zip(batch[0],
                 np.array(batch[1]).astype(int),
                 np.array(batch[2]).astype(int),
                 pred_atk)
  with open(log_atk, 'a') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(data_atk)

  data_nrm = zip(batch[0],
                 np.array(batch[1]).astype(int),
                 np.array(batch[2]).astype(int),
                 pred_nrm)
  with open(log_nrm, 'a') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(data_nrm)

  cleanUp(verbose=Verbose.SILENT, force_all=True)


if __name__ == '__main__':
  main()
