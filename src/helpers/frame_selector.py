import csv
import glob
import random
from pathlib import Path

import numpy as np

from src.helpers import paths


def _print_all_files():
    # create pattern
  pattern = paths.DATA_EXTERNAL_PATH.joinpath('KITTI', '*', '*', 'image_02', 'data', '*.png')

  # find the files
  filenames = glob.glob(str(pattern))
  return filenames


def _random_choose(filenames, frame_count):
  random_files = set()

  # use set to ensure no duplicates
  while len(random_files) < frame_count:
    random_files.add(random.choice(filenames))
  return list(random_files)


def _file_to_frame(path, result, debug=False):
  path = Path(path)
  # ./data/external/KITTI/*/*/image_02/data/*.png

  # get the drive string (e.g. 2011_09_26_drive_0001_sync)
  drive_str = path.parents[2].name
  drive_str = drive_str.split('_')

  # get the file name - the extension
  frame = int(path.stem)

  # get the drive number from the drive string (e.g. 2011_09_26_drive_⁰⁰⁰¹_sync)
  drive = int(drive_str[-2])

  # get the drive date from the drive string (e.g. ²⁰¹¹ ⁰⁹ ²⁶_drive_0001_sync)
  date = drive_str[:3]
  date = '_'.join(date)

  if debug:
    print('date: {date}, drive: {drive}, frame: {frame}'.
          format(frame=frame, drive=drive, date=date))
  result.append((date, drive, frame))


def _split_frames(frames, frame_count, train_pct, valid_pct, test_pct):
  # randomply shuflle the dataset
  np.random.shuffle(frames)

  # compute the boundries of each dataset
  train_len = int(train_pct * frame_count)
  valid_len = int(valid_pct * frame_count)
  test_len = frame_count - (train_len + valid_len)

  # perform the split
  train, valid, test = frames[:train_len], frames[train_len:valid_len +
                                                  train_len], frames[-test_len:]

  return train, valid, test


def _record_files(datasets, csv_directory=None):
  datasets = zip(datasets, ['train', 'valid', 'test'])

  csv_title = ['drive_date', 'drive_number', 'frame']
  if csv_directory is None:
    csv_directory = paths.checkpoints.CHECKPOINT_PATH.joinpath('dataset')

  for dataset, filename in datasets:
    csv_data = np.concatenate(([csv_title], dataset))
    csv_path = csv_directory.joinpath('{}_frames.csv'.format(filename))

    csv_path.parent.mkdir(exist_ok=True, parents=True)
    with open(csv_path, 'w') as csv_file:
      writer = csv.writer(csv_file)
      writer.writerows(csv_data)


def generate_random(frame_count, train_pct, valid_pct, test_pct, log_path=None):
  # print all available files
  all_files = _print_all_files()

  # choose a subset randomly
  rand_files = _random_choose(all_files, frame_count)

  # map file paths to frame information
  rand_frames = list()
  map_frames = np.vectorize(lambda x: _file_to_frame(x, rand_frames))
  map_frames(rand_files)

  # split the dataset into test, valid and train
  datasets = _split_frames(rand_frames, frame_count, train_pct, valid_pct, test_pct)

  # record the frame information
  _record_files(datasets, log_path)
  return datasets


def read_frames(csv_directory):
  datasets = list()
  filenames = ['train', 'valid', 'test']

  for filename in filenames:
    csv_path = csv_directory.joinpath('{}_frames.csv'.format(filename))

    with open(csv_path, 'r') as csv_file:
      reader = csv.reader(csv_file)
      csv_data = list(reader)
      csv_data.pop(0)  # remove header data
    datasets.append([tuple([row[0], int(row[1]), int(row[2])]) for row in csv_data])

  return datasets
