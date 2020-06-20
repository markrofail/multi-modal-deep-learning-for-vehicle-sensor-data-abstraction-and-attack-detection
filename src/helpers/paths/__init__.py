from pathlib import Path

import numpy as np


def _find_ROOT(abs_path):
  while abs_path.name != 'src':
    abs_path = abs_path.parent
  return abs_path.parent


def similar_files(path, as_int=False, basename_only=False):
  pattern = '*' + str(path.suffix)
  filenames = list(path.parent.glob(pattern))

  filenames = np.array(filenames)
  filenames = np.sort(filenames)

  if len(filenames) > 0:
    if basename_only:
      path_to_name = np.vectorize(lambda x: x.name)
      filenames = path_to_name(filenames)
    elif as_int:
      path_to_name = np.vectorize(lambda x: x.stem)
      filenames = path_to_name(filenames).astype(int)

  return filenames


def make_subfolder(path, subfolder):
  """helper method that puts given file in given subfolder"""

  basename, dirname = path.name, path.parent
  path = dirname.joinpath(subfolder, basename)

  path.parent.mkdir(exist_ok=True, parents=True)
  return path


def get_all_dates():
  path = DATA_EXTERNAL_PATH.joinpath('KITTI')
  assert path.exists(), 'No Dataset Found'

  dates = list()
  for current_file in path.iterdir():
    if current_file.is_dir():
      filename = current_file.name.split('_')
      if len(filename) >= 3:
        dates.append(current_file.name)
  return np.sort(dates)

def get_all_drives(drive_date, exclude=[]):
  path = depth.external_frame(drive_date, 0, 0).parents[3]
  if not path.exists():
    return list()

  drives = list()
  for current_file in path.iterdir():
    if current_file.is_dir():
      drive = current_file.name.split('_')

      if len(drive) < 4:
        continue

      drive = int(drive[4])
      if drive not in exclude:
        drives.append(drive)

  drives = np.sort(drives)
  drive_dates = np.tile(np.array([drive_date]), len(drives))

  return list(zip(drive_dates, drives))

def get_all_frames(drive_date, drive):
  path = depth.external_frame(drive_date, drive, 0)
  path_dir = path.parents[3]
  if not path_dir.exists():
    return list()

  frames = similar_files(path, as_int=True)
  frames = np.sort(frames)

  drive_dates = np.tile(np.array([drive_date]), len(frames))
  drives = np.full_like(frames, fill_value=drive)

  return list(zip(drive_dates, drives, frames))

ROOT_PATH = _find_ROOT(Path(__file__).absolute())
DATA_PATH = ROOT_PATH.joinpath('dataset')

DATA_EXTERNAL_PATH = DATA_PATH.joinpath('external')
DATA_RAW_PATH = DATA_PATH.joinpath('raw')
DATA_INTERIM_PATH = DATA_PATH.joinpath('interim')
DATA_PROCESSED_PATH = DATA_PATH.joinpath('processed')


from . import (attack, calibration, checkpoints, depth, logs, mask, models,
               config, reports, rgb, tfrecord, sanity) # noqa

__all__ = ['attack', 'calibration', 'checkpoints', 'depth', 'logs', 'mask', 'models',
           'config', 'reports', 'rgb', 'tfrecord', 'sanity']
