import multiprocessing

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from src.helpers import paths, tfrecord
from src.helpers.flags import AttackModes, Verbose


def _get_input_path(drive_date, drive_number, frame, attack):
  if attack:
    return paths.attack.processed_tensor(drive_date, drive_number, frame)
  return paths.rgb.processed_tensor(drive_date, drive_number, frame)


def _get_output_path(dataset, drive_date, drive_number, frame, attack, attack_type):
  attack_str = str(AttackModes(attack_type).name).lower()[:3]
  if attack:
    suffix = '.{}.atk'.format(attack_str)
  else:
    suffix = '.{}.nrm'.format(attack_str)
  return paths.tfrecord.dataset_frame(dataset, drive_date, drive_number, frame,
                                      extra_suffix=suffix)


def _generate_frame(dataset, drive_date, drive_number, frame, attack=False, attack_type=None,
                    keep=False):
  test_path = paths.depth.processed_tensor(drive_date, drive_number, frame)
  if not test_path.exists():
    return

  # load rgb
  path_rgb = _get_input_path(drive_date, drive_number, frame, attack)
  data_rgb = np.load(path_rgb)

  # load depthmaps
  path_depth = paths.depth.processed_tensor(drive_date, drive_number, frame)
  data_depth = np.load(path_depth)

  # load label
  data_label = np.array([0, 1], np.int64) if attack else np.array([1, 0], np.int64)

  output_path = _get_output_path(dataset, drive_date, drive_number, frame, attack, attack_type)
  output_path.parent.mkdir(exist_ok=True, parents=True)  # ensure directory exists

  # write record
  features = [
      ('data_rgb', data_rgb, 'float'),
      ('data_depth', data_depth, 'float'),
      ('data_label', data_label, 'float'),
  ]
  tfrecord.write_record(output_path=str(output_path), features=features)

  if not keep:
    if path_rgb.exists():
      path_rgb.unlink()
    if path_depth.exists() and attack:
      path_depth.unlink()


def _generate_frames(*args, attack_type, **kargs):
  _generate_frame(*args, **kargs, attack=False, attack_type=attack_type)
  _generate_frame(*args, **kargs, attack=True, attack_type=attack_type)


def generate_frames(frames, dataset, attack_type=None, verbose=Verbose.NORMAL, keep=False):
  if verbose > Verbose.SILENT:
    info = '# generating tfrecords      '  # for logging purposes
    frames = tqdm(frames, ascii=True, desc=info)

  n_jobs = multiprocessing.cpu_count() // 2
  Parallel(n_jobs=n_jobs)(
      delayed(_generate_frames)(dataset, *frame_info, attack_type=attack_type, keep=keep)
      for frame_info in frames)
