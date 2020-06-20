import multiprocessing

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from src.helpers import paths, tfrecord
from src.helpers.flags import Verbose


def _generate_frame(dataset, drive_date, drive_number, frame, keep=False):
    # load rgb
  path_rgb = paths.rgb.processed_tensor(drive_date, drive_number, frame)
  if not path_rgb.exists():
    return
  data_rgb = np.load(path_rgb)

  # load depthmaps
  path_depth = paths.depth.processed_tensor(drive_date, drive_number, frame)
  data_depth = np.load(path_depth)

  # load label
  path_decalib = paths.calibration.decalibration_matrix(drive_date, drive_number, frame)
  data_decalib = np.load(path_decalib)
  data_decalib = data_decalib.flatten()

  output_path = paths.tfrecord.dataset_frame(dataset, drive_date, drive_number, frame)
  output_path.parent.mkdir(exist_ok=True, parents=True)  # ensure directory exists

  features = [
      ('data_rgb', data_rgb, 'float'),
      ('data_depth', data_depth, 'float'),
      ('data_decalib', data_decalib, 'float'),
  ]
  tfrecord.write_record(output_path=str(output_path), features=features)

  if not keep:
    if path_rgb.exists():
      path_rgb.unlink()
    if path_depth.exists():
      path_depth.unlink()
    if path_decalib.exists():
      path_decalib.unlink()


def generate_frames(frames, dataset, verbose=Verbose.NORMAL, keep=False):
  if verbose > Verbose.SILENT:
    info = '# generating tfrecords     '  # for logging purposes
    frames = tqdm(frames, ascii=True, desc=info)

  n_jobs = multiprocessing.cpu_count() // 2
  Parallel(n_jobs=n_jobs)(
      delayed(_generate_frame)(dataset, *frame_info, keep=keep)
      for frame_info in frames)
