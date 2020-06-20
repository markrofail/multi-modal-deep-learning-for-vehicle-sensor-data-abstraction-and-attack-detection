import multiprocessing

import numpy as np
from joblib import Parallel, delayed
from keras.preprocessing import image
from tqdm import tqdm

from src.helpers import paths
from src.helpers.flags import Verbose


def _get_input_path(drive_date, drive_number, frame, data_type):
  path = ''
  if data_type == 'rgb':
    path = paths.rgb.interim_frame(drive_date, drive_number, frame)
  elif data_type == 'depth':
    path = paths.depth.interim_frame(drive_date, drive_number, frame)
  return path


def _get_output_path(drive_date, drive_number, frame, data_type):
  path = ''
  if data_type == 'rgb':
    path = paths.rgb.processed_tensor(drive_date, drive_number, frame)
  elif data_type == 'depth':
    path = paths.depth.processed_tensor(drive_date, drive_number, frame)
  return path


def _generate_frame(drive_date, drive_number, frame, data_type, verbose=Verbose.NORMAL, keep=False):
  input_path = _get_input_path(drive_date, drive_number, frame, data_type)
  if not input_path.exists():
    return

  # loads image as PIL.Image.Image type
  if data_type == 'depth':
    img = image.load_img(input_path, grayscale=True)
  else:
    img = image.load_img(input_path)

  # convert PIL.Image.Image type to tensor
  tensor = image.img_to_array(img)
  tensor = tensor.astype(np.float64) / 255

  output_path = _get_output_path(drive_date, drive_number, frame, data_type)
  output_path.parent.mkdir(exist_ok=True, parents=True)  # ensure directory exists

  np.save(output_path, tensor)

  if input_path.exists() and not keep:
    input_path.unlink()


def generate_frames(frames, data_type, verbose=Verbose.NORMAL, keep=False):
  assert data_type in ['rgb', 'depth'], 'dataset has to be one of \'rgb\' or \'depth\''

  if verbose > Verbose.SILENT:
    info = '# converting {} to tensors'.format(data_type)  # for logging purposes
    frames = tqdm(frames, ascii=True, desc=info)

  n_jobs = multiprocessing.cpu_count() // 2
  Parallel(n_jobs=n_jobs)(
      delayed(_generate_frame)(*frame_info, data_type, verbose, keep=keep)
      for frame_info in frames)
