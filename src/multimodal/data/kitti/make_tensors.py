import multiprocessing

import numpy as np
from joblib import Parallel, delayed
from keras.preprocessing import image
from tqdm import tqdm

from src.helpers import paths
from src.helpers.flags import Verbose


def _generate_frame(drive_date, drive_number, frame, verbose=Verbose.NORMAL, keep=False):
  input_path = paths.attack.interim_frame(drive_date, drive_number, frame)
  if not input_path.exists():
    return

  # loads image as PIL.Image.Image type
  img = image.load_img(input_path)

  # convert PIL.Image.Image type to tensor
  tensor = image.img_to_array(img)
  tensor = tensor.astype(np.float64) / 255

  output_path = paths.attack.processed_tensor(drive_date, drive_number, frame)
  output_path.parent.mkdir(exist_ok=True, parents=True)  # ensure directory exists

  np.save(output_path, tensor)

  if input_path.exists() and not keep:
    input_path.unlink()


def generate_frames(frames, verbose=Verbose.NORMAL, keep=False):
  if verbose > Verbose.SILENT:
    info = '# converting attacks to tensors'  # for logging purposes
    frames = tqdm(frames, ascii=True, desc=info)

  n_jobs = multiprocessing.cpu_count() // 2
  Parallel(n_jobs=n_jobs)(
      delayed(_generate_frame)(*frame_info, verbose, keep=keep)
      for frame_info in frames)
