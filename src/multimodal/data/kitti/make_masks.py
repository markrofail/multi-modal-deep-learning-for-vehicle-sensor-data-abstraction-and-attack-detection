import multiprocessing
import pickle

import cv2
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from src.helpers import paths
from src.helpers.flags import Verbose
from src.multimodal.helpers import mrcnn

###############################################################################
# DATA GENERATION PARAMETERS                                                      #
###############################################################################
config = paths.config.read(paths.config.multimodal())

# Padding parameters
KERNEL = config['DATASET']['MASK_PAD_KERNEL']
ITERATIONS = config['DATASET']['MASK_PAD_ITERATIONS']


def _pad_mask(image, kernel, iterations):
  kernel = np.ones((kernel, kernel), np.uint8)
  erosion = cv2.dilate(image, kernel, iterations=iterations)
  return erosion


def _generate_frame(drive_date, drive_number, frame, padding=True):
  image_path = paths.rgb.external_frame(drive_date, drive_number, frame)
  if not image_path.exists():
    return

  image = cv2.imread(str(image_path))

  input_path = paths.mask.mrcnn_pickle(drive_date, drive_number, frame)
  with open(input_path, 'rb') as handle:
    results = pickle.load(handle)

  mask = mrcnn.get_mask(image, results)

  # add padding to mask
  if padding:
    padding_config = (KERNEL, ITERATIONS)
    mask = _pad_mask(mask, *padding_config)

  output_path = paths.mask.raw_frame(drive_date, drive_number, frame)
  output_path.parent.mkdir(exist_ok=True, parents=True)  # ensure directory exists

  cv2.imwrite(img=mask, filename=str(output_path))


def generate_frames(frames, padding=True, verbose=Verbose.NORMAL):
  if verbose > Verbose.SILENT:
    info = '# generating object masks  '  # for logging purposes
    frames = tqdm(frames, ascii=True, desc=info)

  n_jobs = multiprocessing.cpu_count() // 2
  Parallel(n_jobs=n_jobs)(
      delayed(_generate_frame)(*frame_info, padding)
      for frame_info in frames)
