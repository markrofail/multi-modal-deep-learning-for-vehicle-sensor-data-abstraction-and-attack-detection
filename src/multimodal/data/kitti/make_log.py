import csv
import multiprocessing
import pickle

import cv2
import numpy as np
from joblib import Parallel, delayed
from mrcnn import visualize
from tqdm import tqdm

from src.helpers import paths
from src.helpers.flags import Verbose

###############################################################################
# DATASET PARAMETERS                                                          #
###############################################################################
config = paths.config.read(paths.config.multimodal())
ATTACK_TYPE = config['DATASET']['ATTACK_TYPE']


def _split_masks(masks):
  """splits masks to individual images"""
  masks = visualize.return_masks(masks)
  return masks


def _calculate_difficulty(objects, csv_data, attack_type):
  index = 0
  for obj in objects:
    # area of irregular object is number of pixels
    size = np.count_nonzero(obj)

    cords = np.where(obj > 0)

    # for each obj calculate the bounding box
    y_min, y_max = np.amin(cords[0]), np.amax(cords[0])
    x_min, x_max = np.amin(cords[1]), np.amax(cords[1])

    # calculate center of object using x = (x_min + x_max) / 2
    obj_x, obj_y = np.mean([x_min, x_max]), np.mean([y_min, y_max])

    center_x = obj.shape[1] // 2
    distance_x = np.absolute(obj_x - center_x)
    distance = np.sqrt(distance_x**2 + obj_y**2)

    csv_data.append([index, size, round(distance, 2), attack_type])

    index += 1


def _generate_frame(drive_date, drive_number, frame, attack_type, verbose=Verbose.NORMAL):
  """
  generates a log file corresponding to the attack frame saves it in the raw
  folder with the same structure as the original data, under attack/log folder

  Usage::
      >>> generate_frame('2011_09_26',  1, 0)

  :param drive_date: drive date (ex. '2011_09_26')
  :param drive_number: drive number (ex. 1)
  :param frame: frame within drive (ex. 0)
  """
  image = paths.rgb.external_frame(drive_date, drive_number, frame)
  image = cv2.imread(str(image))

  # load stored mrcnn result pickle file
  pickle_path = paths.mask.mrcnn_pickle(drive_date, drive_number, frame)
  with open(pickle_path, 'rb') as handle:
    result = pickle.load(handle)

  # step 2: split masks to individual images
  masks = _split_masks(result['masks'])

  # step 3: log difficulty
  csv_data = [['index', 'size', 'distance', 'attack_type']]
  _calculate_difficulty(masks, csv_data, attack_type)

  output_path = paths.attack.log_file(drive_date, drive_number, frame, attack_type)
  output_path.parent.mkdir(exist_ok=True, parents=True)  # ensure directory exists

  with open(output_path, 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(csv_data)


def generate_frames(frames, attack_type, verbose=Verbose.NORMAL):
  """
  generates a log file corresponding to the attack frame from multiple saves
  it in the raw folder with the same structure as the original data, under
  attack/log folder

  Usage::
      >>> generate_frames([('2011_09_26', 1, 1), ...])

  :param frames: array of frame info tuples, i.e (drive_date, dive_number, frame)
  """
  if verbose > Verbose.SILENT:
    info = '# generating difficulty log'  # for logging purposes
    frames = tqdm(frames, ascii=True, desc=info)

  n_jobs = multiprocessing.cpu_count() // 2
  Parallel(n_jobs=n_jobs)(
      delayed(_generate_frame)(*frame_info, attack_type, verbose)
      for frame_info in frames)
