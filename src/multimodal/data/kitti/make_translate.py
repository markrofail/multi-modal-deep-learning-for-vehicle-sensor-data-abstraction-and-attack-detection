# pylint: disable=no-member
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
# DATA GENERATION PARAMETERS                                                  #
###############################################################################
config = paths.config.read(paths.config.multimodal())
# Translation parameters
TRANSLATION_CONST = config['DATASET']['TRANSLATION_CONST']

# Padding parameters
KERNEL = config['DATASET']['MASK_PAD_KERNEL']
ITERATIONS = config['DATASET']['MASK_PAD_ITERATIONS']


def _pad_mask(image, kernel, iterations):
  kernel = np.ones((kernel, kernel), np.uint8)
  erosion = cv2.dilate(image, kernel, iterations=iterations)
  return erosion


def _split_masks(masks, padding=True):
  """splits masks to individual images"""

  masks = visualize.return_masks(masks)

  # add padding to mask
  if padding:
    for mask in masks:
      padding_config = (KERNEL, ITERATIONS)
      mask = _pad_mask(mask, *padding_config)
  return masks


def _extract_objects(image, masks):
  """extracts the objects from the image using the masks"""

  objects = []
  for mask in masks:
    mask[:, :] = np.where(mask == 255, image[:, :], mask[:, :])
    objects.append(mask)
  return objects


def _translate_image(objects):
  """translate objects independently"""

  translated_objects = list()
  for obj in objects:
    # for each mask calculate the bounding box
    cords = np.where(obj > 0)
    y_min, y_max = np.amin(cords[0]), np.amax(cords[0])
    x_min, x_max = np.amin(cords[1]), np.amax(cords[1])

    # calculate center of object using x = (x_min + x_max) / 2
    obj_x, obj_y = np.mean([x_min, x_max]), np.mean([y_min, y_max])

    # tranlate on x
    center_x = obj.shape[1] // 2
    distance_x = np.absolute(obj_x - center_x)

    shift_x = int(distance_x / center_x * TRANSLATION_CONST) + 1
    if obj_x > center_x:
      obj = np.pad(obj, [(0, 0), (shift_x, 0), (0, 0)], mode='constant')[:, :-shift_x]
    else:
      obj = np.pad(obj, [(0, 0), (0, shift_x), (0, 0)], mode='constant')[:, shift_x:]

    # tranlate on y
    shift_y = int(obj_y / obj.shape[0] * TRANSLATION_CONST)
    obj = np.pad(obj, [(0, shift_y), (0, 0), (0, 0)], mode='constant')[shift_y:, :]

    translated_objects.append(obj)
  return translated_objects


def _apply_back(image, objects):
  """draw objects to image"""

  for obj in objects:
    image[:, :] = np.where(obj > 0, obj[:, :], image[:, :])
  return image


def _patch_remains(image, masks):
  """inpaint remaining parts of the image"""

  for mask in masks:
    mask = cv2.split(mask)[0]
    image = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)
  return image


def _generate_frame(drive_date, drive_number, frame, padding=True):
  """
  generates translated image attack from a frame and saves it in the raw
  folder with the same structure as the original data, under translate folder

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
  masks = _split_masks(result['masks'], padding)

  # step 3: exctract objects from original image
  objects = _extract_objects(image, masks)

  # step 4: translate every object independently
  objects = _translate_image(objects)

  # step 5: inpaint old objects places
  image = _patch_remains(image, masks)

  # step 5: draw objects on image
  image = _apply_back(image, objects)

  output_path = paths.attack.raw_frame(drive_date, drive_number, frame)
  output_path.parent.mkdir(exist_ok=True, parents=True)  # ensure directory exists

  cv2.imwrite(img=image, filename=str(output_path))


def generate_frames(frames, padding=True, verbose=Verbose.NORMAL):
  """
  generates translated image attack from multiple frames and saves it in the
  raw folder with the same structure as the original data, under translate
  folder

  Usage::
      >>> generate_frames([('2011_09_26', 1, 1), ...])

  :param frames: array of frame info tuples, i.e (drive_date, dive_number, frame)
  """
  if verbose > Verbose.SILENT:
    info = '# generating translation attacks'  # for logging purposes
    frames = tqdm(frames, ascii=True, desc=info)

  n_jobs = multiprocessing.cpu_count() // 2
  Parallel(n_jobs=n_jobs)(
      delayed(_generate_frame)(*frame_info, padding)
      for frame_info in frames)
