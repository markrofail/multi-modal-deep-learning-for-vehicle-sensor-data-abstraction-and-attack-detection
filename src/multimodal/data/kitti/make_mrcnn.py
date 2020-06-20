import pickle

import skimage.io
from tqdm import tqdm

from src.helpers import paths
from src.helpers.flags import Verbose
from src.multimodal.helpers import mrcnn


def _generate_frame(drive_date, drive_number, frame):
  """
  generates mrcnn feedforward result pickles image attack from a frame and
  saves it in the raw folder with the same structure as the original data,
  under mrcnn folder

  Usage::
      >>> generate_frame('2011_09_26',  1, 0)

  :param drive_date: drive date (ex. '2011_09_26')
  :param drive_number: drive number (ex. 1)
  :param frame: frame within drive (ex. 0)
  """
  input_path = paths.rgb.external_frame(drive_date, drive_number, frame)
  image = skimage.io.imread(input_path)

  results = mrcnn.get_results(image)

  output_path = paths.mask.mrcnn_pickle(drive_date, drive_number, frame)
  output_path.parent.mkdir(exist_ok=True, parents=True)  # ensure directory exists

  with open(output_path, 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def generate_frames(frames, verbose=Verbose.NORMAL):
  """
  generates mrcnn feedforward result pickles image attack from multiple
  frames and saves it in the raw folder with the same structure as the
  original data, under mrcnn folder

  Usage::
      >>> generate_frames([('2011_09_26', 1, 1), ...])

  :param frames: array of frame info tuples, i.e (drive_date, dive_number, frame)
  """

  if verbose > Verbose.SILENT:
    info = '# generating mcrnn pickles '  # for logging purposes
    frames = tqdm(frames, ascii=True, desc=info)

  for frame_info in frames:
    _generate_frame(*frame_info)


def delete_frames(frames, keep):
  if keep:
    return

  for frame_info in frames:
    mrcnn_file = paths.mask.mrcnn_pickle(*frame_info)
    if mrcnn_file.exists():
      mrcnn_file.unlink()
