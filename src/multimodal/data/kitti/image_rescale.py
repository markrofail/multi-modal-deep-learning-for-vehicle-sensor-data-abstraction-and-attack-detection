import multiprocessing

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from src.helpers import paths
from src.helpers.flags import Verbose
from src.regnet.data.kitti import image_rescale


def _get_all_attack_frames(frames):
  return [str(paths.attack.raw_frame(*frame_info)) for frame_info in frames]


def _generate_output_paths(image_paths):
  replace_dir = np.vectorize(lambda x: x.replace('raw', 'interim'))
  output_paths = replace_dir(image_paths)

  return np.column_stack((image_paths, output_paths))


def preproccess_frames(frames, verbose=Verbose.NORMAL, keep=False):
  # grab all attackframe paths
  attack_frames = _get_all_attack_frames(frames)

  # gnerate ouputpath corresponding to each inputpath
  image_paths = _generate_output_paths(attack_frames)

  if verbose > Verbose.SILENT:
    info = '# processing attack images '  # for logging purposes
    image_paths = tqdm(image_paths, ascii=True, desc=info)

  n_jobs = multiprocessing.cpu_count() // 2
  Parallel(n_jobs=n_jobs)(
      delayed(image_rescale.preproccess_image)(input_path, output_path,
                                               verbose=verbose, keep=keep)
      for input_path, output_path in image_paths)
