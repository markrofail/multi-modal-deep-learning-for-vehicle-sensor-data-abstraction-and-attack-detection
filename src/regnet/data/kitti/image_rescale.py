import multiprocessing
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from PIL import Image as im
from scipy.stats import entropy as entropy_helper
from tqdm import tqdm

from src.helpers import paths
from src.helpers.flags import Verbose

config = paths.config.read(paths.config.regnet())
IMAGE_SIZE = config['DATA_INFORMATION']['IMAGE_SIZE']

FAKE_KITTI_SIZE = (1392, 512)
REAL_KITTI_SIZE = (1242, 375)
OLD_SCALE_DIM = (1696, 512)
CALC_SCALE_DIM = (1216, 367)

SCALE_DIM = CALC_SCALE_DIM
# SCALE_DIM = OLD_SCALE_DIM


def entropy(image):
  rgbHistogram = np.array(image.histogram())
  print(np.prod(rgbHistogram.shape))
  if np.prod(rgbHistogram.shape) <= 256:
    rgbHistogram = np.tile(rgbHistogram, 3)
  rgbHistogram = np.reshape(rgbHistogram, (3, 256)).astype(float)

  ent = np.zeros((3, 1), dtype=float)
  for i in range(3):
    total = np.sum(rgbHistogram[i])
    rgbHistogram[i] = np.divide(rgbHistogram[i], total)
    ent[i] = entropy_helper(rgbHistogram[i], base=2)

  ent = np.average(ent)
  return ent


def _get_all_depth_frames(frames):
  return [str(paths.depth.raw_frame(*frame_info)) for frame_info in frames]


def _get_all_camera_frames(frames):
  return [str(paths.rgb.external_frame(*frame_info)) for frame_info in frames]


def _generate_output_paths(image_paths):
  image_paths = np.array(image_paths)

  replace_dir = np.vectorize(lambda x: x.replace('external', 'interim'))
  output_paths = replace_dir(image_paths)

  replace_dir = np.vectorize(lambda x: x.replace('raw', 'interim'))
  output_paths = replace_dir(output_paths)

  replace_dir = np.vectorize(lambda x: x.replace('image_02/data', 'rgb'))
  output_paths = replace_dir(output_paths)

  replace_dir = np.vectorize(lambda x: x.replace('depth_maps', 'depth'))
  output_paths = replace_dir(output_paths)

  return np.column_stack((image_paths, output_paths))


def preproccess_image(input_path, output_path, scale_dim=SCALE_DIM, final_dim=IMAGE_SIZE,
                      verbose=Verbose.NORMAL, keep=False):
  input_path, output_path = Path(input_path), Path(output_path)

  if not input_path.exists():
    return
  img = im.open(input_path)

  # scaling the image
  scaled_img = img.resize(scale_dim, resample=im.NEAREST)

  # cropping the image
  old_width, old_height = scale_dim
  new_width, new_height = final_dim

  bounds = ((old_width - new_width) // 2, (old_height - new_height) // 2,
            (old_width + new_width) // 2, (old_height + new_height) // 2)
  croped_img = scaled_img.crop(bounds)

  output_path.parent.mkdir(exist_ok=True, parents=True)  # ensure directory exists
  croped_img.save(str(output_path), format='PNG')

  if verbose == Verbose.DEBUG:
    ent1 = entropy(img)
    ent2 = entropy(croped_img)
    diff = 100 - int(ent1 / ent2 * 100)
    print('{}% information lost'.format(diff))

  if not keep:
    external_path = paths.DATA_EXTERNAL_PATH
    if input_path.exists() and external_path not in input_path.parents:
      input_path.unlink()


def preproccess_frames(frames, verbose=Verbose.NORMAL, keep=False):
  input_paths = list()

  # grab all depthmap paths
  depth_frames = _get_all_depth_frames(frames)
  input_paths.extend(depth_frames)

  # grab all rgb images paths
  camera_frames = _get_all_camera_frames(frames)
  input_paths.extend(camera_frames)

  # gnerate ouputpath corresponding to each inputpath
  image_paths = _generate_output_paths(input_paths)

  if verbose > Verbose.SILENT:
    info = '# processing images        '  # for logging purposes
    image_paths = tqdm(image_paths, ascii=True, desc=info)

  n_jobs = multiprocessing.cpu_count() // 2
  Parallel(n_jobs=n_jobs)(
      delayed(preproccess_image)(input_path, output_path, verbose=verbose, keep=keep)
      for input_path, output_path in image_paths)
