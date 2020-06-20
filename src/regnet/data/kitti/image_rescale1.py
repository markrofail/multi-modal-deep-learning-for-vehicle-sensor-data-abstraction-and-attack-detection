import configparser
import os

import numpy as np
from PIL import Image as im
from scipy.stats import entropy as entropy_helper
from tqdm import tqdm

from src.helpers import paths

config = paths.config.read(paths.config.regnet())
IMAGE_SIZE = tuple(config['DATA_INFORMATION']['IMAGE_SIZE'])


def entropy(image):
  rgbHistogram = image.histogram()
  rgbHistogram = np.reshape(rgbHistogram, (3, 256)).astype(float)

  ent = np.zeros((3, 1), dtype=float)
  for i in range(3):
    total = np.sum(rgbHistogram[i])
    rgbHistogram[i] = np.divide(rgbHistogram[i], total)
    ent[i] = entropy_helper(rgbHistogram[i], base=2)

  ent = np.average(ent)
  return ent


def demostrate_state():
  old_size = 1242, 375
  # new_size = 1392, 512
  new_size = IMAGE_SIZE
  print(IMAGE_SIZE)
  path = ('/home/mark/Dev/BachelorThesis/repo/data/external/KITTI/'
          '2011_09_26/2011_09_26_drive_0001_sync/image_02/data/'
          '0000000000.png')

  image = im.open(path)
  ent = entropy(image)
  print("original entropy: \t\t\t{:.4f}".format(ent))
  print()

  # h_scale = 1392, 574
  h_scale = 1216, 367
  v_scale = 1696, 512

  h_rescale = image.resize(h_scale, resample=im.BILINEAR)
  v_rescale = image.resize(v_scale, resample=im.BILINEAR)

  h_rescale_ent = entropy(h_rescale)
  v_rescale_ent = entropy(v_rescale)

  print("entropy after scaling horizontally: \t{:.4f}".format(h_rescale_ent))
  print("entropy after scaling vertically: \t{:.4f}".format(v_rescale_ent))

  h_width, h_height = h_scale
  v_width, v_height = v_scale

  width, height = old_size
  new_width, new_height = new_size

  h_bounds = ((h_width - new_width) // 2, (h_height - new_height) // 2,
              (h_width + new_width) // 2, (h_height + new_height) // 2)
  v_bounds = ((v_width - new_width) // 2, (v_height - new_height) // 2,
              (v_width + new_width) // 2, (v_height + new_height) // 2)

  h_crop = h_rescale.crop(h_bounds)
  v_crop = v_rescale.crop(v_bounds)
  print()
  h_crop_ent = entropy(h_crop)
  v_crop_ent = entropy(v_crop)

  print("entropy after crop horizontally: \t{:.4f}".format(h_crop_ent))
  print("entropy after crop vertically: \t\t{:.4f}".format(v_crop_ent))
  print()

  diff_h = ent - h_crop_ent
  diff_v = ent - v_crop_ent

  print('Information gain = before - after')
  print("horizontal gain  = {:.4f} - {:.4f} = \t{:.4f}".format(ent,
                                                               h_crop_ent,
                                                               diff_h))
  print("vertical   gain  = {:.4f} - {:.4f} = \t{:.4f}".format(ent,
                                                               v_crop_ent,
                                                               diff_v))
  print()
  print("HORIZONTAL scaling has least information loss")


def demostrate_modes():
  path = paths.get_KITTI_camera_frame(drive_date='2011_09_26',
                                      drive_number=1,
                                      frame=0)

  image = im.open(path)
  ent = entropy(image)
  print("original entropy: \t\t\t{:.4f}".format(ent))
  print()

  h_scale = 1696, 512

  scale_NEAREST = image.resize(h_scale, resample=im.NEAREST)
  scale_BILINEAR = image.resize(h_scale, resample=im.BILINEAR)
  scale_BICUBIC = image.resize(h_scale, resample=im.BICUBIC)
  scale_LANCZOS = image.resize(h_scale, resample=im.LANCZOS)

  ent_NEAREST = entropy(scale_NEAREST)
  ent_BILINEAR = entropy(scale_BILINEAR)
  ent_BICUBIC = entropy(scale_BICUBIC)
  ent_LANCZOS = entropy(scale_LANCZOS)

  diff_NEAREST = ent - ent_NEAREST
  diff_BILINEAR = ent - ent_BILINEAR
  diff_BICUBIC = ent - ent_BICUBIC
  diff_LANCZOS = ent - ent_LANCZOS

  print('Information gain = before - after')
  print("NEAREST gain  =\t {:.4f} - {:.4f} = \t{:.4f}".format(ent,
                                                              ent_NEAREST,
                                                              diff_NEAREST
                                                              ))
  print("BILINEAR gain =\t {:.4f} - {:.4f} = \t{:.4f}".format(ent,
                                                              ent_BILINEAR,
                                                              diff_BILINEAR
                                                              ))
  print("BICUBIC gain  =\t {:.4f} - {:.4f} = \t{:.4f}".format(ent,
                                                              ent_BICUBIC,
                                                              diff_BICUBIC
                                                              ))
  print("LANCZOS gain  =\t {:.4f} - {:.4f} = \t{:.4f}".format(ent,
                                                              ent_LANCZOS,
                                                              diff_LANCZOS
                                                              ))

  print()
  print("NEAREST resampling filter has least information loss")


def get_all_rgb_frames(drive_number, drive_date):
  base_dir = paths.get_KITTI_camera_frame(drive_date=drive_date,
                                          drive_number=drive_number,
                                          frame=0)

  base_dir = os.path.dirname(base_dir)
  onlyfiles = [f[:-4] for f in os.listdir(base_dir)
               if os.path.isfile(os.path.join(base_dir, f))]
  onlyfiles = np.array(onlyfiles).astype(int)
  onlyfiles = np.sort(onlyfiles)
  return onlyfiles


def get_all_depth_frames(drive_number, drive_date):
  base_dir = local_paths.get_KITTI_depth_interim_frame(drive_date=drive_date,
                                                       drive_number=drive_number,
                                                       frame=0)

  base_dir = os.path.dirname(base_dir)
  onlyfiles = [f[:-4] for f in os.listdir(base_dir)
               if os.path.isfile(os.path.join(base_dir, f))]
  onlyfiles = np.array(onlyfiles).astype(int)
  onlyfiles = np.sort(onlyfiles)

  return onlyfiles


def preproccess_image(input_path, output_path, debug=False):
  if not os.path.exists(input_path):
    return

  new_dim = IMAGE_SIZE

  img = im.open(input_path)

  scale = 1696, 512
  scaled_img = img.resize(scale, resample=im.NEAREST)

  new_width, new_height = new_dim
  old_width, old_height = scale
  bounds = ((old_width - new_width) // 2, (old_height - new_height) // 2,
            (old_width + new_width) // 2, (old_height + new_height) // 2)

  croped_img = scaled_img.crop(bounds)

  output_dir = os.path.dirname(output_path)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  croped_img.save(output_path, format='PNG')

  if debug:
    ent1 = entropy(img)
    ent2 = entropy(croped_img)
    diff = 100 - int(ent1 / ent2 * 100)
    print('{}% information lost'.format(diff))


def preproccess_drive_depth(drive_number, drive_date):
  print('processing {}_drive_{:04d} camera images....'.format(drive_date, drive_number))
  velo_frames = get_all_depth_frames(drive_number, drive_date)

  for frame in tqdm(velo_frames):
    # print('processing:  '+drive_date + '_drive_%04d_sync' % drive_number +
    #       ':' + '%010d' % frame)

    input_path = local_paths.get_KITTI_depth_interim_frame(drive_date=drive_date,
                                                           drive_number=drive_number,
                                                           frame=frame)

    output_path = local_paths.get_KITTI_depth_frame(drive_date=drive_date,
                                                    drive_number=drive_number,
                                                    frame=frame)

    preproccess_image(input_path=input_path, output_path=output_path)


def preproccess_drive_rgb(drive_number, drive_date):
  print('processing {}_drive_{:04d} depth images....'.format(drive_date, drive_number))
  rgb_frames = get_all_rgb_frames(drive_number, drive_date)

  for frame in tqdm(rgb_frames):
    # print('processing:  '+drive_date + '_drive_%04d_sync' % drive_number +
    #       ':' + '%010d' % frame)

    input_path = paths.get_KITTI_camera_frame(drive_date=drive_date,
                                              drive_number=drive_number,
                                              frame=frame)

    output_path = local_paths.get_KITTI_rgb_frame(drive_date=drive_date,
                                                  drive_number=drive_number,
                                                  frame=frame)

    preproccess_image(input_path=input_path, output_path=output_path)


def preproccess_drive(drive_number, drive_date):
  path = paths.get_KITTI_camera_frame(drive_date, drive_number, 0)
  if not os.path.exists(path):
    return

  preproccess_drive_rgb(drive_number, drive_date)
  preproccess_drive_depth(drive_number, drive_date)


def preproccess_drives(drives, drive_date):
  for drive in drives:
    preproccess_drive(drive, drive_date)


if __name__ == '__main__':
  demostrate_state()
  # demostrate_modes()
