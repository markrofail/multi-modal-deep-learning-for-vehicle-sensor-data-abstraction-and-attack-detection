import os

import cv2
import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras.preprocessing import image as keras_image

from src.helpers import display, paths
from src.helpers.flags import Verbose
from src.regnet import regnet
from src.regnet.data import make_dataset
from src.regnet.data.kitti.make_decalib import get_Hgt, save_as_txt
from src.regnet.helpers import kitti_foundation, quaternion
from src.regnet.models.predict_model import phi_decal
from src.regnet.visualization.visualize import euclidean_loss_np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(suppress=True, precision=4, sign=' ')

CUSTOM_Layers = {
    'euclidean_loss_layer': regnet.euclidean_loss_layer,
    'r2_score': regnet.r2_score,
    'y_true': regnet.y_true,
    'y_pred': regnet.y_pred
}


def add_image(img_arr, decal_params):
  img_path = path_projection(**decal_params, color='rgb')
  img = cv2.imread(filename=str(img_path))
  img_arr.append(img)


def run(drive_date, drive_number, frame):
  drive_info = {
      'drive_date': drive_date,
      'drive_number': drive_number,
      'frame': frame
  }

  # make_dataset.make_data([(drive_date, drive_number, frame)], 'train')
  make_dataset.make_data(
      [(drive_date, drive_number, frame)],
      'train',
      verbose=Verbose.SILENT,
      keep=True)

  imgs = list()
  print('\n###############ITERATIVE REFINEMENT###############')

  decalibration_original = {'rot': None, 'disp': None}
  inital_run(drive_info, decalibration_original)
  add_image(imgs, decalibration_original)

  decalibration_20_15 = {'rot': 20, 'disp': 1.5}
  feed_forward(drive_info, decalibration_20_15)
  add_image(imgs, decalibration_20_15)

  decalibration_10_10 = {'rot': 10, 'disp': 1.0}
  feed_forward(drive_info, decalibration_10_10, decalibration_20_15)
  add_image(imgs, decalibration_10_10)

  decalibration_05_05 = {'rot': 5, 'disp': 0.5}
  feed_forward(drive_info, decalibration_05_05, decalibration_10_10)
  add_image(imgs, decalibration_05_05)

  decalibration_02_02 = {'rot': 2, 'disp': 0.2}
  feed_forward(drive_info, decalibration_02_02, decalibration_05_05)
  add_image(imgs, decalibration_02_02)

  decalibration_01_01 = {'rot': 1, 'disp': 0.1}
  feed_forward(drive_info, decalibration_01_01, decalibration_02_02)
  add_image(imgs, decalibration_01_01)

  display.preview(*imgs, ncols=1)


def feed_forward(drive_info, decal_params, past_decal_params=None, debug=True):
  print('feeding the {rot}° {disp}m network'.format(**decal_params))

  assert isinstance(drive_info, dict), 'drive_info should be a dict'
  assert isinstance(decal_params, dict), 'decal_params should be a dict'
  if past_decal_params is not None:
    assert isinstance(past_decal_params,
                      dict), 'past_decal_params should be a dict'

  # load inputs
  # # load rgb input
  rgb_path = paths.rgb.processed_tensor(**drive_info)
  assert rgb_path.exists(), 'rgb image can not be found'
  rgb_tensor = np.load(rgb_path)

  # # load depth input
  if past_decal_params is not None:
    depth_path = path_projection(**past_decal_params)
    assert depth_path.exists(), 'depth image can not be found'
    depth_image = keras_image.load_img(depth_path, grayscale=True)
    depth_tensor = keras_image.img_to_array(depth_image)
    depth_tensor = depth_tensor.astype(np.float64) / 255
  else:
    depth_path = paths.depth.processed_tensor(**drive_info)
    assert depth_path.exists(), 'depth image can not be found'
    depth_tensor = np.load(depth_path)

  # # load phi decal
  if past_decal_params is not None:
    decal_path = path_calibration_raw(**past_decal_params)
  else:
    decal_path = paths.calibration.decalibration_matrix(**drive_info)
  assert decal_path.exists(), 'calbiration file not found'
  decal_dq = np.load(file=decal_path)  # random decalibration
  decal_mat = quaternion.dualqt_op.mat4_dualqt(decal_dq)

  # # load label
  label_path = paths.calibration.decalibration_matrix(**drive_info)
  assert label_path.exists(), 'calbiration file not found'
  label_dq = np.load(file=label_path)  # random decalibration
  label_mat = quaternion.dualqt_op.mat4_dualqt(label_dq)

  # load model
  model_path = paths.checkpoints.regnet(**decal_params)
  assert model_path.exists(), 'model hdf5 file can not be found'
  model = keras.models.load_model(
      str(model_path), custom_objects=CUSTOM_Layers)

  # predict
  pred_dq = model.predict([[rgb_tensor], [depth_tensor]])
  pred_mat = quaternion.dualqt_op.mat4_dualqt(pred_dq)

  euc_rot = euclidean_loss_np(
      logits=quaternion.helpers.mat3_mat4(pred_mat),
      labels=quaternion.helpers.mat3_mat4(label_mat))  # noqa
  euc_disp = euclidean_loss_np(
      logits=quaternion.helpers.vec_mat4(pred_mat),
      labels=quaternion.helpers.vec_mat4(label_mat))

  decal_correct = phi_decal(decal_mat, pred_mat)
  decal_dq = quaternion.mat4_op.dualqt_mat4(decal_correct)

  pred_path = path_calibration_raw(**decal_params)
  np.save(file=pred_path, arr=decal_dq)

  # Make H^ for projection
  h_gt = get_Hgt(drive_info['drive_date'])
  h_init = np.dot(decal_mat, h_gt)
  h_predict = phi_decal(h_init, pred_mat)

  pred_path = path_calibration(**decal_params)
  save_as_txt(h_predict, output_path=pred_path)

  # project new depthmap
  project(drive_info, decal_params)

  diff = None
  if debug:
    diff = np.subtract(h_predict, h_gt)
    diff = np.absolute(diff)

  print_results({'loss_rot': euc_rot, 'loss_disp': euc_disp, 'diff': diff})


def inital_run(drive_info, decal_params, debug=True):
  print('before refinement')
  project(drive_info, decal_params)

  label_path = paths.calibration.decalibration_matrix(**drive_info)
  assert label_path.exists(), 'calbiration file not found'
  label_dq = np.load(file=label_path)  # random decalibration
  label_mat = quaternion.dualqt_op.mat4_dualqt(label_dq)

  logits = np.zeros((4, 4))
  euc_rot = euclidean_loss_np(
      logits=quaternion.helpers.mat3_mat4(logits),
      labels=quaternion.helpers.mat3_mat4(label_mat))
  euc_disp = euclidean_loss_np(
      logits=quaternion.helpers.vec_mat4(logits),
      labels=quaternion.helpers.vec_mat4(label_mat))

  h_gt = get_Hgt(drive_info['drive_date'])

  h_init = label_path.with_suffix('.hinit.txt')
  h_init = get_Hgt(path=h_init)

  diff = None
  if debug:
    diff = np.subtract(h_init, h_gt)
    diff = np.absolute(diff)

  print_results({'loss_rot': euc_rot, 'loss_disp': euc_disp, 'diff': diff})


def print_results(kargs):
  for k, v in kargs.items():
    print(' - {k} = {v}'.format(k=k, v=v))
  print()


def project(drive_info, decal_params):
  image_path = str(paths.rgb.interim_frame(**drive_info))
  image = cv2.imread(image_path)

  velo_path = str(paths.depth.external_frame(**drive_info))
  velo_points = kitti_foundation.load_from_bin(velo_path)

  v2c_filepath = str(path_calibration(**decal_params, drive_info=drive_info))
  c2c_filepath = str(paths.calibration.cam_to_cam(drive_info['drive_date']))

  ans, c_ = kitti_foundation.velo3d_2_camera2d_points(
      velo_points,
      v_fov=(-24.9, 2.0),
      h_fov=(-45, 45),
      vc_path=v2c_filepath,
      cc_path=c2c_filepath,
      mode='02')

  result = kitti_foundation.print_projection_cv2(
      points=ans, color=c_, image=image, black_background=True)
  result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

  output_path = str(path_projection(**decal_params, color='grey'))
  cv2.imwrite(img=result, filename=output_path)

  result_c = kitti_foundation.print_projection_cv2(
      points=ans, color=c_, image=image, black_background=False)

  output_path = str(path_projection(**decal_params, color='rgb'))
  cv2.imwrite(img=result_c, filename=output_path)


ITERATIVE_REFINEMENT_PATH = paths.ROOT_PATH.joinpath('iterative_refinement')
ITERATIVE_REFINEMENT_PATH.mkdir(exist_ok=True, parents=True)


def path_calibration_raw(rot, disp):
  file_str = 'pred_{:02d}_{:02d}.npy'.format(rot, int(disp * 10))
  return ITERATIVE_REFINEMENT_PATH.joinpath(file_str)


def path_calibration(rot, disp, drive_info=None):
  if rot is None and disp is None:
    path = paths.calibration.decalibration_matrix(**drive_info)
    return path.with_suffix('.hinit.txt')
  file_str = 'pred_{:02d}_{:02d}.txt'.format(rot, int(disp * 10))
  return ITERATIVE_REFINEMENT_PATH.joinpath(file_str)


def path_projection(rot, disp, color='grey'):
  if rot is None and disp is None:
    file_str = 'initial.png'
  else:
    file_str = 'pred_{}_{:02d}_{:02d}.png'.format(color, rot,
                                                  int(disp * 10))
  return ITERATIVE_REFINEMENT_PATH.joinpath(file_str)


###############################################################################
# TRAIN DATA                                                                  #
###############################################################################
TRAIN_DRIVE_DATE = '2011_09_26'
TRAIN_DRIVE_NUMBER = 1
TRAIN_DRIVE_FRAME = 2
TRAIN_DATA = TRAIN_DRIVE_DATE, TRAIN_DRIVE_NUMBER, TRAIN_DRIVE_FRAME

if __name__ == '__main__':
  run(TRAIN_DRIVE_DATE, TRAIN_DRIVE_NUMBER, TRAIN_DRIVE_FRAME)
