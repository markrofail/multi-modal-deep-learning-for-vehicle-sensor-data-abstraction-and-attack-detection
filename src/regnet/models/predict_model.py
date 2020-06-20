import os

import cv2
import numpy as np

from src.helpers import display, paths
from src.helpers.flags import Verbose
from src.regnet import regnet
from src.regnet.data import make_dataset
from src.regnet.data.kitti import make_decalib
from src.regnet.helpers import kitti_foundation, quaternion
from src.regnet.visualization import visualize

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(suppress=True, precision=4, sign=' ')

###############################################################################
# DATA PARAMETERS
###############################################################################
config = paths.config.read(paths.config.regnet())
ITERATIVE_REFINEMENT = config['DECALIBRATIONS']['ITERATIVE_REFINEMENT']
ROT = config['DECALIBRATIONS']['RANGE_ROTATION'] if ITERATIVE_REFINEMENT else None
DISP = config['DECALIBRATIONS']['RANGE_DISPLACEMENT'] if ITERATIVE_REFINEMENT else None
DUAL_QUATERNIONS = int(config['DATA_INFORMATION']['DUAL_QUATERNIONS'])
VERBOSE = config['ENVIROMENT_CONFIG']['VERBOSE']

DEFAULT_NUMBER = 1
DEFAULT_FRAME = 1
DEFAULT_DRIVE = '2011_09_26'


def euclidean_loss_np(logits, labels):
  return np.mean(np.square(logits - labels)) / 2


def feed_forward(input_rgb, input_depth, h_init, drive_date, h_gt=None, verbose=VERBOSE):
  print_args = dict()

  # Create the network
  net = regnet.Regnet()

  # Load pretrained model
  model_path = str(paths.checkpoints.regnet(rot=ROT, disp=DISP))  # ./checkpoints/regnet/train
  net.model.load_weights(model_path)

  # Predict
  pred = net.model.predict([[input_rgb], [input_depth]])

  # Transform prediction
  if DUAL_QUATERNIONS:
    pred = quaternion.dualqt_op.mat4_dualqt(pred)
  else:
    pred = np.reshape(pred, (4, 4))

  # Log loss
  label = h_init
  if DUAL_QUATERNIONS:
    label = quaternion.dualqt_op.mat4_dualqt(label)

  total_loss = euclidean_loss_np(logits=pred, labels=label)
  total_loss = np.array([total_loss])
  print_args['loss'] = total_loss

  rot_loss = euclidean_loss_np(logits=quaternion.helpers.mat3_mat4(pred),
                                         labels=quaternion.helpers.mat3_mat4(label))
  rot_loss = np.array([rot_loss])
  print_args['rot_loss'] = rot_loss

  disp_loss = euclidean_loss_np(logits=quaternion.helpers.vec_mat4(pred),
                                          labels=quaternion.helpers.vec_mat4(label))
  disp_loss = np.array([disp_loss])
  print_args['disp_loss'] = disp_loss

  # Make H^ for projection
  h_gt = make_decalib.get_Hgt(drive_date)
  h_init = np.dot(label, h_gt)
  h_predict = phi_decal(h_init, pred)  # Hinit . H.gt⁻¹
  save_path = paths.ROOT_PATH.joinpath('pred.txt')
  make_decalib.save_as_txt(h_predict, save_path)

  if verbose == Verbose.DEBUG:
    diff = np.subtract(label, pred)
    diff = np.absolute(diff)

    print_args['prediction'] = pred
    print_args['actual'] = label
    print_args['difference'] = diff
  print_results(**print_args)


def print_results(**kargs):
  print('# results:')

  for k, v in kargs.items():
    print('## {k} = {v}'.format(k=k, v=v))
  print()


def phi_decal(H_init, H_gt, verbose=VERBOSE):
  H_gt_inv = np.linalg.inv(H_gt)

  if verbose == Verbose.DEBUG:
    print('H.gt⁻¹\n{}\n'.format(H_gt_inv))

  return np.dot(H_gt_inv, H_init)


def project_depth(drive_date, drive_number, frame, input_image=None, use_prediction=False):
  """ save one frame about projecting velodyne points into camera image """
  path = paths.depth.external_frame(drive_date, drive_number, frame)
  if not path.exists():
    return

  if input_image is not None:
    image_path = input_image
  else:
    image_path = paths.rgb.external_frame(drive_date, drive_number, frame)
  image = cv2.imread(str(image_path))

  velo_path = paths.depth.external_frame(drive_date, drive_number, frame)
  velo_points = kitti_foundation.load_from_bin(str(velo_path))

  c2c_filepath = paths.calibration.cam_to_cam(drive_date)

  v2c_filepath = paths.calibration.decalibration_matrix(drive_date, drive_number, frame)
  v2c_filepath = v2c_filepath.with_suffix('.hinit.txt')

  if use_prediction:
    v2c_filepath = paths.ROOT_PATH
    v2c_filepath = v2c_filepath.joinpath('pred.txt')

  ans, c_ = kitti_foundation.velo3d_2_camera2d_points(velo_points, v_fov=(-24.9, 2.0),
                                                      h_fov=(-45, 45), vc_path=str(v2c_filepath),
                                                      cc_path=str(c2c_filepath), mode='02')

  result = kitti_foundation.print_projection_cv2(points=ans, color=c_, image=image)

  if use_prediction:
    v2c_filepath.unlink()

  return result


def predict(drive_date, drive_number, frame):
  # create the frame data
  print('# generating data...')
  make_dataset.make_data([(drive_date, drive_number, frame)], verbose=Verbose.SILENT, keep=True)

  rgb_path = paths.rgb.processed_tensor(drive_date, drive_number, frame)
  rgb_data = np.load(rgb_path)

  depth_path = paths.depth.processed_tensor(drive_date, drive_number, frame)
  depth_data = np.load(depth_path)

  label_path = paths.calibration.decalibration_matrix(drive_date, drive_number, frame)
  label_data = np.load(label_path)

  print('# feedforward data...')
  feed_forward(input_rgb=rgb_data, input_depth=depth_data, h_init=label_data,
               drive_date=drive_date)


def get_arguments_interactively():
  print('\nEnter drive details:')
  args_dict = dict()

  default_drive = DEFAULT_DRIVE
  drive_date = input('# drive date (format: \'yyyy_mm_dd\') [\'{}\']:'.
                     format(default_drive))
  args_dict['drive_date'] = drive_date

  default_number = DEFAULT_NUMBER
  drive_number = input('# drive number (int) [{}]:'.format(default_number))
  args_dict['drive_number'] = drive_number

  default_frame = DEFAULT_FRAME
  frame = input('# drive frame (int) [{}]:'.format(default_frame))
  args_dict['frame'] = frame

  return args_dict


def get_arguments_argumentively(args):
  args_dict = dict()
  args_dict['drive_date'] = args.drive_date
  args_dict['drive_number'] = int(args.drive_number)
  args_dict['frame'] = int(args.drive_frame)
  return args_dict


def set_to_defaults(args_dict):
  # check if parameters are empty
  default_drive = DEFAULT_DRIVE
  if args_dict['drive_date'] == '':
    args_dict['drive_date'] = default_drive

  default_number = DEFAULT_NUMBER
  if args_dict['drive_number'] == '':
    args_dict['drive_number'] = default_number
  args_dict['drive_number'] = int(args_dict['drive_number'])

  default_frame = DEFAULT_FRAME
  if args_dict['frame'] == '':
    args_dict['frame'] = default_frame
  args_dict['frame'] = int(args_dict['frame'])

  # load set parameters
  drive_info = (args_dict['drive_date'],
                args_dict['drive_number'],
                args_dict['frame'])

  # load paths from parameters
  args_dict['rgb_path'] = paths.rgb.external_frame(*drive_info)
  args_dict['depth_path'] = paths.depth.external_frame(*drive_info)
  args_dict['label_path'] = paths.calibration.decalibration_matrix(*drive_info)

  return args_dict


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', help='interactively', action='store_true')
  parser.add_argument('-d', help='display results', action='store_true')

  parser.add_argument('--drive_date', help='date of the drive')
  parser.add_argument('--drive_number', help='drive number')
  parser.add_argument('--drive_frame', help='frame within the drive')

  args = parser.parse_args()

  if args.i:
    # take arguments interactively
    args_dict = get_arguments_interactively()
  else:
    # take arguments via command line
    args_dict = get_arguments_argumentively(args)

  # check if empty set to defaults
  args_dict = set_to_defaults(args_dict)

  predict_args = dict()
  predict_args['drive_date'] = args_dict['drive_date']
  predict_args['drive_number'] = args_dict['drive_number']
  predict_args['frame'] = args_dict['frame']
  print('\nPreciting results for date:{drive_date} drive:{drive_number} frame:{frame}'.
        format(**predict_args))
  predict(**predict_args)

  if args.d:
    project_args = predict_args
    project_args['input_image'] = paths.rgb.interim_frame(**project_args)

    img_before = project_depth(**project_args, use_prediction=False)
    img_after = project_depth(**project_args, use_prediction=True)

    cv2.imwrite(filename='img_before.png', img=img_before)
    cv2.imwrite(filename='img_after.png', img=img_after)

    display.preview(img_before, img_after)


'''
python -m src.regnet.models.predict_model -d \
--drive_date 2011_09_26 --drive_number 1 --drive_frame 0

## removed
--input_image /home/mark/Dev/BachelorThesis/repo/data/external/KITTI/2011_09_26/2011_09_26_drive_0001_sync/rgb/0000000000.png \
--input_rgb /home/mark/Dev/BachelorThesis/repo/data/processed/KITTI/2011_09_26/2011_09_26_drive_0001_sync/rgb/0000000000.npy \
--input_depth /home/mark/Dev/BachelorThesis/repo/data/processed/KITTI/2011_09_26/2011_09_26_drive_0001_sync/depth/0000000000.npy \
--init_calib /home/mark/Dev/BachelorThesis/repo/data/processed/KITTI/2011_09_26/2011_09_26_drive_0001_sync/decalibrations/0000000000.npy
''' # noqa
