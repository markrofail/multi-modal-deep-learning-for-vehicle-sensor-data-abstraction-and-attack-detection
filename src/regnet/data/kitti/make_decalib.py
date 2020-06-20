import multiprocessing

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from src.helpers import paths
from src.helpers.flags import Verbose
from src.regnet.helpers import kitti_foundation, quaternion

###############################################################################
# DECALIBRATION PARAMETERS                                                    #
###############################################################################
config = paths.config.read(paths.config.regnet())
RANGE_ROTATION = np.radians(config['DECALIBRATIONS']['RANGE_ROTATION'])
RANGE_DISPLACEMENT = config['DECALIBRATIONS']['RANGE_DISPLACEMENT']
DUAL_QUATERNIONS = config['DATA_INFORMATION']['DUAL_QUATERNIONS']


def uniform_two(a1, a2, b1, b2, size=1):
    # https://stackoverflow.com/questions/41792331/how-to-get-one-random-float-number-from-two-ranges-python
    # Calc weight for each range
  delta_a = a2 - a1
  delta_b = b2 - b1
  if np.random.rand() < delta_a / (delta_a + delta_b):
    return np.random.uniform(a1, a2, size=size)
  else:
    return np.random.uniform(b1, b2, size=size)


def get_Hgt(drive_date=None, path=None):
  assert drive_date is not None or path is not None, \
      'either \'path\' or \'drive_date\' should be set'

  if path is None:
    path = paths.calibration.velo_to_cam(drive_date)
  R_vc, T_vc = kitti_foundation.calib_velo2cam(path)

  velo2cam = np.eye(4)
  velo2cam[:3, :3] = R_vc.reshape(3, 3)
  velo2cam[:3, 3] = T_vc.reshape(3)
  return velo2cam


def save_as_txt(matrix, output_path):
  R = quaternion.helpers.mat3_mat4(matrix)
  R = R.flatten()
  R_str = ''
  for x in R:
    R_str += ' {}'.format(x)
  R_str = R_str[1:-1]

  T = quaternion.helpers.vec_mat4(matrix)
  T = T[:-1]
  T_str = ''
  for x in T:
    T_str += ' {}'.format(x)
  T_str = T_str[1:-1]

  f = open(output_path, "w")
  f.write('R: {}\n'.format(R_str))
  f.write('T: {}\n'.format(T_str))
  f.close()


def _generate_frame(drive_date, drive_number, frame, rot=RANGE_ROTATION, disp=RANGE_DISPLACEMENT,
                    verbose=Verbose.NORMAL):
  decal_mat = np.identity(4, dtype=float)

  x_t, y_t, z_t = tuple(np.random.uniform(low=-disp, high=disp, size=3))
  alpha, beta, gamma = tuple(np.random.uniform(low=-rot, high=rot, size=3))

  sin_a, cos_a = np.sin(alpha), np.cos(alpha)
  sin_b, cos_b = np.sin(beta), np.cos(beta)
  sin_c, cos_c = np.sin(gamma), np.cos(gamma)

  decal_mat[0][0] = cos_a * cos_b
  decal_mat[0][1] = cos_a * sin_b * sin_c - sin_a * cos_c
  decal_mat[0][2] = cos_a * sin_b * cos_c + sin_a * sin_c
  decal_mat[0][3] = x_t

  decal_mat[1][0] = sin_a * cos_b
  decal_mat[1][1] = sin_a * sin_b * sin_c + cos_a * cos_c
  decal_mat[1][2] = sin_a * sin_b * cos_c - cos_a * sin_c
  decal_mat[1][3] = y_t

  decal_mat[2][0] = -sin_b
  decal_mat[2][1] = cos_b * sin_c
  decal_mat[2][2] = cos_b * cos_c
  decal_mat[2][3] = z_t

  if verbose == Verbose.DEBUG:
    print('Decalibration Parameters:')
    print('Δx {:.4f}, Δy {:.4f}, Δz {:.4f}\n'.format(x_t, y_t, z_t))
    print('α {:.4f}, β {:.4f}, ɣ {:.4f}\n'.format(alpha, beta, gamma))

    print('sin α {:.4f},\t cos α {:.4f}'.format(sin_a, cos_a))
    print('sin β {:.4f},\t cos β {:.4f}'.format(sin_b, cos_b))
    print('sin ɣ {:.4f},\t cos ɣ {:.4f}\n'.format(sin_c, cos_c))

    print('Transformation Matrix \n{}\n'.format(decal_mat))

  output_path = paths.calibration.decalibration_matrix(drive_date, drive_number, frame)
  output_path.parent.mkdir(exist_ok=True, parents=True)  # ensure directory exists

  if DUAL_QUATERNIONS:
    decal_quat = quaternion.mat4_op.dualqt_mat4(decal_mat)
    np.save(arr=decal_quat, file=output_path.with_suffix(''))
  else:
    np.save(arr=decal_mat, file=output_path.with_suffix(''))

  h_gt = get_Hgt(drive_date)
  calib_matrix = np.dot(decal_mat, h_gt)

  txt_path = output_path.with_suffix('.hinit.txt')
  save_as_txt(calib_matrix, txt_path)
  return decal_mat


def generate_frames(frames, rot=RANGE_ROTATION, disp=RANGE_DISPLACEMENT, verbose=Verbose.NORMAL):
  if verbose > Verbose.SILENT:
    info = '# generating decalibrations'  # for logging purposes
    frames = tqdm(frames, ascii=True, desc=info)

  n_jobs = multiprocessing.cpu_count() // 2
  Parallel(n_jobs=n_jobs)(
      delayed(_generate_frame)(*frame_info, rot, disp, verbose)
      for frame_info in frames)
