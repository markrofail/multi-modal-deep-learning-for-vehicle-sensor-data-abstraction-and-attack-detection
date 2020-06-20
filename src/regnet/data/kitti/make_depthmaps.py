import multiprocessing

import cv2
from joblib import Parallel, delayed
from tqdm import tqdm

from src.helpers import paths
from src.helpers.flags import Verbose
from src.regnet.helpers import kitti_foundation


def _generate_frame(drive_date, drive_number, frame, calib_batches=False, color=None):
  """ save one frame about projecting velodyne points into camera image """
  velo_path = paths.depth.external_frame(drive_date, drive_number, frame)
  if not velo_path.exists():
    return

  image_path = paths.rgb.external_frame(drive_date, drive_number, frame)
  image = cv2.imread(str(image_path))

  velo_path = paths.depth.external_frame(drive_date, drive_number, frame)
  velo_points = kitti_foundation.load_from_bin(str(velo_path))

  v2c_filepath = paths.calibration.velo_to_cam(drive_date)
  c2c_filepath = paths.calibration.cam_to_cam(drive_date)

  if calib_batches:
    v2c_filepath = paths.calibration.decalibration_matrix(drive_date=drive_date,
                                                          drive_number=drive_number,
                                                          frame=frame)
    v2c_filepath = v2c_filepath.with_suffix('.hinit.txt')

  ans, c_ = kitti_foundation.velo3d_2_camera2d_points(velo_points,
                                                      v_fov=(-24.9, 2.0),
                                                      h_fov=(-45, 45),
                                                      vc_path=v2c_filepath,
                                                      cc_path=c2c_filepath,
                                                      mode='02')

  result = kitti_foundation.print_projection_cv2(points=ans,
                                                 color=c_,
                                                 image=image,
                                                 black_background=True)
  result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

  output_path = paths.depth.raw_frame(drive_date=drive_date,
                                      drive_number=drive_number,
                                      frame=frame)

  output_path.parent.mkdir(exist_ok=True, parents=True)  # ensure directory exists
  cv2.imwrite(img=result, filename=str(output_path))


def generate_frames(frames, color=None, calib_batches=False, verbose=Verbose.NORMAL):
  if verbose > Verbose.SILENT:
    info = '# generating depthmaps     '  # for logging purposes
    frames = tqdm(frames, ascii=True, desc=info)

  n_jobs = multiprocessing.cpu_count() // 2
  Parallel(n_jobs=n_jobs)(
      delayed(_generate_frame)(*frame_info, calib_batches, color)
      for frame_info in frames)
