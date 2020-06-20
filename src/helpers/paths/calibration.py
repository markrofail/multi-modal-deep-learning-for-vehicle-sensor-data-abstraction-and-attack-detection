from .. import paths


def _get_calibration_file(drive_date, filename):
  path = paths.DATA_EXTERNAL_PATH.joinpath('KITTI', drive_date, filename)
  return path


def velo_to_cam(drive_date):
  return _get_calibration_file(drive_date, 'calib_velo_to_cam.txt')


def cam_to_cam(drive_date):
  return _get_calibration_file(drive_date, 'calib_cam_to_cam.txt')


def decalibration_matrix(drive_date, drive_number, frame):
  drive_string = '{}_drive_{:04d}_sync'.format(drive_date, drive_number)
  frame_string = '{:010d}.npy'.format(frame)

  path = paths.DATA_PROCESSED_PATH.joinpath('KITTI', drive_date, drive_string,
                                            'decalibrations', frame_string)
  return path
