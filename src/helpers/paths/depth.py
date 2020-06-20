from .. import paths


def external_frame(drive_date, drive_number, frame):
  drive_string = '{}_drive_{:04d}_sync'.format(drive_date, drive_number)
  frame_string = '{:010d}.bin'.format(frame)

  path = paths.DATA_EXTERNAL_PATH.joinpath('KITTI', drive_date, drive_string,
                                           'velodyne_points', 'data', frame_string)
  return path


def raw_frame(drive_date, drive_number, frame):
  drive_string = '{}_drive_{:04d}_sync'.format(drive_date, drive_number)
  frame_string = '{:010d}.png'.format(frame)

  path = paths.DATA_RAW_PATH.joinpath('KITTI', drive_date, drive_string,
                                      'depth_maps', frame_string)
  return path


def interim_frame(drive_date, drive_number, frame):
  drive_string = '{}_drive_{:04d}_sync'.format(drive_date, drive_number)
  frame_string = '{:010d}.png'.format(frame)

  path = paths.DATA_INTERIM_PATH.joinpath('KITTI', drive_date, drive_string,
                                          'depth', frame_string)
  return path


def processed_tensor(drive_date, drive_number, frame):
  drive_string = '{}_drive_{:04d}_sync'.format(drive_date, drive_number)
  frame_string = '{:010d}.npy'.format(frame)

  path = paths.DATA_PROCESSED_PATH.joinpath('KITTI', drive_date, drive_string,
                                            'depth', frame_string)
  return path
