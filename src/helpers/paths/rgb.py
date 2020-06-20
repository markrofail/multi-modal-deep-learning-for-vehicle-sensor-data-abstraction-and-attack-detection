from .. import paths


def external_frame(drive_date, drive_number, frame):
  drive_string = '{}_drive_{:04d}_sync'.format(drive_date, drive_number)
  frame_string = '{:010d}.png'.format(frame)

  path = paths.DATA_EXTERNAL_PATH.joinpath('KITTI', drive_date, drive_string,
                                           'image_02', 'data', frame_string)
  return path


def interim_frame(drive_date, drive_number, frame):
  drive_string = '{}_drive_{:04d}_sync'.format(drive_date, drive_number)
  frame_string = '{:010d}.png'.format(frame)

  path = paths.DATA_INTERIM_PATH.joinpath('KITTI', drive_date, drive_string,
                                          'rgb', frame_string)
  return path


def processed_tensor(drive_date, drive_number, frame):
  drive_string = '{}_drive_{:04d}_sync'.format(drive_date, drive_number)
  frame_string = '{:010d}.npy'.format(frame)

  path = paths.DATA_PROCESSED_PATH.joinpath('KITTI', drive_date, drive_string,
                                            'rgb', frame_string)
  return path
