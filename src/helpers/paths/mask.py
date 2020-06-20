from .. import paths


def mrcnn_pickle(drive_date, drive_number, frame):
  drive_string = '{}_drive_{:04d}_sync'.format(drive_date, drive_number)
  frame_string = '{:010d}.pickle'.format(frame)

  path = paths.DATA_RAW_PATH.joinpath('KITTI', drive_date, drive_string,
                                      'mrcnn', frame_string)
  return path


def raw_frame(drive_date, drive_number, frame):
  drive_string = '{}_drive_{:04d}_sync'.format(drive_date, drive_number)
  frame_string = '{:010d}.png'.format(frame)

  path = paths.DATA_RAW_PATH.joinpath('KITTI', drive_date, drive_string,
                                      'mask', frame_string)
  return path
