from src.helpers.flags import AttackModes

from .. import paths


def raw_frame(drive_date, drive_number, frame):
  drive_string = '{}_drive_{:04d}_sync'.format(drive_date, drive_number)
  frame_string = '{:010d}.png'.format(frame)

  path = paths.DATA_RAW_PATH.joinpath('KITTI', drive_date, drive_string,
                                      'attack', frame_string)
  return path


def interim_frame(drive_date, drive_number, frame):
  drive_string = '{}_drive_{:04d}_sync'.format(drive_date, drive_number)
  frame_string = '{:010d}.png'.format(frame)

  path = paths.DATA_INTERIM_PATH.joinpath('KITTI', drive_date, drive_string,
                                          'attack', frame_string)
  return path


def processed_tensor(drive_date, drive_number, frame):
  drive_string = '{}_drive_{:04d}_sync'.format(drive_date, drive_number)
  frame_string = '{:010d}.npy'.format(frame)

  path = paths.DATA_PROCESSED_PATH.joinpath('KITTI', drive_date, drive_string,
                                            'attack', frame_string)
  return path


def log_file(drive_date, drive_number, frame, attack_type):
  drive_string = '{}_drive_{:04d}_sync'.format(drive_date, drive_number)
  frame_string = '{:010d}.csv'.format(frame)
  attack_string = 'log_{}'.format(str(AttackModes(attack_type).name).lower())

  path = paths.DATA_PROCESSED_PATH.joinpath('KITTI', drive_date, drive_string,
                                            'attack', attack_string, frame_string)
  return path
