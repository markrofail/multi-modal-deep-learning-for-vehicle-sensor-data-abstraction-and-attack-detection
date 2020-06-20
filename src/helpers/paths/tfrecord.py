from .. import paths


def dataset_frame(dataset, drive_date, drive_number, frame, extra_suffix=None):
  file_string = '{}_{:04d}_{:010d}.tfr'.format(drive_date, drive_number, frame)

  path = paths.DATA_PROCESSED_PATH.joinpath('KITTI', dataset, file_string)

  if extra_suffix is not None:
    path = path.with_suffix('{extra}{orig}'.format(extra=extra_suffix, orig=path.suffix))
  return path
