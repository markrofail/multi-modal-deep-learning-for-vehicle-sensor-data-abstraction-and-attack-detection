from .. import paths


def drive_exists(drive_date, drive_number):
  path = paths.rgb.external_frame(drive_date, drive_number, 0)
  return path.exists()
