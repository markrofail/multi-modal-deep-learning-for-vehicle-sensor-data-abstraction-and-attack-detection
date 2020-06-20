import multiprocessing

from joblib import Parallel, delayed
from tqdm import tqdm

from src.helpers import paths
from src.helpers.flags import Verbose


def _check_frame(drive_date, drive_number, frame):
  path_rgb = paths.rgb.processed_tensor(drive_date, drive_number, frame)

  path_depth = paths.depth.processed_tensor(drive_date, drive_number, frame)

  path_decalib = paths.calibration.decalibration_matrix(drive_date, drive_number, frame)

  path_arr = [path_rgb, path_depth, path_decalib]
  if not (path_rgb.exists() and path_depth.exists() and path_decalib.exists()):
    for f in path_arr:
      try:
        f.unlink()
      except FileNotFoundError:
        pass


def check_frames(frames, verbose=Verbose.NORMAL):
  if verbose > Verbose.SILENT:
    info = '# checking frame sanity    '  # for logging purposes
    frames = tqdm(frames, ascii=True, desc=info)

  n_jobs = multiprocessing.cpu_count() // 2
  Parallel(n_jobs=n_jobs)(
      delayed(_check_frame)(*frame_info)
      for frame_info in frames)
