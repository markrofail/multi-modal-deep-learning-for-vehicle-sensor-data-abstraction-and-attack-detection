import click

from src.helpers import frame_selector, paths, timeit
from src.helpers.flags import Verbose
from src.regnet.data import kitti

###############################################################################
# ENVIROMENT PARAMETERS                                                       #
###############################################################################
config = paths.config.read(paths.config.regnet())
VERBOSE = config['ENVIROMENT_CONFIG']['VERBOSE']

###############################################################################
# DATASET PARAMETERS
###############################################################################
NUM_FRAMES = config['DATASET']['NUM_FRAMES']
TRAIN_PCT = config['DATASET']['TRAIN_PCT']
VALID_PCT = config['DATASET']['VALID_PCT']
TEST_PCT = config['DATASET']['TEST_PCT']


@click.command()
@click.option(
    '--keep', '-k',
    is_flag=True,
    help='keep redundant files',
)
@click.option(
    '--random/--kitti',
    is_flag=True,
)
@timeit.logtime
def main(keep=False, random=True):
  # choose the train, valid, test datasets randomly
  if random:
    datasets = frame_selector.generate_random(
        frame_count=NUM_FRAMES, train_pct=TRAIN_PCT, valid_pct=VALID_PCT, test_pct=TEST_PCT,
        log_path=paths.checkpoints.regnet().with_name('dataset'))
  else:
    train_drives = paths.get_all_drives('2011_09_26', exclude=[5, 70])
    train_dataset = list()
    for drive_info in train_drives:
      train_dataset.extend(paths.get_all_frames(*drive_info))

    valid_drives = [('2011_09_26', 5), ('2011_09_26', 70)]
    valid_dataset = list()
    for drive_info in valid_drives:
      valid_dataset.extend(paths.get_all_frames(*drive_info))

    test_dataset = paths.get_all_frames('2011_09_30', 28)

    datasets = [train_dataset, valid_dataset, test_dataset]

  # generate each dataset
  datasets = zip(datasets, ['train', 'valid', 'test'])
  for dataset, name in datasets:
    make_data(dataset, name, keep=keep)

  if not keep:
    cleanUp()


def make_data(frames, name, verbose=VERBOSE, keep=False):
  if verbose > Verbose.SILENT:
    print('\nGenerating regnet/{} dataset...'.format(name.upper()))

  # generate a unique random decalibration matrix for every frame
  kitti.make_decalib.generate_frames(frames=frames, verbose=verbose)

  # project each lidar scan using the frame's Hinit (Hinit = decalibration * Hgt)
  kitti.make_depthmaps.generate_frames(
      frames=frames, calib_batches=True, color='gray', verbose=verbose)

  # ensure all images are of the same size (resize and crop all images)
  kitti.image_rescale.preproccess_frames(frames=frames, verbose=verbose, keep=keep)

  # convert images to numpy arrays
  kitti.make_tensors.generate_frames(frames=frames, data_type='rgb', verbose=verbose, keep=keep)
  kitti.make_tensors.generate_frames(frames=frames, data_type='depth', verbose=verbose, keep=keep)

  # check that every frame's data is complete (every frame has an rgb image, depthmap and Hinit)
  kitti.data_sanity.check_frames(frames=frames, verbose=verbose)

  # encode every frame into a TFrecord
  kitti.make_records.generate_frames(frames=frames, dataset=name, verbose=verbose, keep=keep)


def cleanUp(verbose=VERBOSE, force_all=False):
  import shutil
  if verbose > Verbose.SILENT:
    print('\nRemoving redundant files to save space...')

  directories = [paths.DATA_RAW_PATH,
                 paths.DATA_INTERIM_PATH,
                 paths.DATA_PROCESSED_PATH.joinpath('KITTI')]

  exclude_folders = list()
  if not force_all:
    exclude_folders.extend(['train', 'test', 'valid'])

  exclude_files = ['.gitkeep']

  for directory in directories:
    if not directory.exists():
      continue

      if verbose > Verbose.SILENT:
        print(' # deleting {}/**'.format(directory.parent.name))

      for child in directory.iterdir():
        if child.is_dir():
          if child.name not in exclude_folders:
            shutil.rmtree(child, ignore_errors=True)

          else:
            if child.name not in exclude_files:
              child.unlink()


if __name__ == '__main__':
  main()
