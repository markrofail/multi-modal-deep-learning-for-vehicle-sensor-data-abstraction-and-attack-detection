import click
from click import echo

from src.helpers import frame_selector, paths, timeit
from src.helpers.flags import AttackModes, Verbose
from src.multimodal.data import kitti
from src.regnet import data as regnet_data

if __name__ == '__main__':
  print('Finished Importing')

###############################################################################
# ENVIROMENT PARAMETERS                                                       #
###############################################################################
config = paths.config.read(paths.config.multimodal())
VERBOSE = config['ENVIROMENT_CONFIG']['VERBOSE']


###############################################################################
# DATASET PARAMETERS                                                          #
###############################################################################
ATTACK_TYPE = config['DATASET']['ATTACK_TYPE']
COMBINED = config['DATASET']['COMBINED']
RANDOM = config['DATASET']['RANDOM']

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
@timeit.logtime
def main(keep=False):
  if COMBINED:
    select_and_generate(attack_type=AttackModes.INPAINT, random=RANDOM, keep=keep)
    select_and_generate(attack_type=AttackModes.TRANSLATE, random=RANDOM, keep=keep)
  else:
    select_and_generate(attack_type=ATTACK_TYPE, random=RANDOM, keep=keep)


def select_and_generate(keep=False, random=RANDOM, attack_type=ATTACK_TYPE):
  # get dataset frames
  if random:
    # getting the csv files directory
    attack_str = 'dataset_{}'.format(str(AttackModes(attack_type).name).lower())
    csv_directory = paths.checkpoints.multimodal().with_name(attack_str)

    # choose the train, valid, test datasets randomly
    datasets = frame_selector.generate_random(frame_count=NUM_FRAMES,
                                              train_pct=TRAIN_PCT,
                                              valid_pct=VALID_PCT,
                                              test_pct=TEST_PCT,
                                              log_path=csv_directory)
  else:
    # read selected frames from regnet
    csv_directory = paths.checkpoints.regnet().parent.joinpath('dataset')
    datasets = frame_selector.read_frames(csv_directory=csv_directory)

  # generate each dataset
  datasets = zip(datasets, ['train', 'valid', 'test'])
  for dataset, name in datasets:
    make_data(dataset, name, attack_type=attack_type, keep=keep)


def make_data(frames, name, attack_type=ATTACK_TYPE, verbose=VERBOSE, keep=False):
  if verbose > Verbose.SILENT:
    attack_str = str(AttackModes(attack_type).name).lower()
    print('\nGenerating {model}/{dataset} dataset: {attack} attack...'.format(model='multimodal',
                                                                              attack=attack_str,
                                                                              dataset=name.upper()))

  # step 1: get results from mrcnn
  kitti.make_mrcnn.generate_frames(frames, verbose=verbose)

  # step 2: get object mask from mrcnn results
  kitti.make_masks.generate_frames(frames, padding=True, verbose=verbose)

  # step 3: generate the attack frames
  if attack_type == AttackModes.INPAINT:
    kitti.make_inpainting.generate_frames(frames, verbose=verbose,
                                          mode=kitti.make_inpainting.InpaintingModes.MAGENTA)
  elif attack_type == AttackModes.TRANSLATE:
    kitti.make_translate.generate_frames(frames, padding=True, verbose=verbose)

  # step 3.1: generate log file specifying difficulty
  kitti.make_log.generate_frames(frames, attack_type=attack_type, verbose=verbose)
  kitti.make_mrcnn.delete_frames(frames, keep=keep)

  # step 4: project depthmaps
  regnet_data.kitti.make_depthmaps.generate_frames(frames, verbose=verbose)

  # step 5: scale and crop images to fit tensors
  regnet_data.kitti.image_rescale.preproccess_frames(frames, verbose=verbose, keep=keep)
  kitti.image_rescale.preproccess_frames(frames, verbose=verbose, keep=keep)

  # step 6: convert images to tensors
  regnet_data.kitti.make_tensors.generate_frames(frames, data_type='rgb',
                                                 verbose=verbose, keep=keep)
  regnet_data.kitti.make_tensors.generate_frames(frames, data_type='depth',
                                                 verbose=verbose, keep=keep)
  kitti.make_tensors.generate_frames(frames, verbose=verbose, keep=keep)

  # step 7: check that every tuple is sane (i.e has rgb and depth components)
  kitti.data_sanity.check_frames(frames, verbose=verbose)

  # step 8: write tensors as tfrecord format
  kitti.make_records.generate_frames(frames, dataset=name, attack_type=attack_type,
                                     verbose=verbose, keep=keep)


if __name__ == '__main__':
  main()
