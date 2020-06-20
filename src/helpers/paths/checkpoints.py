from .. import paths

CHECKPOINT_PATH = paths.ROOT_PATH.joinpath('checkpoints')


def regnet(rot=None, disp=None):
  if rot is not None and disp is not None:
    file_string = 'training_{:02d}_{:02d}.hdf5'.format(rot, int(disp*10))
  else:
    file_string = 'training.hdf5'

  path = CHECKPOINT_PATH.joinpath('regnet', file_string)
  return path


def multimodal():
  path = CHECKPOINT_PATH.joinpath('multimodal', 'training.hdf5')
  return path
