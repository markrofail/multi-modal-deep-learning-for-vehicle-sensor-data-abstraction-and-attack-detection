from .. import paths

LOGS_PATH = paths.ROOT_PATH.joinpath('logs')


def regnet():
  path = LOGS_PATH.joinpath('regnet', 'train')
  return path


def multimodal():
  path = LOGS_PATH.joinpath('multimodal', 'train')
  return path
