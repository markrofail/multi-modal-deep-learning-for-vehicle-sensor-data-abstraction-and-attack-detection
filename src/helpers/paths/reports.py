from .. import paths

REPORTS_PATH = paths.ROOT_PATH.joinpath('reports')


def regnet():
  path = REPORTS_PATH.joinpath('regnet')
  return path


def multimodal():
  path = REPORTS_PATH.joinpath('multimodal')
  return path
