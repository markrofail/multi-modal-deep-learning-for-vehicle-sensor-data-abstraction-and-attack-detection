from .. import paths


def regnet():
  path = paths.ROOT_PATH.joinpath('config_regnet.yml')
  return path


def multimodal():
  path = paths.ROOT_PATH.joinpath('config_multimodal.yml')
  return path


def read(path):
  import yaml
  with open(path, "r") as f:
    return yaml.load(f)
