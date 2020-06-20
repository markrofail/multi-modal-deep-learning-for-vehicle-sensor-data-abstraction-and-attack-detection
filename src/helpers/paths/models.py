from .. import paths


MODELS_PATH = paths.ROOT_PATH.joinpath('models')


def imagenet():
  path = MODELS_PATH.joinpath('nin_imagenet')
  return path


def regnet_tf():
  path = MODELS_PATH.joinpath('regnet-tf')
  return path


def multimodal():
  path = MODELS_PATH.joinpath('multimodal')
  return path
