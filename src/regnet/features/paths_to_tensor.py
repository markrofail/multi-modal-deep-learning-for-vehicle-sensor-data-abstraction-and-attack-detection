import numpy as np
from tqdm import tqdm
from keras.preprocessing import image


def one(path, grayscale=False, target_size=None):
  """"
  Taken from https://github.com/udacity/br-machine-learning/blob/master/projects/dog-project/
      dog_app.ipynb
  :param path: path to image
  :param grayscale: boolean, indicates if image is grayscale
  :param target_size: int tuple (img_height, img_width)
  :return: tensor
  """
  # loads RGB image as PIL.Image.Image type
  img = image.load_img(path, grayscale=grayscale, target_size=target_size)
  # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
  x = image.img_to_array(img)
  # convert 3D tensor to 4D tensor
  return np.expand_dims(x, axis=0)


def many(paths, grayscale=False, target_size=None):
  """"
  Taken from https://github.com/udacity/br-machine-learning/blob/master/projects/dog-project/
      dog_app.ipynb
  :param img_paths: sequence of paths to images
  :param path: path to image
  :param grayscale: boolean, indicates if image is grayscale
  :param target_size: int tuple (img_height, img_width)
  :return: sequence of tensors
  """
  list_of_tensors = [one(img_path, grayscale, target_size) for img_path in tqdm(paths)]
  return np.vstack(list_of_tensors)
