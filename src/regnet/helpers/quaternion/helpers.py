import numpy as np


def mat3_mat4(matrix4x4):
  """
  Extracts a 3x3 matix given a 4x4 matrix
  :param matrix4x4: 4x4 matrix
  :retruns: 3x3 matix
  """
  return np.array(matrix4x4[:3, :3])


def vec_mat4(matrix4x4):
  """
  Extracts the translation vector (last coloumn) from
  a 4x4 matrix
  :param matrix4x4: 4x4 matrix
  :retruns: translation vector
  """
  return np.append(matrix4x4[:3, 3], [[0]])
