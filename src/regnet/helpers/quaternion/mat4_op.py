import numpy as np

from .helpers import mat3_mat4, vec_mat4
from .mat3_op import quat_mat3
from .quat_op import quat_mult, quat_scale

CONST_f = 100


def dualqt_mat4(matrix4x4):
  """
  Converts a 4x4 Matrix to dual-quaternion representation
  :param matrix4x4: a 4x4 matrix
  :returns: resulting dual-quaternion
  """
  rotation_matrix = mat3_mat4(matrix4x4)
  rotation_quat = quat_mat3(rotation_matrix)

  trans_vec = vec_mat4(matrix4x4)
  trans_quat = quat_scale(quat_mult(trans_vec, rotation_quat), 0.5)

  rotation_quat *= CONST_f
  dual_quat = np.append(rotation_quat, trans_quat)
  return dual_quat
