import numpy as np

from src.regnet.helpers.quaternion import mat4_op


def mat4_dualqt(dq):
  """
  Converts a dual-quaternion to a 4x4 matrix
  :param dq: a dual-quaternion
  :returns: resulting 4x4 matrix
  """
  dq = np.array(dq)
  dq = np.split(dq.flatten(), 2)
  dq = np.append(dq[0] / mat4_op.CONST_f, dq[1])

  out = np.zeros(16, dtype=float)

  # rotation matrix
  out[0] = 1.0 - (2.0 * dq[1] ** 2) - (2.0 * dq[2] ** 2)
  out[1] = (2.0 * dq[0] * dq[1]) + (2.0 * dq[3] * dq[2])
  out[2] = (2.0 * dq[0] * dq[2]) - (2.0 * dq[3] * dq[1])
  out[4] = (2.0 * dq[0] * dq[1]) - (2.0 * dq[3] * dq[2])
  out[5] = 1.0 - (2.0 * dq[0] ** 2) - (2.0 * dq[2] ** 2)
  out[6] = (2.0 * dq[1] * dq[2]) + (2.0 * dq[3] * dq[0])
  out[8] = (2.0 * dq[0] * dq[2]) + (2.0 * dq[3] * dq[1])
  out[9] = (2.0 * dq[1] * dq[2]) - (2.0 * dq[3] * dq[0])
  out[10] = 1.0 - (2.0 * dq[0] * dq[0]) - (2.0 * dq[1] * dq[1])

  # translation vector
  out[3] = 2.0 * (-dq[7] * dq[0] + dq[4] * dq[3] - dq[5] * dq[2] + dq[6] * dq[1])
  out[7] = 2.0 * (-dq[7] * dq[1] + dq[4] * dq[2] + dq[5] * dq[3] - dq[6] * dq[0])
  out[11] = 2.0 * (-dq[7] * dq[2] - dq[4] * dq[1] + dq[5] * dq[0] + dq[6] * dq[3])

  out[15] = 1
  out = np.reshape(out, (4, 4))
  # out = np.around(out, decimals=9)
  return out
