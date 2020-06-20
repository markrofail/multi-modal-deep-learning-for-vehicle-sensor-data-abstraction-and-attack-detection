import numpy as np


def quat_mult(a, b):
  """
  Multiplies two quat's
  :param: a the first operand
  :param: b the second operand
  :retruns: resulting quat
  """
  ax, ay, az, aw = a[0], a[1], a[2], a[3]
  bx, by, bz, bw = b[0], b[1], b[2], b[3]
  out = np.zeros((4), dtype=float)

  out[0] = ax * bw + aw * bx + ay * bz - az * by
  out[1] = ay * bw + aw * by + az * bx - ax * bz
  out[2] = az * bw + aw * bz + ax * by - ay * bx
  out[3] = aw * bw - ax * bx - ay * by - az * bz
  # out[0] = ax * bx - ay * by - az * bz - aw * bw
  # out[1] = ax * by + ay * bx - az * bw + aw * bz
  # out[2] = ax * bz + ay * bw + az * bx - aw * by
  # out[3] = ax * bw - ay * bz + az * by + aw * bx
  return out


def quat_scale(a, b):
  """
  Scales a quat by a scalar number
  :param: the quat
  :param: scaling factor
  :retruns: resulting quat
  """
  out = np.zeros((4), dtype=float)
  out[0] = a[0] * b
  out[1] = a[1] * b
  out[2] = a[2] * b
  out[3] = a[3] * b
  return out
