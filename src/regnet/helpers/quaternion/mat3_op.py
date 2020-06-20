import numpy as np


def quat_mat3(m):
  """
  Creates a quaternion from the given 3x3 rotation matrix.
  :param m: rotation matrix
  :returns: the resulting quaternion
  """

  '''
  Algorithm in Ken Shoemake's article in 1987 SIGGRAPH course notes
  article "Quaternion Calculus and Fast Animation".
  '''

  m = m.flatten()
  out = np.zeros((4), dtype=float)

  fTrace = m[0] + m[4] + m[8]
  fRoot = None

  if fTrace > 0.0:
        # |w| > 1/2, may as well choose w > 1/2
    fRoot = np.sqrt(fTrace + 1.0)  # 2w
    out[3] = 0.5 * fRoot
    fRoot = 0.5 / fRoot  # 1/(4w)
    out[0] = (m[5] - m[7]) * fRoot
    out[1] = (m[6] - m[2]) * fRoot
    out[2] = (m[1] - m[3]) * fRoot
  else:
    # |w| <= 1/2
    i = 0
    if (m[4] > m[0]):
      i = 1
    if (m[8] > m[i * 3 + i]):
      i = 2
    j = (i + 1) % 3
    k = (i + 2) % 3

    fRoot = np.sqrt(m[i * 3 + i] - m[j * 3 + j] - m[k * 3 + k] + 1.0)
    out[i] = 0.5 * fRoot
    fRoot = 0.5 / fRoot
    out[3] = (m[j * 3 + k] - m[k * 3 + j]) * fRoot
    out[j] = (m[j * 3 + i] + m[i * 3 + j]) * fRoot
    out[k] = (m[k * 3 + i] + m[i * 3 + k]) * fRoot
  return out
