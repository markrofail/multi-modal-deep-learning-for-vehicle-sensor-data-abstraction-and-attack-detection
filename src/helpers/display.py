import math

import cv2
import matplotlib.pyplot as plt


def preview(*images, cvt=True, ncols=None):
  nimages = len(images)
  if ncols is None:
    ncols = math.ceil(math.sqrt(nimages))
  nrows = nimages // ncols

  fig = plt.figure()
  for idx in range(nimages):
    nplot = nrows * 100 + ncols * 10 + idx + 1

    ax = fig.add_subplot(nplot)
    if cvt:
      ax.imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB))
    else:
      ax.imshow(images[idx])
  plt.show()
