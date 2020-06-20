import multiprocessing
from enum import IntFlag

import cv2
import numpy as np
import skimage
from joblib import Parallel, delayed
from PIL import Image, ImageDraw, ImageFilter
from skimage.restoration import inpaint
from tqdm import tqdm

from src.helpers import paths
from src.helpers.flags import Verbose


class InpaintingModes(IntFlag):
  CV_TELEA = 1
  CV_NS = 2
  SKIMAGE = 3
  MAGENTA = 4


def patch(image, mask):   # pylint: disable=C2
  """
  Inapainting algorithm by Magenta
  source: https://codegolf.stackexchange.com/questions/70483/patch-the-image

  :param image: input image
  :param mask: input mask. must be same shape as image
  :return: patched image
  """
  im = Image.fromarray(image)
  im1 = Image.fromarray(mask)
  a = list(im.getdata())
  b = list(im1.getdata())
  size = list(im.size)

  C = []
  d = []
  y = []
  for x in range(0, len(a)):
    if b[x][0] == 255:
      C.append((0, 0, 0))
    else:
      y = (a[x][0], a[x][1], a[x][2])
      C.append(y)
  im1.putdata(C)
  k = (-1, 0, 1, -2, 0, 2, -1, 0, 1)
  k1 = (-1, -2, -1, 0, 0, 0, 1, 2, 1)
  ix = im.filter(ImageFilter.Kernel((3, 3), k, 1, 128))
  iy = im.filter(ImageFilter.Kernel((3, 3), k1, 1, 128))
  ix1 = list(ix.getdata())
  iy1 = list(iy.getdata())
  d = []
  im2 = Image.new('RGB', size)
  draw = ImageDraw.Draw(im2)
  c = list(C)
  Length = 0
  for L in range(100, 0, -10):
    for x in range(0, size[0]):
      for y in range(0, size[1]):
        n = x + (size[0] * y)
        if c[n] != (0, 0, 0):
          w = (((iy1[n][0] + iy1[n][1] + iy1[n][2]) // 3) - 128)
          z = (((ix1[n][0] + ix1[n][1] + ix1[n][2]) // 3) - 128)
          Length = (w**2 + z**2)**0.5
          if Length == 0:
            w += 1
            z += 1
          Length = (w**2 + z**2)**0.5
          w /= (Length / L)
          z /= (Length / L)
          w = int(w)
          z = int(z)
          draw.line(((x, y, w + x, z + y)), c[n])

  d = list(im2.getdata())
  S = []
  d1 = []

  for x in range(0, size[0]):
    for y in range(0, size[1]):
      n = y + (size[1] * x)
      nx = y + (size[1] * x) - 1
      ny = y + (size[1] * x) - size[0]
      if d[n] == (0, 0, 0):
        S = [0, 0, 0]
        for z in range(0, 3):
          S[z] = (d[nx][z] + d[ny][z]) // 2
        # print(S)
        d1.append(tuple(S))
      else:
        d1.append(tuple(d[n]))
  d = list(d1)
  im2.putdata(d)
  d = im2.getdata()
  f = []

  for v in range(0, len(a)):
    if b[v][0] * b[v][1] * b[v][2] != 0:
      f.append(d[v])
    else:
      f.append(C[v])

  im1.putdata(f)
  return np.array(im1)


def _generate_frame(drive_date, drive_number, frame,
                    mode=InpaintingModes.MAGENTA, verbose=Verbose.NORMAL):
  """
  generates inpainting image attack from a frame and saves it in the raw
  folder with the same structure as the original data, under inpaint folder

  Usage::
      >>> generate_frame('2011_09_26',  1, 0)

  :param drive_date: drive date (ex. '2011_09_26')
  :param drive_number: drive number (ex. 1)
  :param frame: frame within drive (ex. 0)
  :param mode: an int corresponding to an inpainting mode. check InpaintingModes enum.
  :param debug: boolean that indicates if you want to pass inpaint output to
      object detection algorithm
  """
  input_path = paths.rgb.external_frame(drive_date, drive_number, frame)
  image = cv2.imread(str(input_path))

  mask_path = paths.mask.raw_frame(drive_date, drive_number, frame)
  mask = cv2.imread(str(mask_path))

  output_path = paths.attack.raw_frame(drive_date, drive_number, frame)
  output_path.parent.mkdir(exist_ok=True, parents=True)  # ensure directory exists

  if mode == InpaintingModes.CV_TELEA:
    mask = cv2.split(mask)[0]
    image_result = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

  elif mode == InpaintingModes.CV_NS:
    mask = cv2.split(mask)[0]
    image_result = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)

  elif mode == InpaintingModes.SKIMAGE:
    mask = cv2.split(mask)[0]
    image_result = inpaint.inpaint_biharmonic(image, mask, multichannel=True)

  elif mode == InpaintingModes.MAGENTA:
    image_result = patch(image, mask)


  if mode == InpaintingModes.SKIMAGE:
    skimage.io.imsave(str(output_path), image_result)
  else:
    cv2.imwrite(img=image_result, filename=str(output_path))

  if verbose == Verbose.DEBUG:
    from src.multimodal.helpers import mrcnn  # noqa

    mask = mrcnn.get_mask(image_result)
    save_path_mask = paths.make_subfolder(output_path, 'mask')
    cv2.imwrite(img=mask, filename=str(save_path_mask))


def generate_frames(frames, verbose=Verbose.NORMAL, **kargs):
  """
  generates inpainting image attack from multiple frames and saves it in the
  raw folder with the same structure as the original data, under inpaint
  folder

  Usage::
      >>> generate_frames([('2011_09_26', 1, 1), ...])

  :param frames: array of frame info tuples, i.e (drive_date, dive_number, frame)
  """

  if verbose > Verbose.SILENT:
    info = '# generating inpainting attacks'  # for logging purposes
    frames = tqdm(frames, ascii=True, desc=info)

  n_jobs = multiprocessing.cpu_count() // 2
  Parallel(n_jobs=n_jobs)(
      delayed(_generate_frame)(*frame_info, verbose=verbose, **kargs)
      for frame_info in frames)
