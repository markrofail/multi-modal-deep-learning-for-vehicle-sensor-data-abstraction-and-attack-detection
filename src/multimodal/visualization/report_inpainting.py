import glob

import cv2
import numpy as np

from src.helpers import paths

RAW_PATH = paths.DATA_RAW_PATH.joinpath('KITTI')


def count_mask(algorithm_number):
  """counts nonzero pixels left by the given algorithm"""

  # create the pattern
  pattern = RAW_PATH.joinpath('**', 'inpaint', '*.png')
  mask_pattern = make_subfolder(pattern, algorithm_number)

  # find the files
  filenames = glob.glob(str(mask_pattern), recursive=True)

  # count the nonzeros
  return count_nonzero(filenames)


def count_nonzero(filenames):
  """counts nonzero pixels inside given images"""

  count = 0
  for file_ in filenames:
    image = cv2.imread(str(file_))
    count += np.count_nonzero(image != 0)
  return count


def make_subfolder(path, subfolder):
  """helper method that puts given file in given subfolder"""

  basename, dirname = path.name, path.parent
  output_dir = dirname.joinpath(subfolder, basename)
  return output_dir


def main():
  print('======================={}======================='
        .format('Inpainting Algorithm Statistics'))

  # count object pixels before inpainting
  mask0_pattern = RAW_PATH.joinpath('**', 'mask', '*.png')
  mask0_filenames = glob.glob(str(mask0_pattern), recursive=True)
  mask0_zeros = count_nonzero(mask0_filenames)
  print('+ Test across {} images\n\n'.format(len(mask0_filenames)))
  print('+ Before inpainting:\t{}'.format(mask0_zeros))

  print('\n\n+ Object pixels left by:')

  # count object pixels after CV Telea inpainting
  mask1_zeros = count_mask('mask1')
  mask1_per = (mask0_zeros - mask1_zeros) / mask0_zeros * 100
  print('- CV Telea:\t\t{}\t↓{:.2f}%'.format(mask1_zeros, mask1_per))

  # count object pixels after CV Navier-Stokes inpainting
  mask2_zeros = count_mask('mask2')
  mask2_per = (mask0_zeros - mask2_zeros) / mask0_zeros * 100
  print('- CV Navier-Stokes:\t{}\t↓{:.2f}%'.format(mask2_zeros, mask2_per))

  # count object pixels after skimage inpainting
  mask3_zeros = count_mask('mask3')
  mask3_per = (mask0_zeros - mask3_zeros) / mask0_zeros * 100
  print('- Skimage:\t\t{}\t↓{:.2f}%'.format(mask3_zeros, mask3_per))

  # count object pixels after magenta inpainting
  mask4_zeros = count_mask('mask4')
  mask4_per = (mask0_zeros - mask4_zeros) / mask0_zeros * 100
  print('- Magenta:\t\t{}\t↓{:.2f}%'.format(mask4_zeros, mask4_per))

  print('======================================={}======================================'
        .format(''))

if __name__ == '__main__':
  main()
