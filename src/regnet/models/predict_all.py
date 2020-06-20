import click
import numpy as np

from src.helpers import display, paths
from src.helpers.flags import Verbose
from src.regnet.data import make_dataset
from src.regnet.models import predict_model


###############################################################################
# TRAIN DATA                                                                  #
###############################################################################
TRAIN_DRIVE_DATE = '2011_09_26'
TRAIN_DRIVE_NUMBER = 1
TRAIN_DRIVE_FRAME = 2
TRAIN_DATA = TRAIN_DRIVE_DATE, TRAIN_DRIVE_NUMBER, TRAIN_DRIVE_FRAME
###############################################################################
# VALID DATA                                                                  #
###############################################################################
VALID_DRIVE_DATE = '2011_09_26'
VALID_DRIVE_NUMBER = 70
VALID_DRIVE_FRAME = 0
VALID_DATA = VALID_DRIVE_DATE, VALID_DRIVE_NUMBER, VALID_DRIVE_FRAME
###############################################################################
# TEST DATA                                                                   #
###############################################################################
TEST_DRIVE_DATE = '2011_09_30'
TEST_DRIVE_NUMBER = 28
TEST_DRIVE_FRAME = 2
TEST_DATA = TEST_DRIVE_DATE, TEST_DRIVE_NUMBER, TEST_DRIVE_FRAME


def predict(drive_date, drive_number, frame):
  rgb_path = paths.rgb.processed_tensor(drive_date, drive_number, frame)
  rgb_data = np.load(rgb_path)

  depth_path = paths.depth.processed_tensor(drive_date, drive_number, frame)
  depth_data = np.load(depth_path)

  label_path = paths.calibration.decalibration_matrix(drive_date, drive_number, frame)
  label_data = np.load(label_path)

  predict_model.feed_forward(input_rgb=rgb_data, input_depth=depth_data, h_init=label_data,
                             drive_date=drive_date)

  # show prediction
  img_before = predict_model.project_depth(drive_date, drive_number, frame)
  img_after = predict_model.project_depth(drive_date, drive_number, frame,
                                          use_prediction=True)
  return img_before, img_after


click.command()
def main():
  print('\nGenerating inputs...')
  make_dataset.make_data([TRAIN_DATA], 'train', verbose=Verbose.SILENT, keep=True)
  make_dataset.make_data([VALID_DATA], 'valid', verbose=Verbose.SILENT, keep=True)
  make_dataset.make_data([TEST_DATA], 'test', verbose=Verbose.SILENT, keep=True)
  print()

  print('Predicting Training results:')
  train_img_before, train_img_after = predict(*TRAIN_DATA)
  print()

  print('Predicting Validation results:')
  valid_img_before, valid_img_after = predict(*VALID_DATA)
  print()

  print('Predicting Test results:')
  test_img_before, test_img_after = predict(*TEST_DATA)
  print()

  display.preview(
      train_img_before, valid_img_before, test_img_before,
      train_img_after, valid_img_after, test_img_after,
      ncols=3)


if __name__ == '__main__':
  main()
