# flake8: noqa
import numpy as np

from src.helpers import display

imgs = list()

FILE1 = '/home/mark/Dev/BachelorThesis/repo/data/processed/KITTI/2011_09_26/2011_09_26_drive_0001_sync/rgb/0000000002.npy'
img1 = np.load(file=FILE1)
imgs.append(img1)

FILE2 = '/home/mark/Dev/BachelorThesis/test_run/data/processed/KITTI/2011_09_26/2011_09_26_drive_0001_sync/rgb/0000000002.npy'
img2 = np.load(file=FILE2)
imgs.append(img2)

FILE3 = '/home/mark/Dev/BachelorThesis/repo/data/processed/KITTI/2011_09_26/2011_09_26_drive_0001_sync/depth/0000000002.npy'
img3 = np.load(file=FILE3)
img3 = np.tile(img3, 3)
imgs.append(img3)

FILE4 = '/home/mark/Dev/BachelorThesis/test_run/data/processed/KITTI/2011_09_26/2011_09_26_drive_0001_sync/depth/0000000002.npy'
img4 = np.load(file=FILE4)
img4 = np.tile(img4, 3)
imgs.append(img4)

display.preview(*imgs)
