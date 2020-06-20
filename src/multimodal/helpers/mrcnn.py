import os

import mrcnn.model as modellib
from mrcnn import utils, visualize
from mrcnn.config import Config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
ROOT_DIR = visualize.ROOT_DIR
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

"""
Taken from Mask_RCNN's demo notebook, that can be found here
https://github.com/matterport/Mask_RCNN/blob/master/samples/demo.ipynb
"""


class InferenceConfig(Config):
  """Configuration for training on MS COCO.
  Derives from the base Config class and overrides values specific
  to the COCO dataset.
  """
  NAME = "coco"
  IMAGES_PER_GPU = 1
  GPU_COUNT = 1

  # Number of classes (including background)
  NUM_CLASSES = 1 + 80  # COCO has 80 classes


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=InferenceConfig())

# Load weights trained on MS-COCO
if not os.path.exists(COCO_MODEL_PATH):
  utils.download_trained_weights(COCO_MODEL_PATH)
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


def generate_mask(image):
  # Run detection
  results = model.detect([image])

  # Visualize results
  r = results[0]
  mask = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names,
                                     r['scores'])

  return mask


def get_mask(image, r):
  # Visualize results
  mask = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names,
                                     r['scores'])

  return mask


def get_results(image):
  # Run detection
  results = model.detect([image])
  return results[0]
