import os

import click
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.helpers import paths
from src.helpers.flags import AttackModes, Verbose
from src.multimodal import multimodal
from src.multimodal.data import make_dataset

###############################################################################
# DATA PARAMETERS
###############################################################################
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(suppress=True, precision=4, sign=' ')
config = paths.config.read(paths.config.multimodal())
VERBOSE = config['ENVIROMENT_CONFIG']['VERBOSE']

DEFAULT_DRIVE = '2011_09_26'
DEFAULT_NUMBER = 1
DEFAULT_FRAME = 1
DEFAULT_ATTACK = 1
DEFAULT_ATTACK_FLAG = True


def print_results(**kargs):
  keys = np.array(list(kargs.keys()))
  keys = np.sort(keys)

  print('# results:')
  for key in keys:
    value = kargs[key]
    print('## {k} = {v}'.format(k=key, v=value))
  print()


def display_results(drive_date, drive_number, drive_frame, attack, result):
  if attack:
    img_path = paths.attack.interim_frame(drive_date, drive_number, drive_frame)
  else:
    img_path = paths.rgb.interim_frame(drive_date, drive_number, drive_frame)

  img = cv2.imread(str(img_path))
  if img is None:
    raise Exception("could not load image !")

  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  plt.imshow(img, interpolation="bicubic")

  result = 'result: {}!'.format('PASS' if result else 'FAIL')
  plt.xlabel(result)
  plt.show()


def feed_forward(input_image, input_depth, label):
  # Create the network
  net = multimodal.Multimodal()

  # Load pretrained model
  model_path = str(paths.checkpoints.multimodal())  # ./checkpoints/multimodal/train
  net.model.load_weights(model_path)

  # Predict
  pred = net.model.predict([[input_image], [input_depth]])

  print_args = dict()
  print_args['predict_raw'] = pred
  notations = ['Normal', 'Attack']

  pred = np.argmax(pred)
  label = np.argmax(label)
  verdict = 'PASS' if label == pred else 'FAIL'

  print_args['label'] = str(notations[label])
  print_args['predict'] = str(notations[pred])
  print_args['verdict'] = verdict
  print_results(**print_args)

  return label == pred


def predict(drive_date, drive_number, frame, attack, attack_type):
  # create the frame data
  print('# generating data...')
  frame_data = [(drive_date, drive_number, frame)]
  make_dataset.make_data(
      frame_data, attack_type=attack_type, verbose=Verbose.SILENT, keep=True)

  if attack:
    rgb_path = paths.attack.processed_tensor(drive_date, drive_number, frame)
    label_data = np.array([0, 1])
  else:
    rgb_path = paths.rgb.processed_tensor(drive_date, drive_number, frame)
    label_data = np.array([1, 0])

  rgb_data = np.load(rgb_path)

  depth_path = paths.depth.processed_tensor(drive_date, drive_number, frame)
  depth_data = np.load(depth_path)

  print('# feedforward data...')
  return feed_forward(input_image=rgb_data, input_depth=depth_data, label=label_data)


def get_arguments_interactively():
  args_dict = dict()
  print('\nEnter drive details:')

  default_drive = DEFAULT_DRIVE
  input_date = input(
      '# drive date (format: \'yyyy_mm_dd\') [\'{}\']:'.format(
          default_drive))
  if input_date:
    args_dict['drive_date'] = input_date

  default_number = DEFAULT_NUMBER
  input_number = input('# drive number (int) [{}]:'.format(default_number))
  if input_number:
    args_dict['drive_number'] = int(input_number)

  default_frame = DEFAULT_FRAME
  input_frame = input('# drive frame (int) [{}]:'.format(default_frame))
  if input_frame:
    args_dict['drive_frame'] = int(input_frame)

  default_attack_flag = DEFAULT_ATTACK_FLAG
  input_attack_flag = input('# normal/attack (0/1) [{}]:'.format(int(default_attack_flag)))
  if input_attack_flag:
    args_dict['attack'] = bool(int(input_attack_flag))

  if input_attack_flag and not args_dict['attack']:
    print()
    return args_dict

  default_attack = DEFAULT_ATTACK
  input_attack = input('# inpainting/translation (1/2) [{}]:'.format(int(default_attack)))
  if input_attack:
    args_dict['attack_type'] = int(input_attack)

  print()
  return args_dict


@click.option(
    '--drive_date', type=str, default=DEFAULT_DRIVE, help='date of the drive')
@click.option(
    '--drive_number',
    type=int,
    default=DEFAULT_NUMBER,
    help='date of the drive')
@click.option(
    '--drive_frame',
    type=int,
    default=DEFAULT_FRAME,
    help='frame within the drive')
@click.option(
    '--attack_type',
    type=int,
    default=DEFAULT_ATTACK,
    help='frame within the drive')
@click.option(
    '--attack/--normal',
    help='normal or attack',
)
@click.option(
    '--interactive',
    '-i',
    is_flag=True,
    help='interactively',
)
@click.option(
    '--display',
    '-d',
    is_flag=True,
    help='display image',
)
@click.command()
def main(interactive=False,
         display=False,
         drive_date=DEFAULT_DRIVE,
         drive_number=DEFAULT_NUMBER,
         drive_frame=DEFAULT_FRAME,
         attack_type=DEFAULT_ATTACK,
         attack=True):
  if interactive:
    args_dict = get_arguments_interactively()
    if 'drive_date' in args_dict:
      drive_date = args_dict['drive_date']
    if 'drive_number' in args_dict:
      drive_number = args_dict['drive_number']
    if 'drive_frame' in args_dict:
      drive_frame = args_dict['drive_frame']
    if 'attack_type' in args_dict:
      attack_type = args_dict['attack_type']
    if 'attack' in args_dict:
      attack = args_dict['attack']

  path = paths.rgb.external_frame(drive_date, drive_number, drive_frame)
  assert path.exists(), 'frame does not exist'

  result = predict(drive_date, drive_number, drive_frame, attack, attack_type)

  if display:
    display_results(drive_date, drive_number, drive_frame, attack, result)


if __name__ == '__main__':
  main()


'''
# for attack:
python -m src.multimodal.models.predict_model -d \
--drive_date 2011_09_26 --drive_number 1 --drive_frame 1 --attack --attack_type 2

# for normal:
python -m src.multimodal.models.predict_model -d \
--drive_date 2011_09_26 --drive_number 1 --drive_frame 1 --normal --attack_type 2
'''
