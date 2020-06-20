import os

import click
import tensorflow as tf

from src.helpers import frame_selector, paths
from src.regnet import regnet
from src.regnet.data import make_dataset
from src.regnet.features import build_features

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def make_data():
  csv_directory = paths.checkpoints.regnet().with_name('dataset')
  *_, test_frames = frame_selector.read_frames(csv_directory)
  make_dataset.make_data(test_frames, name='test')


def evaluate():
  batch_dataset = build_features.get_test_batches(batch_size=1, infinite=False)
  itr_dataset = batch_dataset.make_one_shot_iterator()

  # Create the network
  *inputs, labels = itr_dataset.get_next()
  net = regnet.Regnet(hasPipeline=True, pipeline_inputs=inputs, pipeline_labels=labels)

  # Load pretrained model
  model_path = paths.checkpoints.regnet()
  net.model.load_weights(str(model_path))

  # Configures the model for evaluation
  metrics = [tf.losses.huber_loss]
  metrics.extend(net.metrics)
  net.model.compile(optimizer=net.train_opt, loss=net.model_loss,
                    target_tensors=[net.label], metrics=metrics)

  # Evaluate
  test_files = build_features.get_dataset_tensors('test')
  print('feed forward on {} test frames'.format(len(test_files)))

  loss = net.model.evaluate(steps=len(test_files))
  loss = zip(net.model.metrics_names, loss)

  for label, value in loss:
    print('{}: {}'.format(label, value))


@click.command()
def main():
  make_data()
  # evaluate()


if __name__ == '__main__':
  main()
