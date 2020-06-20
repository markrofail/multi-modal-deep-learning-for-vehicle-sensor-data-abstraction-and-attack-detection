# pylint: disable=unsupported-assignment-operation
import pickle
import pprint

import bokeh.models as pltm
import bokeh.plotting as plt
import click
from bokeh.layouts import gridplot
from bokeh.models import widgets

from src.helpers import paths

# validtaion and training line colors
TRAINING_OPTIONS = dict([('legend', 'training'), ('color', '#0000cc')])
VALIDATION_OPTIONS = dict([('legend', 'validation'), ('color', '#00cc00')])

# validation lines refrence
VALIDATION_LINES = []
VALIDATION_CIRCLES = []


def draw_line(plot, line_x, line_y, legend, color='blue', dashed='solid'):
  line = plot.line(line_x, line_y, line_width=3, line_color=color, name='continous_loss',
                   line_dash=dashed, legend=legend)
  circles = plot.circle(line_x, line_y, size=7, fill_color='white', line_color=color,
                        name='discrete_loss')
  return line, circles


def draw_basic_graph(title):
  plot = plt.figure(x_axis_type="linear", title=title)
  plot.title.text_font_size = "16px"
  plot.title.align = "center"
  plot.grid.grid_line_alpha = 0.3
  plot.xaxis.axis_label = 'step'
  plot.yaxis.axis_label = title.lower()
  plot.legend.location = "top_left"

  hover = pltm.HoverTool(
      tooltips=[
          ("value", "@y{1.1111}"),
          ("iteration", "@x{int}"),
      ],
      names=["discrete_loss"]
  )
  plot.add_tools(hover)
  return plot


def draw_history(history, variable):
  plot = draw_basic_graph(variable.replace('_', ' ').capitalize())

  diff = len(history[variable]) // len(history['val_{}'.format(variable)])

  values_train_y = history[variable]
  values_train_x = range(len(values_train_y))

  values_valid_y = history['val_{}'.format(variable)]
  values_valid_x = range(0, len(values_train_x), diff)

  draw_line(plot, values_train_x, values_train_y, **TRAINING_OPTIONS)
  line_ref, c_ref = draw_line(plot, values_valid_x, values_valid_y, **VALIDATION_OPTIONS)

  VALIDATION_LINES.append(line_ref)
  VALIDATION_CIRCLES.append(c_ref)
  return plot


def draw_euclidean_loss(history):
  return draw_history(history, 'loss')


def draw_mae(history):
  return draw_history(history, 'mean_absolute_error')


def draw_r2_score(history):
  return draw_history(history, 'r2_score')


def load_data(input_path=None):
  if input_path is None:
    input_path = paths.checkpoints.regnet().with_suffix('.pkl')

  with open(input_path, 'rb') as handle:
    results = pickle.load(handle)

  return results


def make_callback():
  # We write coffeescript to link toggle with visible property of box and line
  code = '''\
  l0.visible = toggle.active
  l1.visible = toggle.active
  l2.visible = toggle.active


  c0.visible = toggle.active
  c1.visible = toggle.active
  c2.visible = toggle.active
  '''

  cbs = pltm.CustomJS.from_coffeescript(code=code, args={})
  btn = pltm.Toggle(label="Hide/Show Validation", button_type="success",
                            callback=cbs, active=True, width=200)
  cbs.args['toggle'] = btn

  idx = 0
  for line in VALIDATION_LINES:
    cbs.args['l{}'.format(idx)] = line
    idx += 1

  idx = 0
  for circle in VALIDATION_CIRCLES:
    cbs.args['c{}'.format(idx)] = circle
    idx += 1

  return btn

@click.command()
def main():
  history = load_data('/home/mark/Dev/BachelorThesis/RegnetCheckpoints/[5][CROPPED][Matricies].pkl')
  plots = list()

  title = 'Regnet results random dataset'
  title_div = widgets.Div(text="""<h1>{}</h1>""".format(title), width=200, height=20)
  plots.extend([title_div, None])

  plot1 = draw_euclidean_loss(history)
  plots.append(plot1)

  plot2 = draw_mae(history)
  plots.append(plot2)

  plot3 = draw_r2_score(history)
  plots.append(plot3)

  btn = make_callback()
  plots.extend([None, btn])

  reports_path = paths.reports.regnet()
  reports_path = reports_path.joinpath('figures', '{}.html'.format(title))
  reports_path.parent.mkdir(exist_ok=True, parents=True)

  plt.output_file(reports_path, title=title)
  plt.show(gridplot(children=plots, ncols=2, sizing_mode='scale_width'))

def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))


def concat(value):
  root_path = '/home/mark/Dev/BachelorThesis/RegnetCheckpoints/[5][CROPPED][Matricies]'

  history_90 = load_data(root_path+'[90].pkl')
  history_160 = load_data(root_path+'[160].pkl')
  history_180 = load_data(root_path+'[180].pkl')
  history_200 = load_data(root_path+'[200].pkl')

  values = list()
  values.extend(history_90[value][:-8])
  values.extend(history_160[value][:-5])
  values.extend(history_180[value][:-3])
  values.extend(history_200[value][:-9])
  return values

import pprint
if __name__ == '__main__':
  main()

  # history_90 = load_data('/home/mark/Dev/BachelorThesis/RegnetCheckpoints/[5][CROPPED][Matricies][90].pkl')
  # output_path = '/home/mark/Dev/BachelorThesis/RegnetCheckpoints/[5][CROPPED][Matricies].pkl'

  # history = dict()
  # for key in history_90.keys():
  #   history[key] = concat(key)

  # with open(output_path, 'wb') as handle:
  #   pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
