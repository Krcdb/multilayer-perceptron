import argparse
import signal
import sys
import pandas as pd
import plotly.express as px

from utils import load_csv_df

class CorrelationMatrix:
  def __init__(self, input):
    self.df = load_csv_df(input)

  def plot(self):
    #self.df = self.df.drop(columns=["feature 1", "feature 3", "feature 4", "feature 7", "feature 13", "feature 21", "feature 23", "feature 28"])
    num_df = self.df.select_dtypes(include='number')

    corr = num_df.corr(method='pearson')

    fig = px.imshow(
      corr,
      text_auto=True,
      color_continuous_scale='RdBu_r',
      title='Matrice de corrélation des features',
      aspect='auto'
    )
    fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))
    fig.show()
    
def optparse():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--input',
    '-i',
    action="store",
    dest="input",
    default="resources/dataset_train.csv",
    help="Set the input path file"
  )
  return parser.parse_args()


def signal_handler(sig, frame):
  sys.exit(0)

if __name__ == '__main__':
  signal.signal(signal.SIGINT, signal_handler)
  option = optparse()
  CorrelationMatrix(option.input).plot()

