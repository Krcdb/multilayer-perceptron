import argparse
import signal
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

from utils import load_csv_df

class Histogram:
  def __init__(self, input):
        self.df = load_csv_df(input)
        
  def compute(self):
    features = self.df.drop(columns=["n"]).select_dtypes(include=[np.number]).columns
    labels = self.df["type"]

    num_features = len(features)
    num_cols = 3
    num_rows = math.ceil(num_features / num_cols)

    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=features)

    for idx, feature in enumerate(features):
      row = idx // num_cols + 1
      col = idx % num_cols + 1

      for label_value in ['M', 'B']:
        filtered_data = self.df[self.df['type'] == label_value][feature].dropna()
        fig.add_trace(
          go.Histogram(
            x=filtered_data,
            name=label_value,
            opacity=0.6,
            showlegend=(idx == 0),
          ),
          row=row,
          col=col
        )

    fig.update_layout(
      title_text="Histogram of Each Feature by Type",
      height=300 * num_rows,
      barmode='overlay'
    )

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
  Histogram(option.input).compute()

