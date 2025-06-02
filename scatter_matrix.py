import argparse
import signal
import sys
import plotly.express as px
from utils import load_csv_df

class ScatterMatrix:
  def __init__(self, input):
    self.df = load_csv_df(input)
  
  def display_scatter_matrix(self):
    desc = self.df.describe(include='all')
    cols = desc.columns
    for i in range(0, len(cols), 5):
      print(desc[cols[i:i+5]])
      print('\n' + '-' * 80 + '\n')
      
    feature = self.df.iloc[:, 2:]
    
    if "type" in self.df.columns:
      color = self.df["type"]
    else:
      color = None
      
    fig = px.scatter_matrix(
        feature,
        dimensions=feature.columns,
        color=color,
        title="Pair Plot of Numeric Features",
        height=1000,
        width=1000,
        opacity=0.3
    )

    fig.update_traces(diagonal_visible=False)
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
  ScatterMatrix(option.input).display_scatter_matrix()
