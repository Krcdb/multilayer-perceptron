import argparse
import signal
import sys

from utils import load_csv_df

class SplitDataset:
  def __init__(self, input, train_output, test_output):
    self.df = load_csv_df(input)
    self.train_output = train_output
    self.test_output = test_output
  
  def split(self):
    
    self.df = self.df.dropna()
    
    header = ["n", "type"] + [f"feature {i + 1}" for i in range(30)]
    self.df.columns = header
    
    split_index = int(len(self.df) * 0.8)
    df1 = self.df.iloc[:split_index]
    df2 = self.df.iloc[split_index:]

    df1.to_csv(self.train_output, index=False)
    df2.to_csv(self.test_output, index=False)
    
    
def optparse():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input',
      '-i',
      action="store",
      dest="input",
      default="resources/data.csv",
      help="Set the input path file"
  )
  parser.add_argument(
      '--train-output',
      '-to',
      action="store",
      dest="train_output",
      default="resources/dataset_train.csv",
      help="Set the train output path file"
  )
  parser.add_argument(
      '--test-output',
      '-tto',
      action="store",
      dest="test_output",
      default="resources/dataset_test.csv",
      help="Set the test output path file"
  )
  return parser.parse_args()


def signal_handler(sig, frame):
  sys.exit(0)

if __name__ == '__main__':
  signal.signal(signal.SIGINT, signal_handler)
  option = optparse()
  SplitDataset(option.input, option.train_output, option.test_output).split()
