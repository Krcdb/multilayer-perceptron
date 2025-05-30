import argparse
import signal
import sys

class ScatterMatrix:
  def __init__(self, input):
    pass
  
  def display_scatter_matrix(self):
    pass
  

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
