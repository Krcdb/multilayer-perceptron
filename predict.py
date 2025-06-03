
import argparse
import signal
import sys

import numpy as np

from utils import load_csv_df

class Predict:
  def __init__(self, input, model):
    self.mean = 0
    self.std = 0
    self.layer = []
    self.param = {}
    self.activations = {}
    self.model = model
    self.dimension = []
    
    self.df = load_csv_df(input)
    self.X = self.df.iloc[:, 2:].to_numpy().T
    self.y = self.df["type"].map({"M": 1, "B": 0}).to_numpy().reshape(1, -1)
  
  def normalize_x(self):
    self.X = (self.X - self.mean) / self.std

  def load_model(self):
    data = np.load(self.model, allow_pickle=True)
    self.mean = data["mean"]
    self.std = data["std"]
    self.layer = data["layer"].tolist()
    self.param = {key: data[key] for key in data.files if key not in ["mean", "std", "layer"]}
    
    for key, value in data.items():
      print(key, value)
  
  def initialization(self):
    self.normalize_x()
    self.dimension = np.append(self.X.shape[0], self.layer)
    self.dimension = np.append(self.dimension, 1)

    self.C = len(self.dimension)
  
  def sigmoid(self, z):
    return 1 / (1 + np.exp(-z))
  
  def forward_propagation(self):
    self.activations["A0"] = self.X
    
    print(self.C)
    
    for c in range(1, self.C):
      print(f"c : {c}")
      self.activations["A" + str(c)] = self.sigmoid(np.dot(self.param["W" + str(c)], (self.activations["A" + str(c - 1)])) + self.param["b" + str(c)])
    
  def compute_metrics(self):
    A_final = self.activations["A" + str(self.C - 1)]
    predictions = (A_final > 0.5).astype(int)
    accuracy = np.mean(predictions == self.y)

    return accuracy

  def predict(self):
    self.load_model()
    self.initialization()
    self.forward_propagation()
    accuracy = self.compute_metrics()
    
    print(f"Accuracy = {accuracy:.4f}")


def optparse():
  parser = argparse.ArgumentParser(description="Training model.")
  parser.add_argument( '--input', '-i', action="store", dest="input", default="resources/dataset_test.csv", help="Set the input path file")
  parser.add_argument( '--model', '-m', action="store", dest="model", default="resources/trained_model.npz", help="Set the model path file")
  return parser.parse_args()

def signal_handler(sig, frame):
  sys.exit(0)

if __name__ == '__main__':
  signal.signal(signal.SIGINT, signal_handler)
  option = optparse()
  Predict(option.input, option.model).predict()