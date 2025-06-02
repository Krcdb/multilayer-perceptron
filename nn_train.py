import argparse
import signal
import sys
import numpy as np
import pandas as pd

from utils import load_csv_df

class NeuralNetwork:
  def __init__(self, input, layer, epochs, batch_size, learning_rate):
    self.df = load_csv_df(input)
    self.layer = layer
    self.epochs = epochs
    self.batch_size = batch_size
    self.learning_rate = learning_rate

    self.param = {}
    self.activations = {}
    self.gradients = {}
    
    self.X = self.df.iloc[:, 2:].to_numpy().T
    self.y = self.df["type"].map({"M": 1, "B": 0}).to_numpy().reshape(1, -1)

    self.mean = 0
    self.std = 0
    
    self.dimension = layer
    
    self.C = 0
  
  def normalize_x(self):
    self.mean = self.X.mean(axis=0)
    self.std = self.X.std(axis=0)
    
    self.X = (self.X - self.mean) / self.std
  
  def sigmoid(self, z):
    return 1 / (1 + np.exp(-z))
  
  def initialization(self):
    self.dimension = np.append(self.X.shape[0], self.dimension)
    self.dimension = np.append(self.dimension, 1)

    self.C = len(self.dimension)
    
    for c in range(1, self.C):
      self.param["W" + str(c)] = np.random.randn(self.dimension[c], self.dimension[c - 1])
      self.param["b" + str(c)] = np.random.randn(self.dimension[c], 1)

    print("Initialization done, NN shape:")
    for key, value in self.param.items():
      print(key, value.shape)
  
  def forward_propagation(self):
    self.activations["A0"] = self.X
    
    for c in range(1, self.C):
      self.activations["A" + str(c)] = self.sigmoid(np.dot(self.param["W" + str(c)], (self.activations["A" + str(c - 1)])) + self.param["b" + str(c)])
    
    print("forward prop shape:")
    for key, value in self.activations.items():
      print(key, value.shape)
    
  def backward_propagation(self):
    print(self.activations["A" + str(self.C - 1)])
    dZ = self.activations["A" + str(self.C - 1)] - self.y
    
    for c in reversed(range(1, self.C)):
      self.gradients["dW" + str(c)] = np.dot(dZ, self.activations["A" + str(c - 1)].T)
      self.gradients["db" + str(c)] = np.sum(dZ, axis=1, keepdims=True)
      if c > 1:
        dZ = np.dot(self.param["W" + str(c)].T, dZ) * self.activations["A" + str(c - 1)] * ( 1 - self.activations["A" + str(c - 1)])
        
    print("backward prop shape:")
    for key, value in self.gradients.items():
      print(key, value.shape)
      
  def update_param(self):
    
    for c in range(1, self.C):
      self.param["W" + str(c)] = self.param["W" + str(c)] - self.learning_rate * self.gradients["dW" + str(c)]
      self.param["b" + str(c)] = self.param["b" + str(c)] - self.learning_rate * self.gradients["db" + str(c)]
      
  
  def train(self):
    self.normalize_x()
    self.initialization()
    self.forward_propagation()
    self.backward_propagation()

def optparse():
  parser = argparse.ArgumentParser(description="Training model.")
  parser.add_argument( '--input', '-i', action="store", dest="input", default="resources/dataset_train.csv", help="Set the input path file")
  parser.add_argument('--layer', '-l', action="store", dest="layer", type=int, nargs='+', default=[10, 10], help='Sizes of each hidden layer')
  parser.add_argument('--epochs', '-e', action="store", dest="epochs", type=int, default=100, help='Number of training epochs')
  parser.add_argument('--batch_size', '-bs', action="store", dest="batch_size", type=int, default=32, help='Batch size')
  parser.add_argument('--learning_rate', '-lr', action="store", dest="learning_rate", type=float, default=0.001, help='Learning rate')
  
  return parser.parse_args()


def signal_handler(sig, frame):
  sys.exit(0)

if __name__ == '__main__':
  signal.signal(signal.SIGINT, signal_handler)
  option = optparse()
  NeuralNetwork(option.input, option.layer, option.epochs, option.batch_size, option.learning_rate).train()