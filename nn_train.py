import argparse
import signal
import sys
import numpy as np
import copy
import plotly.graph_objects as go


from tqdm import tqdm
from utils import load_csv_df

feature_to_drop = ["feature 23"]

class NeuralNetwork:
  def __init__(self, train_input, test_input, output, layer, epochs, display_batch_size, learning_rate, patience, patience_threshold, early_stopping):
    self.train_df = load_csv_df(train_input).drop(columns=feature_to_drop)
    self.test_df = load_csv_df(test_input).drop(columns=feature_to_drop)
    self.output = output
    self.epochs = epochs
    self.layer = layer[:]
    self.display_batch_size = display_batch_size
    self.learning_rate = learning_rate
    self.max_patience = patience
    self.patience_threshold = patience_threshold
    self.early_stopping = early_stopping

    self.param = {}
    self.activations = {} 
    self.gradients = {}
    
    self.X_train = self.train_df.iloc[:, 2:].to_numpy().T
    self.y_train = self.train_df["type"].map({"M": 1, "B": 0}).to_numpy().reshape(1, -1)
    
    self.X_test = self.test_df.iloc[:, 2:].to_numpy().T
    self.y_test = self.test_df["type"].map({"M": 1, "B": 0}).to_numpy().reshape(1, -1)
    

    self.mean_x = 0
    self.std_x = 0
    
    self.dimension = layer
    
    self.C = 0
    
    self.accuracy = []
    self.loss = []
    self.test_accuracy = []
    self.test_loss = []
    
    self.best_model= {}


  def save_model(self):
    print(self.layer)
    to_save = self.best_model.copy()
    to_save["mean"] = self.mean_x
    to_save["std"] = self.std_x
    to_save["layer"] = np.array(self.layer)
    np.savez(self.output, **to_save)

  def normalize_x(self):
    self.mean_x = self.X_train.mean(axis=1, keepdims=True)
    self.std_x = self.X_train.std(axis=1, keepdims=True)

    
    self.X_train = (self.X_train - self.mean_x) / self.std_x
    self.X_test = (self.X_test - self.mean_x) / self.std_x
  
  def sigmoid(self, z):
    return 1 / (1 + np.exp(-z))
  
  def initialization(self):
    self.dimension = np.append(self.X_train.shape[0], self.dimension)
    self.dimension = np.append(self.dimension, 1)

    self.C = len(self.dimension)
    
    for c in range(1, self.C):
      self.param["W" + str(c)] = np.random.randn(self.dimension[c], self.dimension[c - 1])
      self.param["b" + str(c)] = np.random.randn(self.dimension[c], 1)
  
  def forward_propagation(self):
    self.activations["A0"] = self.X_train
    
    for c in range(1, self.C):
      self.activations["A" + str(c)] = self.sigmoid(np.dot(self.param["W" + str(c)], (self.activations["A" + str(c - 1)])) + self.param["b" + str(c)])
    
  def backward_propagation(self):
    dZ = self.activations["A" + str(self.C - 1)] - self.y_train
    
    for c in reversed(range(1, self.C)):
      self.gradients["dW" + str(c)] = np.dot(dZ, self.activations["A" + str(c - 1)].T)
      self.gradients["db" + str(c)] = np.sum(dZ, axis=1, keepdims=True)
      if c > 1:
        dZ = np.dot(self.param["W" + str(c)].T, dZ) * self.activations["A" + str(c - 1)] * ( 1 - self.activations["A" + str(c - 1)])
  
  def update_param(self):
    
    for c in range(1, self.C):
      self.param["W" + str(c)] = self.param["W" + str(c)] - self.learning_rate * self.gradients["dW" + str(c)]
      self.param["b" + str(c)] = self.param["b" + str(c)] - self.learning_rate * self.gradients["db" + str(c)]

  def compute_metrics(self):
    A_final = self.activations["A" + str(self.C - 1)]
    m = self.y_train.shape[1]

    epsilon = 1e-8
    log_loss = - (np.dot(self.y_train, np.log(A_final + epsilon).T) + np.dot(1 - self.y_train, np.log(1 - A_final + epsilon).T)) / m
    log_loss = np.squeeze(log_loss)

    predictions = (A_final > 0.5).astype(int)
    accuracy = np.mean(predictions == self.y_train)

    return log_loss, accuracy

  def evaluate_test_metrics(self):
    A = self.X_test
    for c in range(1, self.C):
        A = self.sigmoid(np.dot(self.param["W" + str(c)], A) + self.param["b" + str(c)])
    
    m = self.y_test.shape[1]
    epsilon = 1e-8
    log_loss = - (np.dot(self.y_test, np.log(A + epsilon).T) + np.dot(1 - self.y_test, np.log(1 - A + epsilon).T)) / m
    log_loss = np.squeeze(log_loss)

    predictions = (A > 0.5).astype(int)
    accuracy = np.mean(predictions == self.y_test)

    return log_loss, accuracy


  def display_loss(self):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=[i * self.display_batch_size for i in range(len(self.loss))], y=self.loss, mode='lines', name='Train Loss'))
    fig.add_trace(go.Scatter(x=[i * self.display_batch_size for i in range(len(self.test_loss))], y=self.test_loss, mode='lines', name='Test Loss'))
    fig.update_layout(title='Binary Cross Entropy Evolution', 
                      xaxis_title='Iteration', 
                      yaxis_title='Log loss')
    fig.show()

  def display_accuracy(self):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=[i * self.display_batch_size for i in range(len(self.accuracy))], y=self.accuracy, mode='lines', name='Train Acc'))
    fig.add_trace(go.Scatter(x=[i * self.display_batch_size for i in range(len(self.test_accuracy))], y=self.test_accuracy, mode='lines', name='Test Acc'))
    fig.update_layout(title='Accuracy Evolution', 
                      xaxis_title='Iteration', 
                      yaxis_title='Accuracy')
    fig.show()
    
  def display_metrics(self):
    self.display_loss()
    self.display_accuracy()
    
  def save_metrics(self):
    log_loss, accuracy = self.compute_metrics()
    test_loss, test_accuracy = self.evaluate_test_metrics()
    self.accuracy.append(accuracy)
    self.loss.append(log_loss)
    self.test_accuracy.append(test_accuracy)
    self.test_loss.append(test_loss)
  
  def batch_gradient_descent(self):
    wait = 0
    best_loss = np.inf
    best_model_iter = 0
    
    for i in tqdm(range(self.epochs)):
      self.forward_propagation() 
      self.backward_propagation()
      self.update_param()
      
      if i % self.display_batch_size == 0 or i == self.epochs - 1:
        self.save_metrics()
        if self.test_loss[-1] < best_loss - self.patience_threshold:
          best_loss = self.test_loss[-1]
          self.best_model = copy.deepcopy(self.param)
          wait = 0
          best_model_iter = i
        else:
          wait += self.display_batch_size
          
        if wait >= self.max_patience:
          print("Early stopping")
          break
      
    
    self.display_metrics()
    self.save_model()
    print(f"Log Loss = {self.loss[len(self.loss) - 1]:.4f} | Accuracy = {self.accuracy[len(self.accuracy) - 1]:.4f} | Best model iter = {best_model_iter}")
  
  def mini_batch_gradient_descent(self):
    pass
  
  def train(self):
    self.normalize_x()
    self.initialization()
    
    self.batch_gradient_descent()
    #self.mini_batch_gradient_descent()
  
def optparse():
  parser = argparse.ArgumentParser(description="Training model.")
  parser.add_argument( '--train_input', '-tri', action="store", dest="train_input", default="resources/dataset_train.csv", help="Set the train input path file")
  parser.add_argument( '--test_input', '-tei', action="store", dest="test_input", default="resources/dataset_test.csv", help="Set the test input path file")
  parser.add_argument( '--output', '-o', action="store", dest="output", default="resources/trained_model.npz", help="Set the output path file")
  parser.add_argument('--layer', '-l', action="store", dest="layer", type=int, nargs='+', default=[10, 10], help='Sizes of each hidden layer')
  parser.add_argument('--epochs', '-e', action="store", dest="epochs", type=int, default=10000, help='Number of training epochs')
  parser.add_argument('--display_batch_size', '-bs', action="store", dest="display_batch_size", type=int, default=50, help='Display batch size')
  parser.add_argument('--learning_rate', '-lr', action="store", dest="learning_rate", type=float, default=0.001, help='Learning rate')
  parser.add_argument('--patience', '-p', action="store", dest="patience", type=int, default=500, help='Patience')
  parser.add_argument('--patience_threshold', '-pt', action="store", dest="patience_threshold", type=int, default=0.00001, help='Patience threshold')
  parser.add_argument('--early_stopping', '-es', action="store_true", dest="early_stopping", default=False, help="Set the early stopping")
  
  
  return parser.parse_args()

def signal_handler(sig, frame):
  sys.exit(0)

if __name__ == '__main__':
  signal.signal(signal.SIGINT, signal_handler)
  option = optparse()
  NeuralNetwork(option.train_input,
                option.test_input,
                option.output,
                option.layer,
                option.epochs, 
                option.display_batch_size,
                option.learning_rate,
                option.patience,
                option.patience_threshold,
                option.early_stopping).train()