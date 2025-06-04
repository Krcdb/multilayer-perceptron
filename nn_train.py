import argparse
import signal
import sys
import numpy as np
import copy
import plotly.graph_objects as go


from tqdm import tqdm
from train_utils import backward_propagation, forward_propagation, save_metrics, update_param
from utils import load_csv_df

BATCH = "batch"
MINIBATCH = "minibatch"
ADAM = "adam"
RMSPROP = "rmsprop"

feature_to_drop = ["feature 23"]

class NeuralNetwork:
  def __init__(self, train_input, test_input, output, layer, epochs, batch_size, display_batch_size, learning_rate, patience, patience_threshold, early_stopping, batch_gradient, mini_batch_gradient, adam, rmsprop, all_mode):
    self.train_df = load_csv_df(train_input).drop(columns=feature_to_drop)
    self.test_df = load_csv_df(test_input).drop(columns=feature_to_drop)
    self.output = output
    self.epochs = epochs
    self.batch_size = batch_size
    self.layer = layer[:]
    self.display_batch_size = display_batch_size
    self.learning_rate = learning_rate
    self.max_patience = patience
    self.patience_threshold = patience_threshold
    self.early_stopping = early_stopping

    self.initial_param = {}

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
    
    self.train_accuracy_history = {}
    self.train_loss_history = {}
    self.test_accuracy_history = {}
    self.test_loss_history = {}
    
    self.best_model= {}
    self.best_model_iter = {}
    
    self.b_batch_gradient = batch_gradient
    self.b_mini_batch_gradient = mini_batch_gradient
    self.b_adam = adam
    self.b_rmsprop = rmsprop
    
    self.b_all_mode = all_mode
    
    if self.b_all_mode:
      self.b_batch_gradient = True
      self.b_mini_batch_gradient = True
      self.b_adam = True
      self.b_rmsprop = True
    if not self.b_batch_gradient and not self.b_mini_batch_gradient and not self.b_adam and not self.b_rmsprop:
      self.b_batch_gradient = True


  def save_model(self, mode):
    print(f"Saving best model for {mode}")
    to_save = self.best_model[mode].copy()
    to_save["mean"] = self.mean_x
    to_save["std"] = self.std_x
    to_save["layer"] = np.array(self.layer)
    np.savez(self.output + "_" + mode + ".npz", **to_save)

  def normalize_x(self):
    self.mean_x = self.X_train.mean(axis=1, keepdims=True)
    self.std_x = self.X_train.std(axis=1, keepdims=True)

    
    self.X_train = (self.X_train - self.mean_x) / self.std_x
    self.X_test = (self.X_test - self.mean_x) / self.std_x
  
  
  def initialization(self):
    self.dimension = np.append(self.X_train.shape[0], self.dimension)
    self.dimension = np.append(self.dimension, 1)
    self.C = len(self.dimension)
    
    for c in range(1, self.C):
      self.initial_param["W" + str(c)] = np.random.randn(self.dimension[c], self.dimension[c - 1])
      self.initial_param["b" + str(c)] = np.random.randn(self.dimension[c], 1)

  def display_loss(self):
    fig = go.Figure()
    
    for key in self.train_loss_history:
      x_train = [i * self.display_batch_size for i in range(len(self.train_loss_history[key]))]
      x_test = [i * self.display_batch_size for i in range(len(self.test_loss_history[key]))]
      
      fig.add_trace(go.Scatter(x=x_train, y=self.train_loss_history[key], mode='lines', name=f"Train Loss {key}"))
      fig.add_trace(go.Scatter(x=x_test, y=self.test_loss_history[key], mode='lines', name=f"Test Loss {key}"))
    
    fig.update_layout(title='Binary Cross Entropy Evolution', 
                      xaxis_title='Iteration', 
                      yaxis_title='Log loss')
    fig.show()

  def display_accuracy(self):
    fig = go.Figure()
    
    for key in self.train_accuracy_history:
      x_train = [i * self.display_batch_size for i in range(len(self.train_accuracy_history[key]))]
      x_test = [i * self.display_batch_size for i in range(len(self.test_accuracy_history[key]))]
      
      fig.add_trace(go.Scatter(x=x_train, y=self.train_accuracy_history[key], mode='lines', name=f"Train Acc {key}"))
      fig.add_trace(go.Scatter(x=x_test, y=self.test_accuracy_history[key], mode='lines', name=f"Test Acc {key}"))
    
    fig.update_layout(title='Accuracy Evolution', 
                      xaxis_title='Iteration', 
                      yaxis_title='Accuracy')
    fig.show()
    
  def display_metrics(self):
    self.display_loss()
    self.display_accuracy()
  
  
  def adam(self):
    self.param = copy.deepcopy(self.initial_param)
    wait = 0
    best_loss = np.inf
    
    mode = ADAM
    self.train_accuracy_history[mode] = []
    self.train_loss_history[mode] = []
    self.test_accuracy_history[mode] = []
    self.test_loss_history[mode] = []
    self.best_model[mode] = {}
    self.best_model_iter[mode] = 0
    
    m = self.X_train.shape[1]
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    m_t = {k: np.zeros_like(v) for k, v in self.param.items()}
    v_t = {k: np.zeros_like(v) for k, v in self.param.items()}
    
    print("starting Adam")
    
    for i in tqdm(range(1, self.epochs + 1)):
      permutation = np.random.permutation(m)
      X_shuffled = self.X_train[:, permutation]
      y_shuffled = self.y_train[:, permutation]
      
      for j in range(0, m, self.batch_size):
        X_batch = X_shuffled[:, j:j + self.batch_size]
        y_batch = y_shuffled[:, j:j + self.batch_size]
      
        forward_propagation(self.activations, X_batch, self.param, self.C) 
        backward_propagation(self.activations, self.gradients, self.param, y_batch, self.C)
        update_param(self.param, self.gradients, self.C, self.learning_rate)
        
        for key in self.param:
          m_t[key] = beta1 * m_t[key] + (1 - beta1) * self.gradients['d' + key]
          v_t[key] = beta2 * v_t[key] + (1 - beta2) * (self.gradients['d' + key] ** 2)

          m_hat = m_t[key] / (1 - beta1 ** i)
          v_hat = v_t[key] / (1 - beta2 ** i)

          self.param[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        
      if i % self.display_batch_size == 0 or i == self.epochs - 1:
        forward_propagation(self.activations, self.X_train, self.param, self.C)
        save_metrics(self.train_accuracy_history, self.train_loss_history, self.test_accuracy_history, self.test_loss_history, mode, self.X_test, self.y_test, self.y_train, self.param, self.C, self.activations)
        if self.test_loss_history[mode][-1] < best_loss - self.patience_threshold:
          best_loss = self.test_loss_history[mode][-1]
          self.best_model[mode] = copy.deepcopy(self.param)
          wait = 0
          self.best_model_iter[mode] = i
        else:
          wait += self.display_batch_size
          
        if self.early_stopping and wait >= self.max_patience:
          break
      
    self.save_model(mode)
    print(f"|{mode}| Best model epoch = {self.best_model_iter[mode]}")
  
  
  def rmsprop(self):
    self.param = copy.deepcopy(self.initial_param)
    wait = 0
    best_loss = np.inf
    
    mode = RMSPROP
    self.train_accuracy_history[mode] = []
    self.train_loss_history[mode] = []
    self.test_accuracy_history[mode] = []
    self.test_loss_history[mode] = []
    self.best_model[mode] = {}
    self.best_model_iter[mode] = 0
    
    m = self.X_train.shape[1]
    
    decay_rate = 0.9
    epsilon = 1e-8
    cache = {k: np.zeros_like(v) for k, v in self.param.items()}
    
    print("starting rmsprop")
    
    for i in tqdm(range(self.epochs)):
      permutation = np.random.permutation(m)
      X_shuffled = self.X_train[:, permutation]
      y_shuffled = self.y_train[:, permutation]
      
      for j in range(0, m, self.batch_size):
        X_batch = X_shuffled[:, j:j + self.batch_size]
        y_batch = y_shuffled[:, j:j + self.batch_size]
      
        forward_propagation(self.activations, X_batch, self.param, self.C) 
        backward_propagation(self.activations, self.gradients, self.param, y_batch, self.C)
        update_param(self.param, self.gradients, self.C, self.learning_rate)
        
        for key in self.param:
          cache[key] = decay_rate * cache[key] + (1 - decay_rate) * (self.gradients['d' + key] ** 2)
          self.param[key] -= self.learning_rate * self.gradients['d' + key] / (np.sqrt(cache[key]) + epsilon)

        
      if i % self.display_batch_size == 0 or i == self.epochs - 1:
        forward_propagation(self.activations, self.X_train, self.param, self.C)
        save_metrics(self.train_accuracy_history, self.train_loss_history, self.test_accuracy_history, self.test_loss_history, mode, self.X_test, self.y_test, self.y_train, self.param, self.C, self.activations)
        if self.test_loss_history[mode][-1] < best_loss - self.patience_threshold:
          best_loss = self.test_loss_history[mode][-1]
          self.best_model[mode] = copy.deepcopy(self.param)
          wait = 0
          self.best_model_iter[mode] = i
        else:
          wait += self.display_batch_size
          
        if self.early_stopping and wait >= self.max_patience:
          break
      
    self.save_model(mode)
    print(f"|{mode}| Best model epoch = {self.best_model_iter[mode]}")
  
  def mini_batch_gradient_descent(self):
    self.param = copy.deepcopy(self.initial_param)
    wait = 0
    best_loss = np.inf
    
    mode = MINIBATCH
    self.train_accuracy_history[mode] = []
    self.train_loss_history[mode] = []
    self.test_accuracy_history[mode] = []
    self.test_loss_history[mode] = []
    self.best_model[mode] = {}
    self.best_model_iter[mode] = 0
    
    m = self.X_train.shape[1]
    
    print("starting mini batch gradient")
    
    for i in tqdm(range(self.epochs)):
      permutation = np.random.permutation(m)
      X_shuffled = self.X_train[:, permutation]
      y_shuffled = self.y_train[:, permutation]
      
      for j in range(0, m, self.batch_size):
        X_batch = X_shuffled[:, j:j + self.batch_size]
        y_batch = y_shuffled[:, j:j + self.batch_size]
      
        forward_propagation(self.activations, X_batch, self.param, self.C) 
        backward_propagation(self.activations, self.gradients, self.param, y_batch, self.C)
        update_param(self.param, self.gradients, self.C, self.learning_rate)
        
      if i % self.display_batch_size == 0 or i == self.epochs - 1:
        forward_propagation(self.activations, self.X_train, self.param, self.C)
        save_metrics(self.train_accuracy_history, self.train_loss_history, self.test_accuracy_history, self.test_loss_history, mode, self.X_test, self.y_test, self.y_train, self.param, self.C, self.activations)
        if self.test_loss_history[mode][-1] < best_loss - self.patience_threshold:
          best_loss = self.test_loss_history[mode][-1]
          self.best_model[mode] = copy.deepcopy(self.param)
          wait = 0
          self.best_model_iter[mode] = i
        else:
          wait += self.display_batch_size
          
        if self.early_stopping and wait >= self.max_patience:
          break
      
    self.save_model(mode)
    print(f"|{mode}| Best model epoch = {self.best_model_iter[mode]}")
  
  def batch_gradient_descent(self):
    self.param = copy.deepcopy(self.initial_param)
    wait = 0
    best_loss = np.inf
    
    mode = BATCH
    self.train_accuracy_history[mode] = []
    self.train_loss_history[mode] = []
    self.test_accuracy_history[mode] = []
    self.test_loss_history[mode] = []
    self.best_model[mode] = {}
    self.best_model_iter[mode] = 0
    
    print("starting batch gradient")
    
    for i in tqdm(range(self.epochs)):
      forward_propagation(self.activations, self.X_train, self.param, self.C) 
      backward_propagation(self.activations, self.gradients, self.param, self.y_train, self.C)
      update_param(self.param, self.gradients, self.C, self.learning_rate)
      
      if i % self.display_batch_size == 0 or i == self.epochs - 1:
        save_metrics(self.train_accuracy_history, self.train_loss_history, self.test_accuracy_history, self.test_loss_history, mode, self.X_test, self.y_test, self.y_train, self.param, self.C, self.activations)
        if self.test_loss_history[mode][-1] < best_loss - self.patience_threshold:
          best_loss = self.test_loss_history[mode][-1]
          self.best_model[mode] = copy.deepcopy(self.param)
          wait = 0
          self.best_model_iter[mode] = i
        else:
          wait += self.display_batch_size
          
        if self.early_stopping and wait >= self.max_patience:
          print("Early stopping")
          break
      
    
    self.save_model(mode)
    print(f"|{mode}| Best model epoch = {self.best_model_iter[mode]}")
  
  def train(self):
    self.normalize_x()
    self.initialization()
    
    if self.b_batch_gradient:
      self.batch_gradient_descent()
    if self.b_mini_batch_gradient:
      self.mini_batch_gradient_descent()
    if self.b_adam:
      self.adam()
    if self.b_rmsprop:
      self.rmsprop()

    self.display_metrics()
  
def optparse():
  parser = argparse.ArgumentParser(description="Training model.")
  parser.add_argument( '--train-input', '-tri', action="store", dest="train_input", default="resources/dataset_train.csv", help="Set the train input path file")
  parser.add_argument( '--test-input', '-tei', action="store", dest="test_input", default="resources/dataset_test.csv", help="Set the test input path file")
  parser.add_argument( '--output', '-o', action="store", dest="output", default="resources/trained_model", help="Set the output path file")
  parser.add_argument('--layer', '-l', action="store", dest="layer", type=int, nargs='+', default=[10, 10], help='Sizes of each hidden layer')
  parser.add_argument('--epochs', '-e', action="store", dest="epochs", type=int, default=10000, help='Number of training epochs')
  parser.add_argument('--batch-size', '-bs', action="store", dest="batch_size", type=int, default=64, help='Batch size')
  parser.add_argument('--display-batch-size', '-dbs', action="store", dest="display_batch_size", type=int, default=50, help='Display batch size')
  parser.add_argument('--learning-rate', '-lr', action="store", dest="learning_rate", type=float, default=0.001, help='Learning rate')
  parser.add_argument('--patience', '-p', action="store", dest="patience", type=int, default=500, help='Patience')
  parser.add_argument('--patience-threshold', '-pt', action="store", dest="patience_threshold", type=int, default=0.00001, help='Patience threshold')
  parser.add_argument('--early-stopping', '-es', action="store_true", dest="early_stopping", default=False, help="Set the early stopping")
  parser.add_argument('--batch-gradient', '-bg', action="store_true", dest="batch_gradient", default=False, help="Run the batch gradient mode")
  parser.add_argument('--mini-batch-gradient', '-mbg', action="store_true", dest="mini_batch_gradient", default=False, help="Run the minibatch gradient mode")
  parser.add_argument('--adam', '-ad', action="store_true", dest="adam", default=False, help="Run the adam mode")
  parser.add_argument('--rmsprop', '-rms', action="store_true", dest="rmsprop", default=False, help="Run the rmsprop mode")
  parser.add_argument('--all_mode', '-all', action="store_true", dest="all_mode", default=False, help="Run all mods")

  return parser.parse_args()

def signal_handler(sig, frame):
  sys.exit(0)

if __name__ == '__main__':
  signal.signal(signal.SIGINT, signal_handler)
  option = optparse()
  
  print("Options sélectionnées :")
  for key, value in vars(option).items():
    print(f"{key}: {value}")
  
  print("")
  
  NeuralNetwork(option.train_input,
                option.test_input,
                option.output,
                option.layer,
                option.epochs,
                option.batch_size,
                option.display_batch_size,
                option.learning_rate,
                option.patience,
                option.patience_threshold,
                option.early_stopping,
                option.batch_gradient,
                option.mini_batch_gradient,
                option.adam,
                option.rmsprop,
                option.all_mode).train()