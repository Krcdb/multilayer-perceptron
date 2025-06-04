import numpy as np


def sigmoid(z):
  return 1 / (1 + np.exp(-z))

def evaluate_test_metrics(X_test, y_test, param, C):
  A = X_test
  for c in range(1, C):
    A = sigmoid(np.dot(param["W" + str(c)], A) + param["b" + str(c)])
    
  m = y_test.shape[1]
  epsilon = 1e-8
  log_loss = - (np.dot(y_test, np.log(A + epsilon).T) + np.dot(1 - y_test, np.log(1 - A + epsilon).T)) / m
  log_loss = np.squeeze(log_loss)

  predictions = (A > 0.5).astype(int)
  accuracy = np.mean(predictions == y_test)

  return log_loss, accuracy

def compute_metrics(activations, y_train, C):
  A_final = activations["A" + str(C - 1)]
  
  m = y_train.shape[1]
  epsilon = 1e-8
  
  log_loss = - (np.dot(y_train, np.log(A_final + epsilon).T) + np.dot(1 - y_train, np.log(1 - A_final + epsilon).T)) / m
  log_loss = np.squeeze(log_loss)
  
  predictions = (A_final > 0.5).astype(int)
  accuracy = np.mean(predictions == y_train)
  return log_loss, accuracy

def save_metrics(
      train_accuracy_history,
      train_loss_history,
      test_accuracy_history,
      test_loss_history,
      mode,
      X_test,
      y_test,
      y_train,
      param,
      C,
      activations
    ):
  log_loss, accuracy = compute_metrics(activations, y_train, C)
  test_loss, test_accuracy = evaluate_test_metrics(X_test, y_test, param, C)
  train_accuracy_history[mode].append(accuracy)
  train_loss_history[mode].append(log_loss)
  test_accuracy_history[mode].append(test_accuracy)
  test_loss_history[mode].append(test_loss)
  
def forward_propagation(activations, X_train, param, C):
  activations["A0"] = X_train
    
  for c in range(1, C):
    activations["A" + str(c)] = sigmoid(np.dot(param["W" + str(c)], (activations["A" + str(c - 1)])) + param["b" + str(c)])

def backward_propagation(activations, gradients, param, y_train, C):
  dZ = activations["A" + str(C - 1)] - y_train
    
  for c in reversed(range(1, C)):
    gradients["dW" + str(c)] = np.dot(dZ, activations["A" + str(c - 1)].T)
    gradients["db" + str(c)] = np.sum(dZ, axis=1, keepdims=True)
    if c > 1:
      dZ = np.dot(param["W" + str(c)].T, dZ) * activations["A" + str(c - 1)] * ( 1 - activations["A" + str(c - 1)])
  
def update_param(param, gradients, C, learning_rate):
    
  for c in range(1, C):
    param["W" + str(c)] = param["W" + str(c)] - learning_rate * gradients["dW" + str(c)]
    param["b" + str(c)] = param["b" + str(c)] - learning_rate * gradients["db" + str(c)]
