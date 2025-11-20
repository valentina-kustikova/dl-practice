import kagglehub
import argparse

import numpy as np # linear algebra
import struct
from array import array
from os.path  import join

import pandas as pd
import os
from sklearn.utils import shuffle

path = kagglehub.dataset_download("hojjatk/mnist-dataset")

def cli_argument_parser():
    """Парсер командной строки для получения параметров"""
    parser = argparse.ArgumentParser(description="backprop params")

    parser.add_argument('-e', '--epochs',
                        help='num of epochs',
                        type=int,
                        default=20,
                        dest='epochs')

    parser.add_argument('-lr', '--lerning-rate',
                        help='lerning rate',
                        type=float,
                        default=0.1,
                        dest='lerning_rate')

    parser.add_argument('-ln', '--layers-num',
                        help='layers-num',
                        type=int,
                        default=300,
                        dest='layers_num')

    parser.add_argument('-bs', '--batch-size',
                        help='batch size',
                        type=int,
                        default=16,
                        dest='batch_size')

    return parser.parse_args()


#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data =  array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)      
    

def ReLU(x):
    return np.maximum(0, x)

def dReLU(x):
     return (x > 0).astype(x.dtype)

def SoftMax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_shift = x - np.max(x, axis=axis, keepdims=True)
    exps = np.exp(x_shift)
    return exps / np.sum(exps, axis=axis, keepdims=True)

def dSoftMax(s: np.ndarray, upstream: np.ndarray, axis: int = -1) -> np.ndarray:
    proj = np.sum(upstream * s, axis=axis, keepdims=True)
    return s * (upstream - proj)

class Network:
    def __init__(self, w, h, s, classes):
        self._input  = np.random.randn(w * h, s) * 0.01
        self._b1     = np.zeros(s)
        self._output = np.random.randn(s, classes) * 0.01
        self._b2     = np.zeros(classes)
        self._size   = s
        self._classes = classes

    def forward(self, X):
        z1 = X @ self._input + self._b1
        a1 = ReLU(z1)
        z2 = a1 @ self._output + self._b2
        yhat = SoftMax(z2)
        return yhat

    def loss(self, prop, y_true):
        return (prop - y_true) ** 2

    def backward(self, X, y, learning_rate=0.1):

        z1 = X @ self._input + self._b1
        a1 = ReLU(z1)
        z2 = a1 @ self._output + self._b2
        yhat = SoftMax(z2)

        B = X.shape[0]
        
        loss = -np.sum(y * np.log(yhat + 1e-8)) / B
        
        dz2 = (yhat - y) / B

        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        da1 = dz2 @ self._output.T
        dz1 = da1 * dReLU(z1)

        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)

        self._input -= learning_rate * dW1
        self._b1 -= learning_rate * db1
        self._output -= learning_rate * dW2
        self._b2 -= learning_rate * db2
        return yhat, loss
    


def accuracy(y_pred, y_true_labels):
    return np.mean(np.argmax(y_pred, axis=1) == y_true_labels)

def one_hot(y, num_classes):
    return np.eye(num_classes, dtype=np.float32)[y]

def __main__():
  args = cli_argument_parser()

  (num_classes, w, h) = (10, 28, 28)
  test_df = os.path.join(path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
  test_lb = os.path.join(path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
  train_lb = os.path.join(path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
  train_df = os.path.join(path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')

  mnist_dataloader = MnistDataloader(train_df, train_lb, test_df, test_lb)
  (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

  x_train = np.asarray(x_train)
  x_test  = np.asarray(x_test)
  y_train = np.asarray(y_train)
  y_test  = np.asarray(y_test)

  x_train = (x_train.astype(np.float32) / 255.0).reshape(-1, w * h)
  x_test  = (x_test.astype(np.float32)  / 255.0).reshape(-1, w * h)
  y_train_oh = one_hot(y_train, num_classes)
  y_test_oh  = one_hot(y_test,  num_classes)


  np.random.seed(42)
  model = Network(w=w, h=h, s=args.layers_num, classes=num_classes)

  print("Data verification:")
  print(f"x_train type: {type(x_train)}")
  print(f"x_train shape: {(x_train.shape)}")
  print(f"y_train_oh type: {type(y_train_oh)}")
  print(f"y_train_oh shape: {y_train_oh.shape}")
  epochs = args.epochs
  batch_size = int(args.batch_size)
  lr = args.lerning_rate

  n = int(x_train.shape[0])

  for epoch in range(1, epochs + 1):
      idx = np.random.permutation(len(x_train))
      x_train_shuf = x_train[idx]
      y_train_shuf = y_train_oh[idx]
      epoch_loss = 0.0
      for i in range(0, n, batch_size):
          xb = x_train_shuf[i:i+batch_size]
          yb = y_train_shuf[i:i+batch_size]

          yhat, loss_b = model.backward(xb, yb, learning_rate=lr)
          epoch_loss += loss_b * (batch_size)

      epoch_loss /= n

      yhat_train = model.forward(x_train)
      train_acc = accuracy(yhat_train, y_train)

      yhat_test = model.forward(x_test)
      test_acc = accuracy(yhat_test, y_test)

      print(f"Epoch {epoch:02d}/{epochs} | loss={epoch_loss:.4f} | "
            f"train_acc≈{train_acc:.4f} | test_acc={test_acc:.4f}")

  preds = np.argmax(model.forward(x_test[:10]), axis=1)
  print("Preds:", preds)
  print("True :", y_test[:10])

if __name__ == "__main__":
  __main__()