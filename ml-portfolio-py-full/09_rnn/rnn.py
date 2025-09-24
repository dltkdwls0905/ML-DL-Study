import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
import csv
from sklearn.preprocessing import MinMaxScaler

class STOCK_RNN(nn.Module):

  def __init__(self, config):

    super(STOCK_RNN, self).__init__()

    self.input_size = config["input_size"]
    self.hidden_size = config["hidden_size"]
    self.output_size = config["output_size"]
    self.num_layers = config["num_layers"]
    self.batch_size = config["batch_size"]

    self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, bidirectional=False, batch_first=True)
    self.linear = nn.Linear(self.hidden_size,self.output_size)

  def forward(self, input_features):

    x, (h_n, c_n) = self.lstm(input_features)

    h_t = x[:,-1,:]

    hypothesis = self.linear(h_t)

    return hypothesis


def load_dataset(fname):

  f = open(fname, 'r', encoding='cp949')


  data = csv.reader(f,delimiter=',')


  next(data)

  data_X = []
  data_Y = []

  for row in data:

    data_X.append([float(i) for i in row[2:]])

    data_Y.append(float(row[1]))


  scaler = MinMaxScaler()
  scaler.fit(data_X)
  data_X = scaler.transform(data_X)

  data_num = len(data_X)
  sequence_len = config["sequence_len"]
  seq_data_X, seq_data_Y = [], []

  for i in range(data_num-sequence_len):
    window_size = i+sequence_len
    seq_data_X.append(data_X[i:window_size])
    seq_data_Y.append([data_Y[window_size-1]])

  (train_X, train_Y) = (np.array(seq_data_X[:]),np.array(seq_data_Y[:]))
  train_X = torch.tensor(train_X, dtype=torch.float)
  train_Y = torch.tensor(train_Y, dtype=torch.float)

  print(train_X.shape) # (73,3,4)
  print(train_Y.shape) # (73,1)

  return (train_X, train_Y)
    

def tensor2list(input_tensor):
    return input_tensor.cpu().detach().numpy().tolist()


def do_test(model, test_dataloader):

  model.eval()

  predicts, golds = [], []

  with torch.no_grad():

    for step, batch in enumerate(test_dataloader):

      batch = tuple(t.cuda() for t in batch)

      input_features, labels = batch
      hypothesis = model(input_features)

      x = tensor2list(hypothesis[:,0])
      y = tensor2list(labels)

      predicts.extend(x)
      golds.extend(y)

    predicts = [round(i,1) for i in predicts]
    golds = [round(i[0],1) for i in golds]

    print("PRED=",predicts)
    print("GOLD=",golds)

def test(config):

  model = STOCK_RNN(config).cuda()

  model.load_state_dict(torch.load(os.path.join(config["output_dir"], config["model_name"])))

  (features, labels) = load_dataset(confing["file_name"])

  test_features = TensorDataset(features, labels)
  test_dataloader = DataLoader(test_features, shuffle=True, batch_size=config["batch_size"])

  do_test(model, test_dataloader)

if(__name__=="__main__"):
    root_dir = "c:/Users/82108/Desktop/스터디 폴더/rnn/stock"
    output_dir = os.path.join(root_dir, "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config = {"mode": "train",
              "model_name":"epoch_{0:d}.pt".format(10),
              "output_dir":output_dir,
              "file_name": "{0:s}/samsung-2020.csv".format(root_dir),
              "sequence_len": 3,
              "input_size": 4,
              "hidden_size": 10,
              "output_size": 1,
              "num_layers": 1,
              "batch_size": 1,
              "learn_rate": 0.1,
              "epoch": 10,
              }

    if(config["mode"] == "train"):
        train(config)
    else:
        test(config)
