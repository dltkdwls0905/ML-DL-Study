import os
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from keras.datasets import mnist

class MNIST_CNN(nn.Module):

  def __init__(self, config):

    super(MNIST_CNN, self).__init__()

    
    self.conv1 = nn.Sequential()
    self.conv1.add_module("conv1", nn.Conv2d(1,32,kernel_size=(3,3), stride=(1,1), padding=(1,1)))
    self.conv1.add_module("relu1", nn.ReLU())
    self.conv1.add_module("maxpool1", nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)))

   
    
    self.conv2 = nn.Sequential(
        nn.Conv2d(32,64,kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2))

   
    self.fnn = nn.Linear(7*7*64,10, bias=True)
    
    nn.init.xavier_uniform_(self.fnn.weight)

  def forward(self, input_features):


    output = self.conv1(input_features)


    output = self.conv2(output)

  
    output = output.view(output.size(0), -1)
    hypothesis = self.fnn(output)

    return hypothesis

def load_dataset():

  (train_X, train_y), (test_X, test_y) = mnist.load_data()
  print(train_X.shape) # (60000, 28, 28)
  print(train_y.shape) # (60000,10)
  print(test_X.shape) # (10000, 28, 28)
  print(test_y.shape) # (10000,10)


  train_X = train_X.reshape(-1, 1, 28, 28)
  test_X  = test_X.reshape(-1, 1, 28, 28)
  print(train_X.shape)
  print(test_X.shape)

  train_X = torch.tensor(train_X, dtype=torch.float)
  train_y = torch.tensor(train_y, dtype=torch.long)
  test_X = torch.tensor(test_X, dtype=torch.float)
  test_y = torch.tensor(test_y, dtype=torch.long)

  return (train_X, train_y), (test_X, test_y)

# 모델 평가 결과 계산을 위해 텐서를 리스트로 변환하는 함수
def tensor2list(input_tensor):
    return input_tensor.cpu().detach().numpy().tolist()

# 평가 수행 함수
def do_test(model, test_dataloader):

  # 평가 모드 셋팅
  model.eval()

  # Batch 별로 예측값과 정답을 저장할 리스트 초기화
  predicts, golds = [], []

  with torch.no_grad():

    for step, batch in enumerate(test_dataloader):

      # .cuda()를 통해 메모리에 업로드
      batch = tuple(t.cuda() for t in batch)

      input_features, labels = batch
      hypothesis = model(input_features)

      # ont-hot 표현으로 변경
      logits = torch.argmax(hypothesis,-1)

      x = tensor2list(logits)
      y = tensor2list(labels)

      # 예측값과 정답을 리스트에 추가
      predicts.extend(x)
      golds.extend(y)

    print("PRED=",predicts)
    print("GOLD=",golds)
    print("Accuracy= {0:f}\n".format(accuracy_score(golds, predicts)))

# 모델 평가 함수
def test(config):

  model = MNIST_CNN(config).cuda()

  # 저장된 모델 가중치 로드
  model.load_state_dict(torch.load(os.path.join(config["output_dir"], config["model_name"])))

  # 데이터 load
  (_, _), (features, labels) = load_dataset()

  test_features = TensorDataset(features, labels)
  test_dataloader = DataLoader(test_features, shuffle=True, batch_size=config["batch_size"])

  do_test(model, test_dataloader)


    
