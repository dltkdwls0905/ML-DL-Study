import os
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from keras.datasets import mnist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MNIST_CNN(nn.Module):

  def __init__(self, config):

    super(MNIST_CNN, self).__init__()

    # 첫번째 층 설계: Convolutional NN
    # (batch, 28, 28, 1) -> (batch, 28, 28, 32) -> (batch, 14, 14, 32)
    self.conv1 = nn.Sequential()
    self.conv1.add_module("conv1", nn.Conv2d(1,32,kernel_size=(3,3), stride=(1,1), padding=(1,1)))
    self.conv1.add_module("relu1", nn.ReLU())
    self.conv1.add_module("maxpool1", nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)))

    # 두번째 층 설계: Convolutional NN
    # (batch, 14, 14, 32) -> (batch, 14, 14, 64) -> (batch, 7, 7, 64)
    self.conv2 = nn.Sequential(
        nn.Conv2d(32,64,kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2))

    # 세번째 층 설계: Fully-Connected NN
    # (batch, 7, 7, 64) -> (batch, 10)
    self.fnn = nn.Linear(7*7*64,10, bias=True)
    # FNN 가중치 초기화
    nn.init.xavier_uniform_(self.fnn.weight)

  def forward(self, input_features):

    # 첫번째 Convolution
    output = self.conv1(input_features)

    # 두번째 Convolution
    output = self.conv2(output)

    # 텐서를 1차원으로 펼치기: (batch, -1)
    # output.size(0): 배치 차원의 크기, -1: 해당 차원은 파이토치가 알아서 설정
    output = output.view(output.size(0), -1)
    hypothesis = self.fnn(output)

    return hypothesis
  
  # 데이터 읽기 함수
def load_dataset():

  (train_X, train_y), (test_X, test_y) = mnist.load_data()
  print(train_X.shape) # (60000, 28, 28)
  print(train_y.shape) # (60000,10)
  print(test_X.shape) # (10000, 28, 28)
  print(test_y.shape) # (10000,10)

  # 채널 추가
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
      batch = tuple(t.to(device) for t in batch)

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

  model = MNIST_CNN(config).to(device)

  # 저장된 모델 가중치 로드
  model.load_state_dict(torch.load(os.path.join(config["output_dir"], config["model_name"])))

  # 데이터 load
  (_, _), (features, labels) = load_dataset()

  test_features = TensorDataset(features, labels)
  test_dataloader = DataLoader(test_features, shuffle=True, batch_size=config["batch_size"])

  do_test(model, test_dataloader)

  # 모델 학습 함수
def train(config):

  # 모델 생성
  model = MNIST_CNN(config).to(device)

  # 데이터 읽기
  (input_features, labels), (_, _) = load_dataset()

  # TensorDataset/DataLoader를 통해 배치(batch) 단위로 데이터를 나누고 셔플(shuffle)
  train_features = TensorDataset(input_features, labels)
  train_dataloader = DataLoader(train_features, shuffle=True, batch_size=config["batch_size"])

  # 크로스엔트로피 비용 함수
  loss_func = nn.CrossEntropyLoss()
  # 옵티마이저 함수 (역전파 알고리즘을 수행할 함수)
  optimizer = torch.optim.Adam(model.parameters(), lr=config["learn_rate"])

  for epoch in range(config["epoch"]+1):

    # 학습 모드 셋팅
    model.train()

    # epoch 마다 평균 비용을 저장하기 위한 리스트
    costs = []

    for (step, batch) in enumerate(train_dataloader):

      # batch = (input_features[step], labels[step])*batch_size
      # .cuda()를 통해 메모리에 업로드
      batch = tuple(t.to(device) for t in batch)

      # 각 feature 저장
      input_features, labels = batch

      # 역전파 변화도 초기화
      # .backward() 호출 시, 변화도 버퍼에 데이터가 계속 누적한 것을 초기화
      optimizer.zero_grad()

      # H(X) 계산: forward 연산
      hypothesis = model(input_features)
      # 비용 계산
      cost = loss_func(hypothesis, labels)
      # 역전파 수행
      cost.backward()
      optimizer.step()

      # 현재 batch의 스텝 별 loss 저장
      costs.append(cost.data.item())

    # 에폭마다 평균 비용 출력하고 모델을 저장
    print("Average Loss= {0:f}".format(np.mean(costs)))
    torch.save(model.state_dict(), os.path.join(config["output_dir"], "epoch_{0:d}.pt".format(epoch)))
    do_test(model, train_dataloader)

if __name__=="__main__":

    root_dir = "C:/Users/82108/Desktop/스터디 폴더/Convolutional Neural Network"
    output_dir = os.path.join(root_dir, "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config = {
        "mode": "train",
        "model_name": "epoch_{0:d}.pt".format(10),
        "output_dir": output_dir,
        "learn_rate": 0.001,
        "batch_size": 32,
        "epoch": 10,
    }

    if config["mode"] == "train":
        train(config)
    else:
        test(config)