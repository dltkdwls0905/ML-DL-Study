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

    # LSTM 설계
    self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, bidirectional=False, batch_first=True)
    # 출력층 설계
    self.linear = nn.Linear(self.hidden_size,self.output_size)

  def forward(self, input_features):

    # LSTM 리턴 = output (배치, 시퀀스, 은닉 상태), (hidden_state, cell_state)
    x, (h_n, c_n) = self.lstm(input_features)

    # output에서 마지막 시퀀스의 (배치, 은닉 상태) 정보를 가져옴
    h_t = x[:,-1,:]

    # 출력층: (배치, 출력)
    hypothesis = self.linear(h_t)

    return hypothesis

# 데이터 읽기 함수
def load_dataset(fname):

  f = open(fname, 'r', encoding='cp949')

  # CSV 파일 읽기
  data = csv.reader(f,delimiter=',')

  # 헤더 건너뛰기
  next(data)

  data_X = []
  data_Y = []

  for row in data:
    # 오픈, 고가, 저가, 거래량 -> 숫자 변환
    data_X.append([float(i) for i in row[2:]])
    # 종가 -> 숫자 변환
    data_Y.append(float(row[1]))

  # MinMax 정규화 (예측하려는 종가 제외)
  scaler = MinMaxScaler()
  scaler.fit(data_X)
  data_X = scaler.transform(data_X)

  data_num = len(data_X)
  sequence_len = config["sequence_len"]
  seq_data_X, seq_data_Y = [], []

  # 윈도우 크기만큼 슬라이딩 하면서 데이터 생성
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

      x = tensor2list(hypothesis[:,0])
      y = tensor2list(labels)

      # 예측값과 정답을 리스트에 추가
      predicts.extend(x)
      golds.extend(y)

    # 소숫점 이하 1자리로 변환
    predicts = [round(i,1) for i in predicts]
    golds = [round(i[0],1) for i in golds]

    print("PRED=",predicts)
    print("GOLD=",golds)

# 모델 평가 함수
def test(config):

  model = STOCK_RNN(config).cuda()

  # 저장된 모델 가중치 로드
  model.load_state_dict(torch.load(os.path.join(config["output_dir"], config["model_name"])))

  # 데이터 load
  (features, labels) = load_dataset(confing["file_name"])

  test_features = TensorDataset(features, labels)
  test_dataloader = DataLoader(test_features, shuffle=True, batch_size=config["batch_size"])

  do_test(model, test_dataloader)

if(__name__=="__main__"):
    root_dir = "C:/Users/82108/Desktop/스터디 폴더/Recurrent Neural Network"
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
