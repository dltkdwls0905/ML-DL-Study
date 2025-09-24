import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

def load_dataset(file, device):
  data = np.loadtxt(file)
  print("DATA=",data)

  input_features = data[:,0:-1]
  print("INPUT_FEATURES=",input_features)

  labels = np.reshape(data[:,-1],(4,1))
  print("LABELS=",labels)

  input_features = torch.tensor(input_features, dtype=torch.float)
  labels = torch.tensor(labels, dtype=torch.float)

  return (input_features, labels)

def tensor2List(input_tensor):
  return input_tensor.cpu().detach().numpy().tolist()

# GPU 사용 가능 여부 확인
if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'

input_features, labels = load_dataset("C:/Users/82108/Desktop/스터디 폴더/Artificial Neural Network/XOR/train.txt",device)

# NN 모델 만들기
model = nn.Sequential(
          nn.Linear(2, 10, bias=True), nn.ReLU(), nn.Dropout(0.1),
          nn.Linear(10, 10, bias=True), nn.ReLU(), nn.Dropout(0.1),
          nn.Linear(10, 10, bias=True), nn.ReLU(), nn.Dropout(0.1),
          nn.Linear(10, 10, bias=True), nn.ReLU(), nn.Dropout(0.1),
          nn.Linear(10, 10, bias=True), nn.ReLU(), nn.Dropout(0.1),
          nn.Linear(10, 10, bias=True), nn.ReLU(), nn.Dropout(0.1),
          nn.Linear(10, 10, bias=True), nn.ReLU(), nn.Dropout(0.1),
          nn.Linear(10, 1, bias=True), nn.Sigmoid()).to(device)

# 이진분류 크로스엔트로피 비용 함수
loss_func = torch.nn.BCELoss().to(device)
# 옵티마이저 함수 (역전파 알고리즘을 수행할 함수)
optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

# 학습 모드 셋팅
model.train()

# 모델 학습
for epoch in range(3001):

    # 기울기 계산한 것들 초기화
    optimizer.zero_grad()

    # H(X) 계산: forward 연산
    hypothesis = model(input_features)

    # 비용 계산
    cost = loss_func(hypothesis, labels)
    # 역전파 수행
    cost.backward()
    optimizer.step()

    # 1000 에폭마다 비용 출력
    if epoch % 300 == 0:
        print(epoch, cost.item())

# 평가 모드 셋팅 (학습 시에 적용했던 드랍 아웃 여부 등을 비적용)
model.eval()

# 역전파를 적용하지 않도록 context manager 설정
with torch.no_grad():
    hypothesis = model(input_features)
    logits = (hypothesis > 0.5).float()
    predicts = tensor2List(logits)
    golds = tensor2List(labels)
    print("PRED=",predicts)
    print("GOLD=",golds)
    print("Accuracy : {0:f}".format(accuracy_score(golds, predicts)))
