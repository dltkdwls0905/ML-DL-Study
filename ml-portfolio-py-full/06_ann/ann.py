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

  input_features = torch.tensor(input_features, dtype=torch.float).to(device)
  labels = torch.tensor(labels, dtype=torch.float).to(device)

  return (input_features, labels)

def tensor2list(input_tensor):
    return input_tensor.cpu().detach().numpy().tolist()


if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'

input_features, labels = load_dataset("/gdrive/My Drive/colab/ann/xor/train.txt",device)


model = nn.Sequential(
          nn.Linear(2, 10, bias=True), nn.ReLU(), nn.Dropout(0.1),
          nn.Linear(10, 10, bias=True), nn.ReLU(), nn.Dropout(0.1),
          nn.Linear(10, 10, bias=True), nn.ReLU(), nn.Dropout(0.1),
          nn.Linear(10, 10, bias=True), nn.ReLU(), nn.Dropout(0.1),
          nn.Linear(10, 10, bias=True), nn.ReLU(), nn.Dropout(0.1),
          nn.Linear(10, 10, bias=True), nn.ReLU(), nn.Dropout(0.1),
          nn.Linear(10, 10, bias=True), nn.ReLU(), nn.Dropout(0.1),
          nn.Linear(10, 1, bias=True), nn.Sigmoid()).to(device)


loss_func = torch.nn.BCELoss().to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.2)


model.train()


for epoch in range(3001):


    optimizer.zero_grad()


    hypothesis = model(input_features)


    cost = loss_func(hypothesis, labels)
    cost.backward()
    optimizer.step()

   
    if epoch % 300 == 0:
        print(epoch, cost.item())

model.eval()

with torch.no_grad():
    hypothesis = model(input_features)
    logits = (hypothesis > 0.5).float()
    predicts = tensor2list(logits)
    golds = tensor2list(labels)
    print("PRED=",predicts)
    print("GOLD=",golds)
    print("Accuracy : {0:f}".format(accuracy_score(golds, predicts)))
