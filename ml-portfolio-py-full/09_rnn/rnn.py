# LSTM으로 사인파 예측 (toy 시계열)
import torch, torch.nn as nn, torch.optim as optim
import math

T = 200
xs = torch.linspace(0, 8*math.pi, T)
ys = torch.sin(xs).unsqueeze(1)

# sequence 데이터 만들기
window = 20
X=[]; Y=[]
for i in range(T-window):
    X.append(ys[i:i+window])
    Y.append(ys[i+window])
X = torch.stack(X)  # (N, window, 1)
Y = torch.stack(Y)  # (N, 1)

class LSTMReg(nn.Module):
    def __init__(self, hid=16):
        super().__init__()
        self.lstm = nn.LSTM(1, hid, batch_first=True)
        self.fc = nn.Linear(hid, 1)
    def forward(self, x):
        o,(h,c) = self.lstm(x)
        return self.fc(o[:,-1])
model = LSTMReg()
opt = optim.Adam(model.parameters(), lr=0.01)
lossf = nn.MSELoss()

for ep in range(200):
    opt.zero_grad(); pred = model(X); loss = lossf(pred, Y); loss.backward(); opt.step()
print("final MSE:", loss.item())
