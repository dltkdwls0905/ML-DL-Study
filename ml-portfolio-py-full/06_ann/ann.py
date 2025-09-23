import torch, torch.nn as nn, torch.optim as optim

# XOR 데이터셋
X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
y = torch.tensor([0,1,1,0], dtype=torch.long)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8), nn.ReLU(),
            nn.Linear(8, 2)
        )
    def forward(self, x): return self.net(x)

model = MLP()
opt = optim.Adam(model.parameters(), lr=0.02)
lossf = nn.CrossEntropyLoss()

for ep in range(800):
    opt.zero_grad()
    out = model(X)
    loss = lossf(out, y)
    loss.backward(); opt.step()
print("Pred:", model(X).argmax(1).tolist())
