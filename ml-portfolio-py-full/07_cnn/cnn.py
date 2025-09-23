# 간단 CNN으로 sklearn digits(8x8) 분류 (다운로드 없이 가능)
import torch, torch.nn as nn, torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np

digits = load_digits()
X = digits.images.astype(np.float32) / 16.0  # (n,8,8)
y = digits.target
X = X[:,None,:,:]  # (n,1,8,8)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

Xtr = torch.tensor(Xtr); ytr = torch.tensor(ytr, dtype=torch.long)
Xte = torch.tensor(Xte); yte = torch.tensor(yte, dtype=torch.long)

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 8->4
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32, 10))
    def forward(self,x): return self.classifier(self.features(x))

model = SmallCNN()
opt = optim.Adam(model.parameters(), lr=1e-3)
lossf = nn.CrossEntropyLoss()

for ep in range(10):
    model.train(); opt.zero_grad()
    out = model(Xtr); loss = lossf(out, ytr); loss.backward(); opt.step()
    acc = (out.argmax(1)==ytr).float().mean().item()
    print(f"[{ep}] loss={loss.item():.4f} acc={acc:.3f}")

model.eval()
te_acc = (model(Xte).argmax(1)==yte).float().mean().item()
print("Test Acc:", te_acc)
