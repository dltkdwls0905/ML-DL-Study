# PyTorch Transformer로 toy copy task (입력=출력) 학습
import torch, torch.nn as nn, torch.optim as optim

torch.manual_seed(0)
vocab_size = 20
seq_len = 12
N = 512

X = torch.randint(1, vocab_size, (N, seq_len))  # 0은 패딩 가정
Y = X.clone()

class TinyTransformer(nn.Module):
    def __init__(self, vocab, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model)*0.01)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab)
    def forward(self, x):
        h = self.emb(x) + self.pos[:, :x.size(1), :]
        h = self.encoder(h)
        return self.fc(h)

model = TinyTransformer(vocab_size)
opt = optim.Adam(model.parameters(), lr=1e-3)
lossf = nn.CrossEntropyLoss()

for ep in range(200):
    opt.zero_grad()
    logits = model(X)             # (N, L, V)
    loss = lossf(logits.view(-1, vocab_size), Y.view(-1))
    loss.backward(); opt.step()
    if (ep+1)%50==0: print(f"epoch {ep+1}, loss {loss.item():.3f}")

with torch.no_grad():
    out = model(X[:1]).argmax(-1)
    print("input :", X[0].tolist())
    print("output:", out[0].tolist())
