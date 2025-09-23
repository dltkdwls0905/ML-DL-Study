# 1D GAN: 정규분포 N(2, 0.5) 근사
import torch, torch.nn as nn, torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"
target_mu, target_sigma = 2.0, 0.5

def sample_real(n):
    return torch.randn(n,1)*target_sigma + target_mu

def sample_z(n):
    return torch.randn(n,8)

G = nn.Sequential(nn.Linear(8,16), nn.ReLU(), nn.Linear(16,16), nn.ReLU(), nn.Linear(16,1)).to(device)
D = nn.Sequential(nn.Linear(1,16), nn.ReLU(), nn.Linear(16,16), nn.ReLU(), nn.Linear(16,1), nn.Sigmoid()).to(device)

optG = optim.Adam(G.parameters(), lr=1e-3)
optD = optim.Adam(D.parameters(), lr=1e-3)
bce = nn.BCELoss()

for step in range(2000):
    # Train D
    xr = sample_real(64).to(device)
    zr = sample_z(64).to(device); xf = G(zr).detach()
    Dr = D(xr); Df = D(xf)
    lossD = bce(Dr, torch.ones_like(Dr)) + bce(Df, torch.zeros_like(Df))
    optD.zero_grad(); lossD.backward(); optD.step()

    # Train G
    z = sample_z(64).to(device); xf = G(z)
    Dxf = D(xf)
    lossG = bce(Dxf, torch.ones_like(Dxf))
    optG.zero_grad(); lossG.backward(); optG.step()

    if (step+1)%400==0:
        print(f"step {step+1}  lossD={lossD.item():.3f}  lossG={lossG.item():.3f}")

with torch.no_grad():
    samples = G(sample_z(1000).to(device)).cpu().numpy()
    print("Generated mean/std:", samples.mean(), samples.std())
