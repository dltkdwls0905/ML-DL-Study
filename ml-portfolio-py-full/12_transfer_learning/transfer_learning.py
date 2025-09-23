# 전이학습 개념 데모: torchvision이 없으면 간단 CNN로 대체
import torch, torch.nn as nn

try:
    import torchvision.models as models
    print("Using torchvision ResNet18 (random init, head replace)")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2-class
except Exception as e:
    print("torchvision 사용 불가 -> 간단 CNN로 대체:", e)
    model = nn.Sequential(
        nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(), nn.Linear(16,2)
    )

x = torch.randn(4,3,64,64)
logits = model(x)
print("logits shape:", logits.shape)
