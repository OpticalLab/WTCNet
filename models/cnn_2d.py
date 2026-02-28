import torch.nn as nn
import torch

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(12, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # -> (bs,64,1,1)
        self.fc = nn.Linear(64, 6)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.global_pool(x)       # (bs,64,1,1)
        x = x.view(x.size(0), -1)     # (bs,64)
        return self.fc(x)             # (bs,5)

if __name__ == '__main__':
    model = CNN()
    print('Total params:', sum(p.numel() for p in model.parameters()))
    dummy = torch.randn(2, 12, 100, 100)
    out = model(dummy)
    print(out.shape)   # torch.Size([2, 5])