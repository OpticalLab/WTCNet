import torch
from torch import nn
import torch.nn.functional as F
from functools import partial


class SKConv(nn.Module):
    def __init__(self, in_channels, out_channels=None, stride=1,
                 M=2, r=16, L=32):
        super(SKConv, self).__init__()
        out_channels = out_channels or in_channels
        self.M = M
        self.out_channels = out_channels
        d = max(out_channels // r, L)

        # 多分支卷积
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride,
                          padding=1 + i, dilation=1 + i,
                          groups=32, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for i in range(M)
        ])

        self.gap = nn.AdaptiveAvgPool2d(1)

        # 用两个 Linear 代替 Conv1d
        self.fc = nn.Sequential(
            nn.Linear(out_channels, d, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d, out_channels * M, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch, c, h, w = x.size()

        # 1. 多分支特征
        feats = [conv(x) for conv in self.convs]
        feats = torch.stack(feats, dim=1)          # (B,M,C,H,W)

        # 2. 融合 U
        U = feats.sum(dim=1)                       # (B,C,H,W)
        s = self.gap(U).view(batch, c)             # (B,C)

        # 3. 通道注意力
        z = self.fc(s).view(batch, self.M, self.out_channels, 1, 1)  # (B,M,C,1,1)
        att = self.softmax(z)                      # 沿 M 维 softmax

        # 4. 重加权融合
        out = (feats * att).sum(dim=1)
        return out


class CNN_SKA(nn.Module):
    def __init__(self, num_classes=6):
        super(CNN_SKA, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(64, 128, 3, 1, 1)
        self.relu4 = nn.ReLU()

        # ===== 在这里插入 SK Attention =====
        self.sk = SKConv(128, 128)   # 输入输出通道一致

        # 后续分类头
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.maxpool1(self.relu1(self.conv1(x)))
        x = self.maxpool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))

        x = self.relu4(self.conv4(x))

        x = self.sk(x)           # <-- SK 模块

        x = self.avgpool(x).flatten(1)
        out = self.fc(x)
        return out


# ----------- 测试用例 -----------
if __name__ == '__main__':
    net = CNN_SKA(num_classes=6)
    x = torch.randn(2, 1, 4000, 12)
    print(net(x).shape)   # -> torch.Size([2, 6])