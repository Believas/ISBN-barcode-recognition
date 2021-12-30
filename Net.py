import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=784, out_features=512),
            nn.Dropout(0.5),
            nn.BatchNorm1d(512),  # N, H, W
            # nn.LayerNorm(512),  # C, H, W
            # nn.InstanceNorm1d(512),  # H, W  (要求输入数据三维)
            # nn.GroupNorm(2, 512)  # C, H, W,  将512分成两组
            nn.ReLU()
        )  # N, 512
        self.layer2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.Dropout(0.5),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )  # N, 256
        self.layer3 = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            nn.Dropout(0.5),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )  # N, 128
        self.layer4 = nn.Sequential(
            nn.Linear(in_features=128, out_features=12),
        )  # N, 10

    def forward(self, x):
        # x = torch.reshape(x, [1, x.size(0), -1])  # 形状[1, N, C*H*W]
        # print(x.shape)
        # y1 = self.layer1(x)[0]   # 这两行代码适用于在InstanceNorm1d的情况。将第一维去掉，变成两维

        x = torch.reshape(x, [x.size(0), -1])  # 形状[N, C*H*W]
        y1 = self.layer1(x)
        # y1 = torch.dropout(y1, 0.5, True)

        y2 = self.layer2(y1)
        # y2 = torch.dropout(y2, 0.5, True)

        y3 = self.layer3(y2)
        # y3 = torch.dropout(y3, 0.5, True)

        self.y4 = self.layer4(y3)

        out = self.y4
        return out