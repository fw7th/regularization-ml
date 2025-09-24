import torch.nn.functional as F
import torch.nn as nn
import torch


# Define a small CNN
class MiniCNN(nn.Module):
    """
    Basic CNN for experiments
    """

    def __init__(self, conv_layers, fc_layers):
        super().__init__()
        self.conv = nn.Sequential(*conv_layers)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # GAP for variable input
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, pool=True, dropout=False, drop_val=0.0
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        if dropout:
            layers.append(nn.Dropout2d(drop_val))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class FCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False, drop_val=0.0):
        super().__init__()
        layers = [nn.Linear(in_channels, out_channels), nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(drop_val))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
