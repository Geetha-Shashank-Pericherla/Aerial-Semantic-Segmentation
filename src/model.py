import torch
import torch.nn as nn

class TwoConvBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TwoConvBlock(3, 64, 64)
        self.decoder = TwoConvBlock(64, 128, 128)
        self.final_conv = nn.Conv2d(128, 24, kernel_size=1)  # 24 classes

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final_conv(x)
        return x
