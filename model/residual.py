import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.elu = nn.ELU()
        self.conv2 = nn.Conv1d(out_channels, in_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.elu(out)
        out = self.conv2(out)
        out = self.elu(out)
        out = out + residual
        out = self.conv3(out)
        out = self.elu(out)
        return out

class ResidualModel(nn.Module):
    def __init__(self, in_channels=1):
        super(ResidualModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 8, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.resblock1 = ResidualBlock(8, 16)
        self.resblock2 = ResidualBlock(16, 24)
        self.resblock3 = ResidualBlock(24, 32)
        self.resblock4 = ResidualBlock(32, 48)
        self.resblock5 = ResidualBlock(48, 64)
        self.conv2 = nn.Conv1d(64, 1, kernel_size=3, stride=1, padding=1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.elu = nn.ELU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.elu(x)
        x = self.maxpool(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.conv2(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        return x

if __name__ == '__main__':
    model = ResidualModel(in_channels=3)
    summary(model, input_size=(1, 3, 300),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        row_settings=["var_names"])