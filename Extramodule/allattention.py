import torch
import torch.nn as nn
import torch.nn.functional as F

class SEAttention(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super(SEAttention, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return x * out

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super(ChannelAttention, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avgpool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.maxpool(x))))
        out = self.sigmoid(avg_out + max_out)
        return x * out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        return x * out
class CBAMAttention(nn.Module):
    def __init__(self, in_channels, reduction=32, kernel_size=3):
        super(CBAMAttention, self).__init__()
        self.cbam_attention = nn.Sequential(ChannelAttention(in_channels, reduction),
                                            SpatialAttention(kernel_size))

    def forward(self, x):
        # CBAM Attention
        cbam_out = self.cbam_attention(x)
        return cbam_out

class CombinedAttention(nn.Module):
    def __init__(self, in_channels, reduction=32, kernel_size=3):
        super(CombinedAttention, self).__init__()
        self.se_attention = SEAttention(in_channels, reduction)
        self.cbam_attention = nn.Sequential(ChannelAttention(in_channels, reduction),
                                            SpatialAttention(kernel_size))
        self.conv = nn.Conv2d(in_channels *2, in_channels, kernel_size=1)

    def forward(self, x):
        # SE Attention
        se_out = self.se_attention(x)
        # CBAM Attention
        cbam_out = self.cbam_attention(x)
        combined_out = torch.cat([se_out,cbam_out], dim=1)
        return self.conv(combined_out)
