from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(ConvBlock, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        return self.double_conv(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.down_sample = nn.Sequential(nn.MaxPool2d(2),
                                         ConvBlock(in_channels, out_channels))
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        return self.down_sample(x)



class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, learnable=True):
        if learnable:
            self.up_sample = UpSampleConv(in_channels, out_channels)
        else:
            self.up_sample = UpSampleInterpolation(in_channels, out_channels)

    def forward(self, x):
        return self.up_sample(x)

class UpSampleConv:
    def __init__(self, in_channels, out_channels):
        self.up_sample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.up_sample(x)
        return self.conv(x)

class UpSampleInterpolation:
    def __init__(self, in_channels, out_channels):
        self.up_sample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.up_sample(x)
        return self.conv(x)
