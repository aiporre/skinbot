from torch import nn
from torch import cat


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

class UpSampleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampleConv, self).__init__()
        self.up_sample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, x_skip):
        x = self.up_sample(x)
        x = cat([x_skip, x], dim=0)
        return self.conv(x)

class UpSampleInterpolation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampleInterpolation, self).__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_channels, out_channels, mid_channels=in_channels//2)

    def forward(self, x, x_skip):
        x = self.up_sample(x)
        x = cat([x_skip, x], dim=0)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, learnable_upsample=True):
        self.first = ConvBlock(in_channels, 64)
        # downsample plath
        self.down_1 = DownSample(64, 128)
        self.down_2 = DownSample(128, 256)
        self.down_3 = DownSample(256, 512)
        # upsample path
        if learnable_upsample:
            self.down_4 = DownSample(512, 1024)
            self.up_1 = UpSampleConv(1024, 512)
            self.up_2 = UpSampleConv(512, 256)
            self.up_3 = UpSampleConv(256, 128)
            self.up_4 = UpSampleConv(128, 64)
        else:
            self.down_4 = DownSample(512, 512) # concat +512
            self.up_1 = UpSampleInterpolation(1024, 256) # concat +256
            self.up_2 = UpSampleInterpolation(512, 128) # concat +128
            self.up_3 = UpSampleInterpolation(256, 64) # concat +64
            self.up_4 = UpSampleInterpolation(128, 64) # concat +
        self.outc = ConvBlock(64, num_classes)

    def forward(self, x):
        x1 = self.first(x)
        d1 = self.down_1(x)
        d2 = self.down_2(x)
        d3 = self.down_3(x)
        d4 = self.down_4(x)
        x = self.up_1(d4, d3)
        x = self.up_2(x, d2)
        x = self.up_3(x, d1)
        x = self.up_4(x, x1)
        x = self.outc(x)
        return x

