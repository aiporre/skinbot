from torch import nn
from torch import cat


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(ConvBlock, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding='same', bias=False),
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

def pad_to(x, stride):
    h, w = x.shape[-2:]

    if h % stride > 0:
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0:
        new_w = w + stride - w % stride
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pads = (lw, uw, lh, uh)

    # zero-padding by default.
    # See others at https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad
    out = nn.functional.pad(x, pads, "constant", 0)

    return out, pads

def unpad(x, pad):
    if pad[2]+pad[3] > 0:
        x = x[:,:,pad[2]:-pad[3],:]
    if pad[0]+pad[1] > 0:
        x = x[:,:,:,pad[0]:-pad[1]]
    return x

def pad_to_match(x, H, W):
    H_in, W_in = x.shape[-2], x.shape[-1]
    dH, dW = H-H_in, W-W_in
    padding = (dW // 2, dW - dW // 2,
               dH // 2, dH - dH // 2)
    return nn.functional.pad(x, padding)

class UpSampleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampleConv, self).__init__()
        self.up_sample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, x_skip):
        x = self.up_sample(x)
        x = pad_to_match(x, x_skip.shape[-2], x_skip.shape[-1])
        x = cat([x_skip, x], dim=1)
        return self.conv(x)

class UpSampleInterpolation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampleInterpolation, self).__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_channels, out_channels, mid_channels=in_channels//2)

    def forward(self, x, x_skip):
        x = self.up_sample(x)
        x = pad_to_match(x, x_skip.shape[-2], x_skip.shape[-1])
        x = cat([x_skip, x], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, learnable_upsample=True):
        super(UNet, self).__init__()
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
        x, padding = pad_to(x, 2)
        print(padding)
        x1 = self.first(x)
        d1 = self.down_1(x1)
        d2 = self.down_2(d1)
        d3 = self.down_3(d2)
        d4 = self.down_4(d3)
        x = self.up_1(d4, d3)
        x = self.up_2(x, d2)
        x = self.up_3(x, d1)
        x = self.up_4(x, x1)
        x = self.outc(x)
        x = unpad(x, padding)
        return x

