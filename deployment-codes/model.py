"""
SiameseUNet_ASPP_DS — model architecture.
Kept identical to training so state_dict loads cleanly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout2d(dropout)
        self.conv1  = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.bn1    = nn.BatchNorm2d(out_ch)
        self.relu1  = nn.ReLU(inplace=True)
        self.conv2  = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2    = nn.BatchNorm2d(out_ch)
        self.relu2  = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        def block(k, p, d):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, padding=p, dilation=d),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.conv1    = block(1,  0,  1)
        self.conv2    = block(3,  6,  6)
        self.conv3    = block(3, 12, 12)
        self.conv4    = block(3, 18, 18)
        self.gpool    = nn.AdaptiveAvgPool2d(1)
        self.conv_gp  = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.project  = nn.Conv2d(out_ch * 5, out_ch, 1)

    def forward(self, x):
        sz = x.shape[-2:]
        f1 = self.conv1(x)
        f2 = self.conv2(x)
        f3 = self.conv3(x)
        f4 = self.conv4(x)
        gp = F.interpolate(self.conv_gp(self.gpool(x)),
                           size=sz, mode='bilinear', align_corners=False)
        return self.project(torch.cat([f1, f2, f3, f4, gp], dim=1))


class Encoder(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        self.conv1 = DoubleConv(in_channels, 32)
        self.conv2 = DoubleConv(32,  64)
        self.conv3 = DoubleConv(64,  128)
        self.conv4 = DoubleConv(128, 256)
        self.conv5 = DoubleConv(256, 512)
        self.pool  = nn.MaxPool2d(2)

    def forward(self, x):
        f1 = self.conv1(x);  p1 = self.pool(f1)
        f2 = self.conv2(p1); p2 = self.pool(f2)
        f3 = self.conv3(p2); p3 = self.pool(f3)
        f4 = self.conv4(p3); p4 = self.pool(f4)
        f5 = self.conv5(p4)
        return [f1, f2, f3, f4, f5], f5


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, 1),
        )
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:],
                              mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class SiameseUNet_ASPP_DS(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        self.encoder = Encoder(in_channels)
        self.aspp    = ASPP(512, 256)

        self.up5 = UpBlock(256 * 3, 512 * 3, 512)
        self.up4 = UpBlock(512,     256 * 3, 256)
        self.up3 = UpBlock(256,     128 * 3, 128)
        self.up2 = UpBlock(128,      64 * 3,  64)
        self.up1 = UpBlock( 64,      32 * 3,  32)

        self.out_main = nn.Conv2d(32,  1, 1)
        self.out3     = nn.Conv2d(256, 1, 1)
        self.out2     = nn.Conv2d(128, 1, 1)

    def forward(self, t1, t2):
        feat1, b1 = self.encoder(t1)
        feat2, b2 = self.encoder(t2)

        b1 = self.aspp(b1)
        b2 = self.aspp(b2)

        diff = torch.abs(b1 - b2)
        x    = torch.cat([b1, b2, diff], dim=1)

        s5 = torch.cat([feat1[4], feat2[4], torch.abs(feat1[4]-feat2[4])], dim=1)
        s4 = torch.cat([feat1[3], feat2[3], torch.abs(feat1[3]-feat2[3])], dim=1)
        s3 = torch.cat([feat1[2], feat2[2], torch.abs(feat1[2]-feat2[2])], dim=1)
        s2 = torch.cat([feat1[1], feat2[1], torch.abs(feat1[1]-feat2[1])], dim=1)
        s1 = torch.cat([feat1[0], feat2[0], torch.abs(feat1[0]-feat2[0])], dim=1)

        x    = self.up5(x,  s5)
        x    = self.up4(x,  s4);  out3 = self.out3(x)
        x    = self.up3(x,  s3);  out2 = self.out2(x)
        x    = self.up2(x,  s2)
        x    = self.up1(x,  s1)

        return self.out_main(x), out3, out2