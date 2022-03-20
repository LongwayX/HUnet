""" -*- coding: utf-8 -*-
@ Time: 2021/5/16 9:33
@ author: Zhang Chi
@ e-mail: zhang_chi@seu.edu.cn
@ file: preprocessing.py
@ project: Scale_Attention
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, 1, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class WindowTrunction(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(WindowTrunction, self).__init__()
        # 在输入这个网络之前，就应该进行一次预截断，-1024~1024
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_ch, 1, 1, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(1, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.coord_att = CoordAtt(1, 1)

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.threshold = nn.Sequential(
            nn.Linear(in_features=1, out_features=1),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1, out_features=1),
            nn.Sigmoid())

        self.threshold_linear = nn.Sequential(
            nn.Linear(in_features=1, out_features=1),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1, out_features=1),
            nn.Sigmoid())

        self.post_conv = nn.Sequential(
            nn.Conv2d(1, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # th = torch.tensor([1024.0]).to('cuda')
        # th = th.expand_as(x)
        # x = torch.div(x, th)  # 这里做了归一化

        x = self.pre_conv(x)
        x = x + self.conv(x)  # 这里做了平移变换

        squeeze = self.squeeze(self.coord_att(x).detach())
        threshold_up = self.threshold(squeeze)

        # b, c, _, _ = x.size()
        # if b != 1:
        #     for item in range(b):
        #         x[item] = torch.clamp(x[item], 1e-6, threshold_up[item].item())
        # else:
        #     x = torch.clamp(x, 1e-6, threshold_up.item())

        x = torch.clamp(x, 1e-6, threshold_up.mean().item())
        x = self.post_conv(x)

        return x


class Preprocessing(nn.Module):
    def __init__(self, in_channels, num_classes, pretrained=False):
        super(Preprocessing, self).__init__()

        self.window = WindowTrunction(in_channels, 1)

        # encoder
        self.conv1 = DoubleConv(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = DoubleConv(512, 1024)

        # decoder
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x = self.window(x)

        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)

        return c10
