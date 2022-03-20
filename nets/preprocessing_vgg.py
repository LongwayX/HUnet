""" -*- coding: utf-8 -*-
@ Time: 2021/10/22 23:53
@ author: Zhang Chi
@ e-mail: zhang_chi@seu.edu.cn
@ file: preprocessing_vgg.py
@ project: My_Seg_Pytorch
"""
import torch
import torch.nn as nn
from nets.vgg import VGG16


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


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
            nn.Conv2d(1, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, 3, padding=1),
            nn.BatchNorm2d(1),
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
        self.vgg = VGG16(pretrained=pretrained, in_channels=in_channels)
        in_filters = [192, 384, 768, 1024]
        out_filters = [64, 128, 256, 512]

        self.window = WindowTrunction(in_channels, in_channels)
        # decoder
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        # final conv (without any concat)
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

    def forward(self, x):
        x = self.window(x)

        feat1 = self.vgg.features[:4](x)
        feat2 = self.vgg.features[4:9](feat1)
        feat3 = self.vgg.features[9:16](feat2)
        feat4 = self.vgg.features[16:23](feat3)
        feat5 = self.vgg.features[23:-1](feat4)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        final = self.final(up1)

        return final

    def _initialize_weights(self, *stages):
        for modules in stages:
            for module in modules.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()