""" -*- coding: utf-8 -*-
@ Time: 2021/11/14 16:18
@ author: Zhang Chi
@ e-mail: zhang_chi@seu.edu.cn
@ file: unet_axis_vgg.py
@ project: My_Seg_Pytorch
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

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


class AxisAttention(nn.Module):
    def __init__(self, inp, reduction=32):
        super(AxisAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv_q = nn.Conv2d(in_channels=inp, out_channels=inp, kernel_size=1, stride=1,
                                padding=0, bias=False)
        self.conv_k = nn.Conv2d(in_channels=inp, out_channels=inp, kernel_size=1, stride=1,
                                padding=0, bias=False)
        self.conv_v = nn.Conv2d(in_channels=inp, out_channels=inp, kernel_size=1, stride=1,
                                padding=0, bias=False)

        self.softmax = nn.Softmax(dim=1)

        self.med_conv = nn.Sequential(
            nn.Conv2d(2 * inp, inp, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(inp),
            nn.Conv2d(inp, inp, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(inp)
        )

        self.softmax2 = nn.Sigmoid()

        self.conv_mask = nn.Conv2d(in_channels=inp, out_channels=inp, kernel_size=1, stride=1,
                                   padding=0, bias=False)

    def forward(self, x):
        identity = x
        b, c, h, w = x.size()

        x_h_q = self.conv_q(x).view(b, h, c * w)
        x_h_k = self.conv_k(x).view(b, c * w, h)

        x_w_q = self.conv_q(x).permute(0, 1, 3, 2).view(b, w, c * h)
        x_w_k = self.conv_k(x).permute(0, 1, 3, 2).view(b, c * h, w)

        x_h_qk = self.softmax(torch.matmul(x_h_q, x_h_k))  # attention的计算结果
        x_w_qk = self.softmax(torch.matmul(x_w_q, x_w_k))

        x_h_qkv = torch.matmul(x_h_qk, x_h_q)  # attention的加权结果
        x_w_qkv = torch.matmul(x_w_qk, x_w_q)

        x_w_qkv = x_w_qkv.permute(0, 2, 1)

        x_h_qkv = x_h_qkv.view(b, h, c, w).permute(0, 2, 1, 3)
        x_w_qkv = x_w_qkv.view(b, w, c, h).permute(0, 2, 3, 1)

        x_h_w_qkv = torch.cat([x_h_qkv, x_w_qkv], dim=1)
        x_h_w_qkv = self.softmax2(self.med_conv(x_h_w_qkv))

        out = identity * self.conv_mask(x_h_w_qkv)  # 问题应该出现在这里

        return out


class UNet_Axis_Att(nn.Module):
    def __init__(self, num_classes=21, in_channels=3, pretrained=False):
        super(UNet_Axis_Att, self).__init__()
        self.vgg = VGG16(pretrained=pretrained, in_channels=in_channels)
        in_filters = [192, 384, 768, 1024]
        out_filters = [64, 128, 256, 512]
        filter = [64, 128, 256, 512, 1024]

        # encoder
        self.axis1 = AxisAttention(filter[0])

        self.axis2 = AxisAttention(filter[1])

        self.axis3 = AxisAttention(filter[2])

        self.axis4 = AxisAttention(filter[3])

        # decoder
        # upsampling
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
        conv1_1 = self.vgg.features[:4](x)
        conv1 = conv1_1 + self.axis1(conv1_1)

        conv2_1 = self.vgg.features[4:9](conv1)
        conv2 = conv2_1 + self.axis2(conv2_1)

        conv3_1 = self.vgg.features[9:16](conv2)
        conv3 = conv3_1 + self.axis3(conv3_1)

        conv4_1 = self.vgg.features[16:23](conv3)
        conv4 = conv4_1 + self.axis4(conv4_1)

        conv5 = self.vgg.features[23:-1](conv4)

        up4 = self.up_concat4(conv4_1, conv5)
        up3 = self.up_concat3(conv3_1, up4)
        up2 = self.up_concat2(conv2_1, up3)
        up1 = self.up_concat1(conv1_1, up2)

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




