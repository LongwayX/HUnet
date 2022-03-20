""" -*- coding: utf-8 -*-
@ Time: 2021/10/22 23:20
@ author: Zhang Chi
@ e-mail: zhang_chi@seu.edu.cn
@ file: unet_saa_apg_vgg.py
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

        x_h_q = self.conv_q(x).contiguous().view(b, h, c * w)
        x_h_k = self.conv_k(x).contiguous().view(b, c * w, h)

        # 之前这里忘记permute了，如果不permute，纵向计算的结果和横向是一模一样的
        x_w_q = self.conv_q(x).permute(0, 1, 3, 2).contiguous().view(b, w, c * h)
        x_w_k = self.conv_k(x).permute(0, 1, 3, 2).contiguous().view(b, c * h, w)

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


class HybridConv(nn.Module):
    def __init__(self, in_ch, out_ch, reduction=4):
        super(HybridConv, self).__init__()
        # 其实我们的多尺度卷积，完全可以通过迭代conv算子来实现
        # 比如，第一尺度，就一个卷积，然后第二尺度在第一尺度基础上级联一个卷积，依次类推
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // reduction, 1, padding=0),
            nn.BatchNorm2d(out_ch // reduction),
            nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(out_ch // reduction, out_ch // reduction, 3, padding=1),
            nn.BatchNorm2d(out_ch // reduction),
            nn.ReLU(inplace=True))
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(out_ch // reduction, out_ch // reduction, 3, padding=1),
            nn.BatchNorm2d(out_ch // reduction),
            nn.ReLU(inplace=True))
        self.conv1_4 = nn.Sequential(
            nn.Conv2d(out_ch // reduction, out_ch // reduction, 3, padding=1),
            nn.BatchNorm2d(out_ch // reduction),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        conv1_1 = self.conv1_1(input)
        conv1_2 = self.conv1_2(conv1_1)
        conv1_3 = self.conv1_3(conv1_2)
        conv1_4 = self.conv1_4(conv1_3)
        conv = torch.cat((conv1_1, conv1_2, conv1_3, conv1_4), dim=1)

        return conv


class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.axis_att = AxisAttention(in_planes)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.axis_att(x)
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class UNet_SAA_AGP(nn.Module):
    def __init__(self, in_channels, num_classes, pretrained=False):
        super(UNet_SAA_AGP, self).__init__()
        filter = [64, 128, 256, 512, 1024]
        in_filters = [192, 384, 768, 1024]
        out_filters = [64, 128, 256, 512]

        self.vgg = VGG16(pretrained=pretrained, in_channels=in_channels)
        # encoder
        self.conv1_2 = HybridConv(filter[0], filter[0])
        self.ca1 = ChannelAttention(filter[0])

        self.conv2_2 = HybridConv(filter[1], filter[1])
        self.ca2 = ChannelAttention(filter[1])

        self.conv3_2 = HybridConv(filter[2], filter[2])
        self.ca3 = ChannelAttention(filter[2])

        self.conv4_2 = HybridConv(filter[3], filter[3])
        self.ca4 = ChannelAttention(filter[3])

        # decoder
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
        conv1_2 = self.conv1_2(conv1_1)
        conv1_2 = self.ca1(conv1_2) * conv1_2

        conv2_1 = self.vgg.features[4:9](conv1_2)
        conv2_2 = self.conv2_2(conv2_1)
        conv2_2 = self.ca2(conv2_2) * conv2_2

        conv3_1 = self.vgg.features[9:16](conv2_2)
        conv3_2 = self.conv3_2(conv3_1)
        conv3_2 = self.ca3(conv3_2) * conv3_2

        conv4_1 = self.vgg.features[16:23](conv3_2)
        conv4_2 = self.conv4_2(conv4_1)
        conv4_2 = self.ca4(conv4_2) * conv4_2

        conv5 = self.vgg.features[23:-1](conv4_2)

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