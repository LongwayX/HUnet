""" -*- coding: utf-8 -*-
@ Time: 2021/12/21 15:16
@ author: Zhang Chi
@ e-mail: zhang_chi@seu.edu.cn
@ file: unet_outlook_vgg.py
@ project: My_Seg_Pytorch
"""
import torch
import torch.nn as nn
import numpy as np
from torch.nn import init
import math
from torch.nn import functional as F

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


class OutlookAttention(nn.Module):

    def __init__(self, dim, num_heads=1, kernel_size=3, padding=1, stride=1, qkv_bias=False,
                 attn_drop=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.scale = self.head_dim ** (-0.5)

        self.v_pj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn = nn.Linear(dim, kernel_size ** 4 * num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(attn_drop)

        self.unflod = nn.Unfold(kernel_size, padding, stride)  # 手动卷积
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

    def forward(self, x):
        B, H, W, C = x.shape

        # 映射到新的特征v
        v = self.v_pj(x).permute(0, 3, 1, 2)  # B,C,H,W
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)
        v = self.unflod(v).reshape(B, self.num_heads, self.head_dim, self.kernel_size * self.kernel_size,
                                   h * w).permute(0, 1, 4, 3, 2).contiguous()  # B,num_head,H*W,kxk,head_dim

        # 生成Attention Map
        attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # B,H,W,C
        attn = self.attn(attn).reshape(B, h * w, self.num_heads, self.kernel_size * self.kernel_size,
                                       self.kernel_size * self.kernel_size).permute(0, 2, 1, 3,
                                                                                    4).contiguous()  # B，num_head，H*W,kxk,kxk
        attn = self.scale * attn
        attn = attn.softmax(-1)
        attn = self.attn_drop(attn)

        # 获取weighted特征
        out = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size,
                                                        h * w)  # B,dimxkxk,H*W
        out = F.fold(out, output_size=(H, W), kernel_size=self.kernel_size,
                     padding=self.padding, stride=self.stride)  # B,C,H,W
        out = self.proj(out.permute(0, 2, 3, 1).contiguous())  # B,H,W,C
        out = self.proj_drop(out)

        return out


class UNet_VOLO(nn.Module):
    def __init__(self, num_classes=21, in_channels=3, pretrained=False):
        super(UNet_VOLO, self).__init__()
        self.vgg = VGG16(pretrained=pretrained, in_channels=in_channels)
        in_filters = [192, 384, 768, 1024]
        out_filters = [64, 128, 256, 512]
        filter = [64, 128, 256, 512, 1024]

        # encoder
        self.volo1 = OutlookAttention(dim=filter[0])

        self.volo2 = OutlookAttention(dim=filter[1])

        self.volo3 = OutlookAttention(dim=filter[2])

        self.volo4 = OutlookAttention(dim=filter[3])

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
        conv1 = conv1_1 + self.volo1(conv1_1.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()

        conv2_1 = self.vgg.features[4:9](conv1)
        conv2 = conv2_1 + self.volo2(conv2_1.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()

        conv3_1 = self.vgg.features[9:16](conv2)
        conv3 = conv3_1 + self.volo3(conv3_1.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()

        conv4_1 = self.vgg.features[16:23](conv3)
        conv4 = conv4_1 + self.volo4(conv4_1.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()

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


if __name__ == '__main__':
    input = torch.rand(1, 3, 512, 512)
    model = UNet_VOLO(3, 3)
    output = model(input)







