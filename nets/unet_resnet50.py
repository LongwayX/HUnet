""" -*- coding: utf-8 -*-
@ Time: 2021/11/2 10:30
@ author: Zhang Chi
@ e-mail: zhang_chi@seu.edu.cn
@ file: unet_resnet50.py
@ project: My_Seg_Pytorch
"""
import torch
import torch.nn as nn
from torchvision.models import resnet50


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


class UNetRes(nn.Module):
    def __init__(self, num_classes=21, in_channels=3, pretrained=False):
        super(UNetRes, self).__init__()
        # self.backbone = resnet50()

        out_filters = [64, 256, 512, 1024, 2048]
        in_filters = [192, 384, 768, 1536, 3072]

        # self.conv1 = resnet50().conv1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_filters[0], 3, padding=1),
            nn.BatchNorm2d(out_filters[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_filters[0], out_filters[0], 3, padding=1),
            nn.BatchNorm2d(out_filters[0]),
            nn.ReLU(inplace=True)
        )

        self.bn1 = resnet50().bn1
        self.relu = resnet50().relu
        self.maxpool = resnet50().maxpool
        self.layer1 = resnet50().layer1
        self.layer2 = resnet50().layer2
        self.layer3 = resnet50().layer3
        self.layer4 = resnet50().layer4

        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[4], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[3], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[2], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[1], out_filters[0])

        # final conv (without any concat)
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

    def forward(self, inputs):
        conv1 = self.conv(inputs)  # 64×512×512
        conv2 = self.bn1(conv1)  # 64×512×512
        conv2 = self.relu(conv2)  # 64×512×512
        max_pool = self.maxpool(conv2)  # 64×256×256
        layer1 = self.layer1(max_pool)  # 256×256×256
        layer2 = self.layer2(layer1)  # 512×128×128
        layer3 = self.layer3(layer2)  # 1024×64×64
        layer4 = self.layer4(layer3)  # 2048×32×32

        up4 = self.up_concat4(layer3, layer4)  # 1024×64×64   2048×32×32
        up3 = self.up_concat3(layer2, up4)  # 512×128×128   1024×64×64
        up2 = self.up_concat2(layer1, up3)  # 256×256×256   512×128×128
        up1 = self.up_concat1(torch.cat([conv2, conv2], dim=1), up2)  # 128×512×512   256×256×256

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
