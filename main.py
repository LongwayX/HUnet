""" -*- coding: utf-8 -*-
@ Time: 2021/10/21 19:48
@ author: Zhang Chi
@ e-mail: zhang_chi@seu.edu.cn
@ file: main.py
@ project: KiTS_seg
"""
import os
import argparse
import torch
import random
import nibabel as nib
import numpy as np
import torch.backends.cudnn as cudnn

from train_valid import load_dict, training
from nets.unet_vgg import Unet
from nets.preprocessing_vgg import Preprocessing
from nets.unet_saa_agp_vgg import UNet_SAA_AGP
from nets.unet_se_agp_vgg import UNet_SE_AGP
from nets.unet_axis_vgg import UNet_Axis_Att
from nets.unet_resnet50 import UNetRes
from nets.unet_gam_agp_vgg import UNet_GAM_AGP
from nets.unet_eca_agp_vgg import UNet_ECA_AGP
from nets.unet_outlook_vgg import UNet_VOLO
from nets.HUnet import HUnet

def parse_args():
    # 训练之前检查参数
    # 训练数据集地址
    # 测试数据集地址
    # 模型名称
    # 损失是否只用dice
    # 是否加入预训练
    # 训练和测试用的txt文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="/mnt/data/zhang_chi_new/dataset_split/KiTS",
                        help="数据集地址", choices=["./raw_nii_data", "../KiTS_seg/unet_pytorch/Medical_Datasets",
                                               r"G:\dataset\KiTS_split\KiTS\test"])
    parser.add_argument('--window_value', type=list, default=[200, 30])
    # KiTS [200,30]

    parser.add_argument('--model_name', type=str, default="HUNet", choices=["preprocessing", "unet", "saa_agp", "gam_agp", "eca_agp", "volo", 'HUNet'])
    parser.add_argument('--Net', type=dict,
                        default={'unet': Unet, 'preprocessing': Preprocessing, 'saa_agp': UNet_SAA_AGP,
                                 'se_agp': UNet_SE_AGP, 'axis': UNet_Axis_Att, 'unet_res': UNetRes, 
                                 'gam_agp': UNet_GAM_AGP, 'eca_agp': UNet_ECA_AGP, 'volo': UNet_VOLO,
                                 'HUNet': HUnet})
    parser.add_argument('--dice_loss_only', type=bool, default=False)
    parser.add_argument('--dice_loss_use', type=bool, default=True, help="是否使用dice作为loss")


    parser.add_argument('--breakpoint_train', type=bool, default=False)
    parser.add_argument('--breakpoint', type=str,
                        default="./logs/model_saa_agp/Epoch32-Total_Loss0.0903.pth")

    parser.add_argument('--pretrained_use', type=bool, default=False, help="是否导入预训练模型")
    parser.add_argument('--pretrained_path', type=str, default=r"model_data/unet_voc.pth", choices=["unet_voc.pth", r"model_data/resnet50.pth"], help="预训练模型的地址")

    parser.add_argument('--test_dataset_path', type=str, default="/mnt/data/zhang_chi_new/dataset_split/KiTS/test")
    parser.add_argument('--train_npy', type=str, default="./train_set_txt/train_tumor_set.npy")
    parser.add_argument('--test_txt', type=str, default="./test_set_txt")

    parser.add_argument('--train_all_data', type=bool, default=False, help="很多CT图像是全黑的,并不包含目标,这里设置False可以只训练有目标的数据")
    parser.add_argument('--label_reverse', type=bool, default=False, help="选择是否对标签进行翻转，因为血管数据集的标签值是黑的")
    parser.add_argument('--shuffle', type=bool, default=True, help="数据集是否打乱")
    parser.add_argument('--nii_use', type=bool, default=True, help="数据集的格式是否为nii,如果是nii，则需要换另一种方式打开")
    parser.add_argument('--window_trunction', type=bool, default=True, help="是否进行窗位窗宽截断")

    parser.add_argument('--num_workers', type=int, default=4, help="")
    parser.add_argument('--cuda_use', type=bool, default=True, help="是否使用GPU训练")
    parser.add_argument('--log_dir', type=str, default="./logs/log", help="训练曲线的保存地址")
    parser.add_argument('--model_save_dir', type=str, default="./logs/model", help="训练模型的保存地址")
    parser.add_argument('--img_size', type=int, default=512, help="输入图像的尺寸")
    parser.add_argument('--input_channel', type=int, default=3, help="输入图像的通道数")
    parser.add_argument('--num_classes', type=int, default=3, help="类别=目标数量+1, 包括背景")
    # parser.add_argument('--depth', type=int, default=3, help="2.5D网络深度")
    return parser.parse_args()


if __name__ == "__main__":
    cfg = parse_args()
    log_dir = cfg.log_dir + '_' + cfg.model_name
    model_save_dir = cfg.model_save_dir + '_' + cfg.model_name
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    if not os.path.isdir(model_save_dir):
        os.makedirs(model_save_dir)
    # --------------------------------------------------------------------#
    #   建议选项：dice_loss
    #   种类少（几类）时，设置为True
    #   种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
    #   种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
    # ---------------------------------------------------------------------#
    print("Training " + cfg.model_name + " Model......")

    model = cfg.Net[cfg.model_name](num_classes=cfg.num_classes, in_channels=cfg.input_channel,
                                    ).train()  # 获取model
    # 继续训练saa-agp模型
    if cfg.breakpoint_train:
        model_dict = torch.load(cfg.breakpoint)
        model.load_state_dict(model_dict)
    #
    if cfg.pretrained_use:
        model_path = cfg.pretrained_path
        load_dict(model, model_path)

    if cfg.cuda_use:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    # 所有的数据集模式---Images/0/0.png、Labels/0/0.png
    train_lines = np.load(cfg.train_npy)
    if cfg.shuffle:
        np.random.shuffle(train_lines)

    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Interval_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#
#   GAM-batchsize-4-2; ECA-batchsize-4-4
#   VOLO-batchsize-2-2
    if True:
        training(cfg, model, train_lines, freeze=True, lr=1e-4, Init_Epoch=0, Interval_Epoch=10,
                 Batch_size=2)

    if True:
        training(cfg, model, train_lines, freeze=False, lr=4.76e-6, Init_Epoch=10, Interval_Epoch=50,
                 Batch_size=2)

    if True:
        training(cfg, model, train_lines, freeze=False, lr=1e-6, Init_Epoch=50, Interval_Epoch=100,
                 Batch_size=2)

    if True:
        training(cfg, model, train_lines, freeze=False, lr=1e-7, Init_Epoch=100, Interval_Epoch=150,
                 Batch_size=2)

    if True:
        training(cfg, model, train_lines, freeze=False, lr=1e-8, Init_Epoch=150, Interval_Epoch=200,
                 Batch_size=2)
