""" -*- coding: utf-8 -*-
@ Time: 2021/5/15 17:04
@ author: Zhang Chi
@ e-mail: zhang_chi@seu.edu.cn
@ file: dataloader.py
@ project: Scale_Attention
"""
import os
import torch
import numpy as np
import nibabel as nib
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from utils import *


class DataFeeder(Dataset):
    """
    若不保存中间处理过程 加载时直接用transform = transform_cdata
    若已保存中间处理过程 加载时transform = None即可 其他部分应该不大用修改 -.- 我个人这么觉得
    """

    def __init__(self, root_dir, data_name, contain_object_list, select_data=True, windowWidth=200, windowCenter=50, img_height=512, img_width=512,
                 transform=window_trunction, crop=crop_img, need_transform=False, need_crop=False, need_normal=False, train_tumor=False, balance_train=False):
        super(DataFeeder).__init__()
        self.data_name = data_name
        self.img_height = img_height
        self.img_width = img_width

        self.data_dir = os.path.join(root_dir, self.data_name, 'train', 'data')  # 不同的数据集的data路径
        self.label_dir = os.path.join(root_dir, self.data_name, 'train', 'label')

        self.windowWidth = windowWidth
        self.windowCenter = windowCenter

        self.need_transform = need_transform
        self.need_crop = need_crop
        self.need_normal = need_normal

        self.transform = transform
        self.crop = crop

        self.contain_object = contain_object_list
        self.select_data = select_data

        self.train_tumor = train_tumor
        self.balance_train = balance_train

        self.data_files = []
        self.label_files = []

        i = 0
        data_dirs = os.listdir(self.data_dir)
        data_dirs.sort(key=lambda x: int(x))
        for sub_dir in data_dirs:
            sub_dir_path = os.path.join(self.data_dir, sub_dir)
            for file in os.listdir(sub_dir_path):
                if self.balance_train:
                    if self.contain_object[i] == 2:
                        file_path = os.path.join(sub_dir_path, file)
                        self.data_files.append(file_path)
                    elif self.contain_object[i] != 2:
                        if i % 3 == 0:
                            file_path = os.path.join(sub_dir_path, file)
                            self.data_files.append(file_path)
                        else:
                            pass
                else:
                    if self.select_data and self.train_tumor and self.contain_object[i] == 2:
                        file_path = os.path.join(sub_dir_path, file)
                        self.data_files.append(file_path)
                    elif self.select_data and not self.train_tumor and self.contain_object[i] == 1:
                        file_path = os.path.join(sub_dir_path, file)
                        self.data_files.append(file_path)
                    elif not self.select_data:
                        file_path = os.path.join(sub_dir_path, file)
                        self.data_files.append(file_path)
                i += 1

        i = 0
        label_dirs = os.listdir(self.label_dir)
        label_dirs.sort(key=lambda x: int(x))
        for sub_dir in label_dirs:
            sub_dir_path = os.path.join(self.label_dir, sub_dir)
            for file in os.listdir(sub_dir_path):
                if self.balance_train:
                    if self.contain_object[i] == 2:
                        file_path = os.path.join(sub_dir_path, file)
                        self.label_files.append(file_path)
                    elif self.contain_object[i] != 2:
                        if i % 3 == 0:
                            file_path = os.path.join(sub_dir_path, file)
                            self.label_files.append(file_path)
                        else:
                            pass
                else:
                    if self.select_data and self.train_tumor and self.contain_object[i] == 2:
                        file_path = os.path.join(sub_dir_path, file)
                        self.label_files.append(file_path)
                    elif self.select_data and not self.train_tumor and self.contain_object[i] == 1:
                        file_path = os.path.join(sub_dir_path, file)
                        self.label_files.append(file_path)
                    elif not self.select_data:
                        file_path = os.path.join(sub_dir_path, file)
                        self.label_files.append(file_path)
                i += 1

        if len(self.data_files) != len(self.label_files):
            raise ValueError('The number of data should be equal to the number of label')

    def __len__(self):
        if len(self.data_files) == 0:
            raise ValueError('The number of data_filesnames can not be zero')
        return len(self.data_files)

    def __getitem__(self, index):
        ct_nii = nib.load(self.data_files[index])
        ct_image = ct_nii.get_fdata().astype('float32')

        label_nii = nib.load(self.label_files[index])
        label_image = label_nii.get_fdata().astype('float32')

        ct_image = ct_image.clip(-1024, 1024)
        label_image = label_image.clip(-1024, 1024)

        if self.train_tumor:
            label_image[label_image < 1.5] = 0
            label_image[label_image > 1.5] = 1
        else:
            label_image[label_image > 0] = 1

        # 首先需要窗位窗宽截断
        if self.need_transform:
            ct_image = self.transform(ct_image, self.windowWidth, self.windowCenter)
            ct_image = ct_image.astype('float32')

        if self.need_crop:
            ct_image = self.crop(ct_image)
            label_image = self.crop(label_image)

        # 其次reshape图片
        """这里应该先读取图像的空间分辨率，然后按照空间分辨率的尺寸来reshape"""
        ct_image = np.reshape(ct_image, (self.img_height, self.img_width, 1))
        label_image = np.reshape(label_image, (self.img_height, self.img_width, 1))
        # 然后需要归一化
        if self.need_normal:
            ct_image /= 255.0
            label_image /= 255.0
        else:
            ct_image /= 1024

        # 最后再ToTensor
        image_sample = ToTensor()(ct_image).float()
        label_sample = ToTensor()(label_image).float()
        return image_sample, label_sample
