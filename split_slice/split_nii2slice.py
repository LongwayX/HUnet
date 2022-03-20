""" -*- coding: utf-8 -*-
@ Time: 2021/5/19 20:50
@ author: Zhang Chi
@ e-mail: zhang_chi@seu.edu.cn
@ file: split_nii2slice.py
@ project: scale_attention_preprocessing
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import zipfile
import nibabel as nib
import argparse
# import pydicom
import torch
from skimage import exposure
from scipy.ndimage import binary_fill_holes
from nibabel.viewers import OrthoSlicer3D
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
# from utils import unpack, unzip_nii, transform_ctdata
import time


def split_nii(setting):
    """
    预处理函数
    """
    time_start = time.time()

    root = os.path.join(setting.root_path, setting.data_name)
    data_path = os.path.join(root, "data")
    label_path = os.path.join(root, "label")
    result_data_path = os.path.join(setting.result_path, setting.data_name, "data")
    result_label_path = os.path.join(setting.result_path, setting.data_name, "label")

    # 这里取得不同数据集的窗宽
    if setting.data_name == 'LiTS':
        windowWidth, windowCenter = [200, 50]
        # if setting.need_unpack:
        #     unpack()
        #     unzip_nii()
    elif setting.data_name == 'MSD':
        windowWidth, windowCenter = [2000, -450]
    elif setting.data_name == 'covid-19':
        windowWidth, windowCenter = [2000, -450]
    elif setting.data_name == 'KiTS':
        windowWidth, windowCenter = [300, 45]
    elif setting.data_name == 'Synapse':  # 暂不明确里面都是那些器官
        windowWidth, windowCenter = [0, 0]
    else:
        windowWidth, windowCenter = [0, 0]
    print(' Preprocessing with data_name "{0}"\n'.format(setting.data_name),
          'windowWidth is {0}, windowCenter is {1}\n'.format(str(windowWidth), str(windowCenter)),
          'root_dir is "{0}",\n result_dir is "{1}",\n'.format(setting.root_path, setting.result_path),
          'need save =={}'.format(setting.is_save))
    contain_object_list = []
    if setting.is_save:  # 存储转换后切片结果

        for _, _, files in os.walk(data_path):
            files.sort(key=lambda x: int(x.split("volume-")[-1].split(".")[0]))
            for i, file in enumerate(files):
                all_data = nib.load(os.path.join(data_path, file))
                img_affine = all_data.affine
                img = all_data.get_fdata().astype('float32')
                if setting.data_name != 'KiTS':
                    num_of_slice = all_data.shape[2]
                    print(file, " ", num_of_slice, end="")

                    for count in range(num_of_slice):
                        if not os.path.isdir(os.path.join(result_data_path, str(i))):
                            os.makedirs(os.path.join(result_data_path, str(i)))
                        nib.Nifti1Image(img[:, :, count], img_affine).to_filename(
                            os.path.join(result_data_path, str(i) + '/file_{}.nii').format(count))
                else:
                    num_of_slice = all_data.shape[0]
                    print(file, " ", num_of_slice, end="")

                    for count in range(num_of_slice):
                        if not os.path.isdir(os.path.join(result_data_path, str(i))):
                            os.makedirs(os.path.join(result_data_path, str(i)))
                        nib.Nifti1Image(img[count, :, :], img_affine).to_filename(
                            os.path.join(result_data_path, str(i) + '/file_{}.nii').format(count))
                print("Have finshed {} patients in data.".format(i))

        for _, _, files in os.walk(label_path):
            files.sort(key=lambda x: int(x.split("segmentation-")[1].split(".")[0]))
            for i, file in enumerate(files):
                all_data = nib.load(os.path.join(label_path, file))
                img_affine = all_data.affine
                img = all_data.get_fdata().astype('float32')
                if setting.data_name != 'KiTS':
                    num_of_slice = all_data.shape[2]
                    print(file, " ", num_of_slice, end="")

                    for count in range(num_of_slice):
                        if not os.path.isdir(os.path.join(result_label_path, str(i))):
                            os.makedirs(os.path.join(result_label_path, str(i)))
                        nib.Nifti1Image(img[:, :, count], img_affine).to_filename(
                            os.path.join(result_label_path, str(i) + '/file_{}.nii').format(count))
                        if img[:, :, count].max() < 0.5:
                            contain_object_list.append(0)
                        else:
                            contain_object_list.append(1)
                else:
                    num_of_slice = all_data.shape[0]
                    print(file, " ", num_of_slice, end="")

                    for count in range(num_of_slice):
                        if not os.path.isdir(os.path.join(result_label_path, str(i))):
                            os.makedirs(os.path.join(result_label_path, str(i)))
                        nib.Nifti1Image(img[count, :, :], img_affine).to_filename(
                            os.path.join(result_label_path, str(i) + '/file_{}.nii').format(count))
                        if img[count, :, :].max() < 0.5:
                            contain_object_list.append(0)
                        else:
                            contain_object_list.append(1)

                print("Have finshed {} patients in label.".format(i))

    print("Using {} time!".format(time.time() - time_start))
    np.save("contain_object_list.npy", np.array(contain_object_list))





