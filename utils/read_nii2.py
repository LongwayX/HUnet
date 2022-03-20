""" -*- coding: utf-8 -*-
@ Time: 2021/10/22 14:46
@ author: Zhang Chi
@ e-mail: zhang_chi@seu.edu.cn
@ file: temp.py
@ project: My_Seg_Pytorch
"""
import nibabel as nib
from utils.window_trunction import window_trunction

if __name__ == '__main__':
    # LiTS和KiTS数据集不一样
    # LiTS是(Height, Width, Depth)
    # KiTS是(Depth, Height, Width)
    ct_nii = nib.load("../raw_nii_data/Images/ct_300.nii")
    ct_image = ct_nii.get_fdata().astype('float32')

    label_nii = nib.load("../raw_nii_data/Labels/file_500.nii")
    label_image = label_nii.get_fdata().astype('float32')

    ct_image = ct_image.clip(-1024, 1024)
    label_image = label_image.clip(-1024, 1024)

    # 首先需要窗位窗宽截断
    if True:
        ct_image = window_trunction(ct_image, windowWidth=200, windowCenter=50)
        ct_image = ct_image.astype('float32')

    pass



