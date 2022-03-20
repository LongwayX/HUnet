""" -*- coding: utf-8 -*-
@ Time: 2021/10/24 15:19
@ author: Zhang Chi
@ e-mail: zhang_chi@seu.edu.cn
@ file: generate_test_list.py
@ project: My_Seg_Pytorch
"""
import os
import nibabel as nib

from main import parse_args

def write_train_txt(cfg):
    dirs = os.listdir(cfg.test_dataset_path + "/Images")
    dirs.sort(key=lambda x: int(x))
    for i, dir in enumerate(dirs):
        file = open(os.path.join(cfg.test_dataset_path, "test_set_" + str(dir) + ".txt"), "a")
        print("第{}个dir".format(i))
        dir_path = os.path.join(cfg.test_dataset_path + "/Images", dir)
        imgs = os.listdir(dir_path)
        imgs.sort(key=lambda x: int(x.split('.')[0].split('_')[1]))
        for img in imgs:
            label_path = os.path.join(cfg.test_dataset_path + "/Labels", dir, img)
            label_nii = nib.load(label_path)
            label_image = label_nii.get_fdata().astype('float32')
            if label_image.max() < 0.1:
                continue
            else:
                file.writelines(os.path.join(dir, img))
                file.writelines("\n")
        file.close()


def read_train_txt():
    cfg = parse_args()
    with open(os.path.join(cfg.test_dataset_path, "train_set.txt"), "r") as f:
        train_lines = f.readlines()
    print(train_lines)


if __name__ == '__main__':
    write_train_txt(parse_args())
    # read_train_txt()
