""" -*- coding: utf-8 -*-
@ Time: 2021/5/19 20:52
@ author: Zhang Chi
@ e-mail: zhang_chi@seu.edu.cn
@ file: main.py
@ project: scale_attention_preprocessing
"""
import argparse
from split_nii2slice import split_nii


def arg_setting():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, required=False, default='LiTS', help='name of database',
                        choices=['LiTS', 'KiTS', 'MSD', 'Synapse', 'covid_19'])
    parser.add_argument('--is_crop', type=bool, required=False, default=False, help='need crop or not')
    parser.add_argument('--is_save', type=bool, required=False, default=True, help='need save as slice or not')

    # parser.add_argument('--root_path', type=str,
    #                     default=r"E:\1-code\seu_project_paper\scale_attention_preprocessing\dataset")
    # parser.add_argument('--result_path', type=str,
    #                     default=r"E:\1-code\seu_project_paper\scale_attention_preprocessing\split_result")
    # parser.add_argument('--root_path', type=str,
    #                     default=r"F:\dataset\LiTS2017\Training_Batch")
    # parser.add_argument('--result_path', type=str,
    #                     default=r"G:\dataset\LiTS_split")
    parser.add_argument('--root_path', type=str,
                        default="../../dataset")
    parser.add_argument('--result_path', type=str,
                        default="../../dataset_split")

    # parser.add_argument('--root_path', type=str,
    #                     default=r"F:\dataset\LiTS2017\Testing_Batch", choices=[r"F:\dataset\LiTS2017\Testing_Batch", r"F:\dataset\KiTS\test"])
    # parser.add_argument('--result_path', type=str,
    #                     default=r"F:\dataset\LiTS2017\LiTS_test_split", choices=[r"F:\dataset\LiTS2017\LiTS_test_split", r"G:\dataset\KiTS_split"])

    return parser.parse_args()


if __name__ == '__main__':
    setting = arg_setting()
    split_nii(setting)
