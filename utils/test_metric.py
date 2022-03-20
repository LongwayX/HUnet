""" -*- coding: utf-8 -*-
@ Time: 2021/10/23 9:33
@ author: Zhang Chi
@ e-mail: zhang_chi@seu.edu.cn
@ file: test_metric.py
@ project: My_Seg_Pytorch
"""
import numpy as np
import cv2 as cv
import os

from utils.surface_distance.metrics import *


def calPrecision(predict, label):
    """
    计算Precision，predict中，正值的比例
    """
    label_positive = label == 255
    predict_positive = predict == 255

    a = label_positive.astype(np.int16)
    b = predict_positive.astype(np.int16)
    all_positive = cv.bitwise_and(a, b)

    if len(predict[predict_positive]) == 0:
        if len(label[label_positive == 1]) == 0:
            Precision = 1.0
        else:
            Precision = 0.0
    else:
        Precision = len(label[all_positive == 1]) / len(predict[predict_positive])

    return Precision


def calAccuracy(predict, label):
    """
    计算Accuracy，正确点数量
    """
    row, col = predict.shape  # 矩阵的行与列
    equal = len(label[predict == label])
    Accuracy = equal / (row * col)

    return Accuracy


def calRecall(predict, label):
    """
    计算召回率Recall，label中正确占比
    """
    label_positive = label == 255
    predict_positive = predict == 255

    a = label_positive.astype(np.int16)
    b = predict_positive.astype(np.int16)
    all_positive = cv.bitwise_and(a, b)

    if len(label[label_positive]) == 0:
        if len(predict[predict_positive]) == 0:
            Recall = 1.0
        else:
            Recall = 0.0
    else:
        Recall = len(label[all_positive == 1]) / len(label[label_positive])

    return Recall


# 计算DICE系数
def calDice(predict, label):
    pred_liver = (predict == 1).astype(np.int16)
    pred_tumor = (predict == 2).astype(np.int16)
    label_liver = (label == 1).astype(np.int16)
    label_tumor = (label == 2).astype(np.int16)

    liver_intersection = cv.bitwise_and(pred_liver, label_liver)
    tumor_intersection = cv.bitwise_and(pred_tumor, label_tumor)

    tumor_flag = True

    if len(label_liver[label_liver == 1]) == 0:
        if len(pred_liver[pred_liver == 1]) == 0:
            liver_dice = 1.0
        else:
            liver_dice = 0.0
    else:
        a = len(label_liver[liver_intersection == 1])
        b = len(label_liver[label_liver == 1])
        c = len(pred_liver[pred_liver == 1])
        x = 2*a/(b+c)
        liver_dice = 2 * len(label_liver[liver_intersection == 1]) / (
            len(label_liver[label_liver == 1]) + len(pred_liver[pred_liver == 1]))

    if len(label_tumor[label_tumor == 1]) == 0:
        tumor_flag = False
        if len(pred_tumor[pred_tumor == 1]) == 0:
            tumor_dice = 1.0
        else:
            tumor_dice = 0.0
    else:
        a = len(label_tumor[tumor_intersection == 1])
        b = len(label_tumor[label_tumor == 1])
        c = len(pred_tumor[pred_tumor == 1])
        x = 2*a/(b+c)
        tumor_dice = 2 * len(label_tumor[tumor_intersection == 1]) / (
            len(label_tumor[label_tumor == 1]) + len(pred_tumor[pred_tumor == 1]))
    return liver_dice, tumor_dice, tumor_flag


def calJaccard(predict, label):
    """
    Jaccard系数，A∩B / A + B - A∩B
    """
    label_positive = label == 255
    predict_positive = predict == 255

    a = label_positive.astype(np.int16)
    b = predict_positive.astype(np.int16)
    all_positive = cv.bitwise_and(a, b)

    A = len(label[label_positive])
    B = len(predict[predict_positive])
    A_B = len(label[all_positive == 1])

    if A == 0:
        if B == 0:
            Jaccard = 1.0
        else:
            Jaccard = 0.0
    else:
        Jaccard = A_B / (A + B - A_B)

    return Jaccard


def calFscore(predict, label):
    """
    F-measure or balanced F-score(Recall and Precision)
    """
    recall = calRecall(predict, label)
    precision = calPrecision(predict, label)
    # 这里应该将accuracy改成Precision
    if recall + precision == 0:
        Fscore = 0
    else:
        Fscore = 2 * recall * precision / (recall + precision)

    return Fscore


def cal_ASSD(predict, label):
    """
    Average Symmetric Surface Distance (ASSD)
    平均表面距离
    满分：(0.0, 0.0)
    :param predict:
    :param label:gt(ground truth)
    :return:
    """
    predict = predict.astype(np.bool)
    label = label.astype(np.bool)
    surface_distances = compute_surface_distances(
        label, predict, spacing_mm=(1.0, 1.0))
    avg_surf_dist = compute_average_surface_distance(surface_distances)
    return avg_surf_dist


def cal_hausdorff(predict, label):
    """
     豪斯多夫距离
     满分：0.0
    :param predict:
    :param label:
    :return:
    """
    predict = predict.astype(np.bool)
    label = label.astype(np.bool)
    surface_distances = compute_surface_distances(
        label, predict, spacing_mm=(1.0, 1.0))
    hd_dist_95 = compute_robust_hausdorff(surface_distances, 95)
    return hd_dist_95


def cal_surface_overlap(predict, label):
    """
    Surface overlap
    表面重叠度
    满分：(1.0, 1.0)
    :param predict:
    :param label:
    :return:
    """
    predict = predict.astype(np.bool)
    label = label.astype(np.bool)
    surface_distances = compute_surface_distances(
        label, predict, spacing_mm=(1.0, 1.0))
    surface_overlap = compute_surface_overlap_at_tolerance(surface_distances, 1)
    return surface_overlap


def cal_RVD(predict, label):
    """
    计算RVD系数
    """
    label_positive = label == 255
    predict_positive = predict == 255

    A = len(predict[predict_positive])
    B = len(label[label_positive])

    if B == 0:
        if A == 0:
            RVD = 0.0
        else:
            RVD = 0.0  # 在标签为全0的情况下，若仍然有预测，这里RVD设置为0.0(理论上此时的RVD为无穷，所以抛弃)
    else:
        RVD = A / B - 1.0

    return RVD
