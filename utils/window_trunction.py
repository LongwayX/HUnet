""" -*- coding: utf-8 -*-
@ Time: 2021/10/22 15:01
@ author: Zhang Chi
@ e-mail: zhang_chi@seu.edu.cn
@ file: window_trunction.py
@ project: My_Seg_Pytorch
"""


def window_trunction(image, windowWidth, windowCenter, normal=False):
    """
    注意，这个函数的self.image一定得是float类型的，否则就无效！
    return: trucated image according to window center and window width
    """
    minWindow = float(windowCenter) - 0.5 * float(windowWidth)  # 窗宽最低值
    newimg = (image - minWindow) / float(windowWidth)  # （图像减阈值）
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    if not normal:
        newimg = (newimg * 255).astype('uint8')
    return newimg
