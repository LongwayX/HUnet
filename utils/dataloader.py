import os
from numpy.random import shuffle
import cv2
import numpy as np
from PIL import Image
import nibabel as nib
from torch.utils.data.dataset import Dataset

from utils.window_trunction import window_trunction


def letterbox_image(image, label, size):
    label = Image.fromarray(np.array(label))
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    label = label.resize((nw, nh), Image.NEAREST)
    new_label = Image.new('L', size, (0))
    new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))

    return new_image, new_label


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


class DeeplabDataset(Dataset):
    def __init__(self, cfg, train_lines, dataset_path, random_data=False, shuffle=True):
        super(DeeplabDataset, self).__init__()
        self.cfg = cfg
        self.shuffle = shuffle
        self.nii_use = cfg.nii_use
        self.window_trunction = cfg.window_trunction
        self.depth = cfg.input_channel
        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = [1, cfg.input_channel, cfg.img_size, cfg.img_size]
        self.num_classes = cfg.num_classes
        self.random_data = random_data
        self.dataset_path = dataset_path
        self.label_reverse = cfg.label_reverse

    def __len__(self):
        return self.train_batches

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.1, hue=.0, sat=1.1, val=1.1):
        image = image.convert("RGB")
        label = Image.fromarray(np.array(label))

        h, w = input_shape
        # resize image
        rand_jit1 = rand(1 - jitter, 1 + jitter)
        rand_jit2 = rand(1 - jitter, 1 + jitter)
        new_ar = w / h * rand_jit1 / rand_jit2

        scale = rand(0.5, 1.5)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)
        label = label.convert("L")

        # flip image or not
        flip = rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        # place image
        dx = int(rand(0, w - nw))
        dy = int(rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_label = Image.new('L', (w, h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255
        return image_data, label

    def __getitem__(self, index):
        if index == 0 and self.shuffle:
            shuffle(self.train_lines)
        annotation_lines = self.train_lines[index]
        names = []
        for lines in annotation_lines:
            names.append(lines.split()[0])
        # 从文件中读取图像
        if self.nii_use:
            jpgs = []
            for i in range(self.depth):
                ct_nii = nib.load(os.path.join(os.path.join(self.dataset_path, "Images"), names[i]))
                ct_image = ct_nii.get_fdata().astype('float32')
                jpg = ct_image.clip(-1024, 1024)
                jpgs.append(jpg)
                if i == (self.depth // 2 + 1):
                    label_nii = nib.load(os.path.join(os.path.join(self.dataset_path, "Labels"), names[i]))
                    label_image = label_nii.get_fdata().astype('float32')
                    png = label_image.clip(-1024, 1024)

            # 首先需要窗位窗宽截断
        if self.window_trunction:
            # jpg = window_trunction(jpg, windowWidth=200, windowCenter=50)
            # 将窗位窗宽修改为450，25
            for j in range(len(jpgs)):
                jpg = jpgs[j]
                jpg = window_trunction(jpg, windowWidth=self.cfg.window_value[0], windowCenter=self.cfg.window_value[1])
                jpg = jpg.astype('float32')
                jpgs[j] = jpg
                # jpg = Image.fromarray(jpg)  # 将numpy数组转换为PIL格式
                # png = Image.fromarray(png)
        # else:
            # jpg = Image.open(os.path.join(os.path.join(self.dataset_path, "Images"), names[i]))
            # png = Image.open(os.path.join(os.path.join(self.dataset_path, "Labels"), names[]))

        if self.random_data:
            jpg, png = self.get_random_data(jpg, png, (int(self.image_size[1]), int(self.image_size[0])))
        # else:
            # jpg, png = letterbox_image(jpg, png, (int(self.image_size[1]), int(self.image_size[0])))
        png = np.array(png,dtype='int32')
        # png[png >= self.num_classes] = self.num_classes

        if self.label_reverse:
            modify_png = np.zeros_like(png)  # 构造全零数组
            modify_png[png <= 127.5] = 1  # 这是因为血管数据集的mask是全黑的
        else:
            modify_png = png
        seg_labels = modify_png
        seg_labels = np.eye(self.num_classes + 1)[seg_labels.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.image_size[2]), int(self.image_size[3]), self.num_classes + 1))

        if not self.window_trunction:
            for k in range(len(jpgs)):
                jpgs[k] = np.transpose(np.array(jpgs[k]), [2, 0, 1]) / 1024
        else:
            for k in range(len(jpgs)):
                jpgs[k] = np.array(jpgs[k]) / 255
        jpgs = np.array(jpgs)
        jpgs = jpgs.reshape((self.image_size))
        return jpgs, modify_png, seg_labels


# DataLoader中collate_fn使用
def deeplab_dataset_collate(batch):
    images = []
    pngs = []
    seg_labels = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images = np.array(images)
    pngs = np.array(pngs)
    seg_labels = np.array(seg_labels)
    return images, pngs, seg_labels




