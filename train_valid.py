""" -*- coding: utf-8 -*-
@ Time: 2021/10/24 9:22
@ author: Zhang Chi
@ e-mail: zhang_chi@seu.edu.cn
@ file: train_valid.py
@ project: My_Seg_Pytorch
"""
import os
import argparse
import numpy as np
import torch
import cv2 as cv
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.loss import CE_Loss, Dice_loss
from utils.dataloader import DeeplabDataset, deeplab_dataset_collate
from utils.metrics import f_score
from utils.test_metric import calDice


# f-score其实就是dice，和iou类似
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def load_dict(model, model_path):
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.items() and np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')


def fit_one_epoch(cfg, model, optimizer, epoch, epoch_size, gen, Epoch):
    total_loss = 0
    total_f_score = 0

    model = model.train()
    print("--------------------------Start training--------------------------")
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            imgs, pngs, labels = batch

            with torch.no_grad():
                imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
                pngs = torch.from_numpy(pngs).type(torch.FloatTensor).long()
                labels = torch.from_numpy(labels).type(torch.FloatTensor)
                if cfg.cuda_use:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()

            optimizer.zero_grad()

            outputs = model(imgs)
            loss = CE_Loss(outputs, pngs, num_classes=cfg.num_classes)
            if cfg.dice_loss_use:
                if cfg.dice_loss_only:
                    loss = Dice_loss(outputs, labels)
                else:
                    main_dice = Dice_loss(outputs, labels)
                    loss = loss + main_dice
            #print('loss:',loss)
            with torch.no_grad():
                _f_score = f_score(outputs, labels)  # 计算f_score

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_f_score += _f_score.item()

            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'f_score': total_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f' % (total_loss / (epoch_size + 1)))
    print('Saving state, iter:', str(epoch + 1))
    torch.save(model.state_dict(),
               cfg.model_save_dir + '_' + cfg.model_name + '/Epoch%d-Total_Loss%.4f.pth' % (
                   (epoch + 1), total_loss / (epoch_size + 1)))
    print("--------------------------End training--------------------------")

    print("------------------------Start Validation------------------------")
    if not os.path.isdir("./valid_result/" + cfg.model_name):
        os.makedirs("./valid_result/" + cfg.model_name)
    file = open("./valid_result/" + cfg.model_name + "/valid_result_" + cfg.model_name + "_epoch" + str(epoch + 1) + ".txt", "a")
    file.writelines("model: " + cfg.model_name + "\n")
    test_dataset_path = cfg.test_dataset_path
    test_txt_path = os.path.join(test_dataset_path, cfg.test_txt)

    batch_size = 1
    num_workers = 1

    txts_list = os.listdir(test_txt_path)
    txts_list.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    total_kidney_dice = []
    total_tumor_dice = []
    total_test_loss = []

    total_kidney_inter = 0
    total_kidney_pred = 0
    total_kidney_gt = 0
    total_tumor_inter = 0
    total_tumor_pred = 0
    total_tumor_gt = 0

    case_kidney_dice_list = []
    case_tumor_dice_list = []
    i = 0
    for txt in txts_list:
        case_kidney_inter = 0
        case_kidney_pred = 0
        case_kidney_gt = 0
        case_tumor_inter = 0
        case_tumor_pred = 0
        case_tumor_gt = 0
        txt_path = os.path.join(test_txt_path, txt)
        file.writelines("Patient: " + txt.split('_')[-1].split('.')[0] + " Result" + "\n")
        with open(txt_path, "r") as f:
            test_kidney_lines = f.readlines()

        # 切记，切记，在测试图像分割时，random_data需要设置为False，否则就会数据增强
        test_dataset = DeeplabDataset(cfg, test_kidney_lines, dataset_path=test_dataset_path, random_data=False,
                                      shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                 drop_last=True, collate_fn=deeplab_dataset_collate)

        model = model.eval()
        progress_bar = tqdm(test_loader, total=len(test_loader))
        for index, data in enumerate(progress_bar):
            imgs, pngs, labels = data  # 图像、标签(单通道)、标签(多通道)
            old_img = imgs

            with torch.no_grad():
                imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
                pngs = torch.from_numpy(pngs).type(torch.FloatTensor).long()
                labels = torch.from_numpy(labels).type(torch.FloatTensor)
                if cfg.cuda_use:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()

                outputs = model(imgs)

                # 下面在计算loss的时候，pred(多特征图)可以与gt(单特征图)直接计算交叉熵，因为有函数；但是dice需要pred和gt(多特征图)计算
                loss = CE_Loss(outputs, pngs, num_classes=cfg.num_classes)
                if cfg.dice_loss_use:
                    main_dice = Dice_loss(outputs, labels)
                    loss = loss + main_dice

                # 获得的是包含0，1，2...的单张特征图
                output_numpy = torch.softmax(outputs.transpose(1, 2).transpose(2, 3).contiguous(), -1)
                output_numpy = output_numpy.cpu().detach().numpy()
                output_numpy = np.reshape(output_numpy,
                                          (output_numpy.shape[1], output_numpy.shape[2], output_numpy.shape[3])).argmax(
                    axis=-1)
                label_numpy = labels.cpu().detach().numpy()[:, :, :, :-1]
                label_numpy = np.reshape(label_numpy,
                                         (label_numpy.shape[1], label_numpy.shape[2], label_numpy.shape[3])).argmax(
                    axis=-1)

                # 这里计算性能指标, 输入的是网络的pred和gt
                kidney_dice, tumor_dice, tumor_flag = calDice(output_numpy, label_numpy)
                pred_kidney = (output_numpy == 1).astype(np.int16)
                pred_tumor = (output_numpy == 2).astype(np.int16)
                label_kidney = (label_numpy == 1).astype(np.int16)
                label_tumor = (label_numpy == 2).astype(np.int16)
                kidney_intersection = cv.bitwise_and(pred_kidney, label_kidney)
                tumor_intersection = cv.bitwise_and(pred_tumor, label_tumor)

                total_kidney_gt += len(label_kidney[label_kidney == 1])
                total_kidney_pred += len(pred_kidney[pred_kidney == 1])
                total_kidney_inter += len(label_kidney[kidney_intersection == 1])

                total_tumor_gt += len(label_tumor[label_tumor == 1])
                total_tumor_pred += len(pred_tumor[pred_tumor == 1])
                total_tumor_inter += len(label_tumor[tumor_intersection == 1])

                case_kidney_gt += len(label_kidney[label_kidney == 1])
                case_kidney_pred += len(pred_kidney[pred_kidney == 1])
                case_kidney_inter += len(label_kidney[kidney_intersection == 1])

                case_tumor_gt += len(label_tumor[label_tumor == 1])
                case_tumor_pred += len(pred_tumor[pred_tumor == 1])
                case_tumor_inter += len(label_tumor[tumor_intersection == 1])

                total_kidney_dice.append(kidney_dice)
                if tumor_flag:
                    total_tumor_dice.append(tumor_dice)
                total_test_loss.append(loss.cpu().detach().item())
                i += 1

            progress_bar.set_postfix(**{'kidney_dice': np.mean(total_kidney_dice),
                                        'tumor_dice': 0.0 if len(total_tumor_dice) == 0 else np.mean(total_tumor_dice),
                                        'loss': np.mean(total_test_loss)})
            file.writelines(str(index) + " --- kidney dice: " + str(
                "%.3f" % (kidney_dice)) + " --- tumor dice: " + str(
                "%.3f" % (tumor_dice)) + " --- loss: " + str(
                "%.3f" % (loss.cpu().detach().item())) + "\n")
            progress_bar.update(1)

        case_kidney_dice = 2 * case_kidney_inter / (case_kidney_gt + case_kidney_pred)
        if case_tumor_gt == 0:
            if case_tumor_pred == 0:
                case_tumor_dice = 1.0
            else:
                case_tumor_dice = 0.0
        else:
            case_tumor_dice = 2 * case_tumor_inter / (case_tumor_gt + case_tumor_pred)
        case_kidney_dice_list.append(case_kidney_dice)
        case_tumor_dice_list.append(case_tumor_dice)
        file.writelines("Case " + txt.split('_')[-1].split('.')[0] + " kidney dice: " + str(case_kidney_dice) + "\n")
        file.writelines("Case " + txt.split('_')[-1].split('.')[0] + " tumor dice: " + str(case_tumor_dice) + "\n")
        print("Case " + txt.split('_')[-1].split('.')[0] + " kidney dice: " + str(case_kidney_dice))
        print("Case " + txt.split('_')[-1].split('.')[0] + " tumor dice: " + str(case_tumor_dice))

    global_kidney_dice = 2 * total_kidney_inter / (total_kidney_gt + total_kidney_pred)
    global_tumor_dice = 2 * total_tumor_inter / (total_tumor_gt + total_tumor_pred)

    file.writelines("Last Result: " + "\n")
    file.writelines("Slice mean kidney dice: " + str(np.mean(total_kidney_dice)) + "\n")
    file.writelines("Slice mean tumor dice: " + str(np.mean(total_tumor_dice)) + "\n")
    file.writelines("Slice mean loss: " + str(np.mean(total_test_loss)) + "\n")
    file.writelines("Global kidney dice: " + str(global_kidney_dice) + "\n")
    file.writelines("Global tumor dice: " + str(global_tumor_dice) + "\n")
    file.writelines("PerCase kidney dice: " + str(np.mean(case_kidney_dice_list)) + "\n")
    file.writelines("PerCase tumor dice: " + str(np.mean(case_tumor_dice_list)) + "\n")
    file.writelines("Case kidney dice list: " + str(case_kidney_dice_list) + "\n")
    file.writelines("Case tumor dice list: " + str(case_tumor_dice_list) + "\n")
    file.close()
    print("------------------------End Validation------------------------")


def training(cfg, model, train_lines, freeze=False, lr=1e-4, Init_Epoch=0, Interval_Epoch=50, Batch_size=2):
    optimizer = optim.Adam(model.parameters(), lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

    train_dataset = DeeplabDataset(cfg, train_lines, dataset_path=cfg.dataset_path, random_data=False, shuffle=True)
    gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=cfg.num_workers, pin_memory=True,
                     drop_last=True, collate_fn=deeplab_dataset_collate)

    epoch_size = len(train_lines) // Batch_size

    if epoch_size == 0:
        raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True
#    if freeze:
##        for param in model.conv1.parameters():
##            param.requires_grad = False
#        for param in model.bn1.parameters():
#            param.requires_grad = False
#        for param in model.relu.parameters():
#            param.requires_grad = False
#        for param in model.maxpool.parameters():
#            param.requires_grad = False
#        for param in model.layer1.parameters():
#            param.requires_grad = False
#        for param in model.layer2.parameters():
#            param.requires_grad = False
#        for param in model.layer3.parameters():
#            param.requires_grad = False
#        for param in model.layer4.parameters():
#            param.requires_grad = False
#    else:
##        for param in model.conv1.parameters():
##            param.requires_grad = True
#        for param in model.bn1.parameters():
#            param.requires_grad = True
#        for param in model.relu.parameters():
#            param.requires_grad = True
#        for param in model.maxpool.parameters():
#            param.requires_grad = True
#        for param in model.layer1.parameters():
#            param.requires_grad = True
#        for param in model.layer2.parameters():
#            param.requires_grad = True
#        for param in model.layer3.parameters():
#            param.requires_grad = True
#        for param in model.layer4.parameters():
#            param.requires_grad = True

    for epoch in range(Init_Epoch, Interval_Epoch):
        fit_one_epoch(cfg, model, optimizer, epoch, epoch_size, gen, Interval_Epoch)
        lr_scheduler.step()
