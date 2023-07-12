# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 15:25:33 2022

@author: Lenovo
"""

import sys
import json
import pickle
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import os
from PIL import Image
from torchvision import transforms
import random
from fund_detect.pre_deal import augment, reshape, test_get_boxes
from fund_detect.src.utils import get_center
from swtf_tf_sgm.patch_process import patch2global, global2patch
from skimage import morphology

# -----------------------------------------------use for training-------------------------------------------------------
data_transform = {
    "train": transforms.Compose([transforms.Resize([224, 224]),
                                 transforms.RandomHorizontalFlip(p=0.5),
                                 transforms.RandomVerticalFlip(p=0.5),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize([224, 224]),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.BCEWithLogitsLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.sigmoid(pred).gt(0.5).int()
        accu_num += torch.eq(pred_classes.squeeze(1), labels.to(device)).sum()

        labels = labels.float()
        loss = loss_function(pred, labels.unsqueeze(-1).to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.BCEWithLogitsLoss()
    model.eval()
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.sigmoid(pred).gt(0.5).int()
        accu_num += torch.eq(pred_classes.squeeze(1), labels.to(device)).sum()

        labels = labels.float()
        loss = loss_function(pred, labels.unsqueeze(-1).to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    # 记录optimize要训练的权重参数
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    # 记录对应的权重名称
    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


# -----------------------------------------------use for predicting-----------------------------------------------------
def quality_check(device, img_path, data_transform, qcmodel):
    ori = Image.open(img_path)
    img = data_transform(ori)
    img = torch.unsqueeze(img, dim=0)
    qcmodel.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(qcmodel(img.to(device))).cpu()
        pred_class = torch.sigmoid(output).gt(0.5).int()
    if pred_class == 0:
        return True
    else:
        return False


def get_fovea_point(image_path, fovea_model, device, idx=1):
    image = Image.open(image_path)
    image = augment(image)
    image, scale_factor = reshape(image)
    boxes = [0, 0, 0, 0]
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    # create labels
    labels = torch.tensor([1], dtype=torch.int64)
    image_id = torch.tensor([idx])
    iscrowd = torch.zeros((1), dtype=torch.int64)
    target = {}
    target["boxes"] = boxes
    target["labels"] = labels
    target["image_id"] = image_id
    target["area"] = torch.tensor([0], dtype=torch.int64)
    target["iscrowd"] = iscrowd
    image = image.to(device)
    img, Fovea_predicted_box = test_get_boxes(fovea_model, image, target, device=device, threshold=0.08, img_idx=idx)
    predicted_center = get_center(Fovea_predicted_box)
    predicted_center.insert(0, idx)
    predicted_center[1] = predicted_center[1] / scale_factor[0]
    predicted_center[2] = predicted_center[2] / scale_factor[1]

    f_resh = (predicted_center[1], predicted_center[2])
    return f_resh


def create_circular_mask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask


def get_grade1_cut(img, f_p):  # cut original image to grade1 roi
    h, w = img.shape[:2]
    r0 = h / 25.4
    r = r0 * 6
    image_part = img.copy()
    cv2.circle(image_part, (int(f_p[0]), int(f_p[1])), int(r), (0, 0, 0), thickness=-1)
    image_part = image_part[:, int(w / 2) - int(h / 2):int(w / 2) + int(h / 2)]
    return image_part


def get_grade2_cut(img, f_p):  # cut original image to grade2 roi
    h, w = img.shape[:2]
    r0 = h / 25.4
    r1 = r0 * 3
    r2 = r0 * 6
    image_part = img.copy()
    cv2.circle(image_part, (int(f_p[0]), int(f_p[1])), int(r1), (0, 0, 0), thickness=-1)
    center = (int(f_p[0]), int(f_p[1]))
    mask = create_circular_mask(h, w, center=center, radius=int(r2))
    masked_img = image_part.copy()
    masked_img[~mask] = 0
    masked_img = masked_img[center[1] - int(r2):center[1] + int(r2), center[0] - int(r2):center[0] + int(r2)]
    return masked_img


def get_grade3_cut(img, f_p):  # cut original image to grade3 roi
    h, w = img.shape[:2]
    r0 = h / 25.4
    r1 = r0 * 3
    image_part = img.copy()
    cv2.circle(image_part, (int(f_p[0]), int(f_p[1])), int(r0), (0, 0, 0), thickness=-1)
    center = (int(f_p[0]), int(f_p[1]))
    mask = create_circular_mask(h, w, center=center, radius=r1)
    masked_img = image_part.copy()
    masked_img[~mask] = 0
    masked_img = masked_img[center[1] - int(r1):center[1] + int(r1), center[0] - int(r1):center[0] + int(r1)]
    return masked_img


def get_grade4_cut(img, f_p):  # cut original image to grade4 roi
    h, w = img.shape[:2]
    r0 = h / 25.4
    image_part = img.copy()
    center = (int(f_p[0]), int(f_p[1]))
    mask = create_circular_mask(h, w, center=center, radius=r0)
    masked_img = image_part.copy()
    masked_img[~mask] = 0
    masked_img = masked_img[center[1] - int(r0):center[1] + int(r0), center[0] - int(r0):center[0] + int(r0)]
    return masked_img


def judge_grade(device, img_path, data_transform, fovea, grade, model, weights_root):
    weights_path = os.path.join(weights_root, 'grade' + str(grade) + '_part.pth')

    ori = np.array(Image.open(img_path))
    if grade == 0:
        img = ori
    elif grade == 1:
        img = get_grade1_cut(img=ori, f_p=fovea)
    elif grade == 2:
        img = get_grade2_cut(img=ori, f_p=fovea)
    elif grade == 3:
        img = get_grade3_cut(img=ori, f_p=fovea)
    else:
        img = get_grade4_cut(img=ori, f_p=fovea)

    img = data_transform(Image.fromarray(np.uint8(img)))
    img = torch.unsqueeze(img, dim=0)

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        pred_class = torch.sigmoid(output).gt(0.5).int()
    if pred_class == 1:
        result = True
    else:
        result = False
    return result, torch.sigmoid(output).numpy()


def seg_ft(image_path, net, min_size=1500,
           transform=transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
           patch_size=224,):
    test_image_PIL = Image.open(image_path).convert('RGB')
    patches, coordinates, templates, sizes, ratios = global2patch([test_image_PIL], (patch_size, patch_size))
    net.eval()
    with torch.no_grad():
        predict_patch_list = []
        for patch in patches[0]:
            patch_tensor = transform(patch)
            out_patch = net(torch.unsqueeze(patch_tensor, dim=0))
            predict_patch_list.append(out_patch)
    predict_patch_tensor = torch.concat(predict_patch_list, dim=0)
    results = patch2global(predict_patch_tensor, n_class=2, sizes=sizes, coordinates=coordinates,
                           p_size=(patch_size, patch_size))
    _segment_image_mask_save_sofmax = torch.softmax(torch.tensor(results[0]), dim=0)
    _segment_image_mask_save_max = np.argmax(torch.detach(_segment_image_mask_save_sofmax).numpy(), axis=0)
    bool_segment_image_mask_save_max = np.array(_segment_image_mask_save_max, dtype=bool)
    dst = morphology.remove_small_objects(bool_segment_image_mask_save_max, min_size=min_size, connectivity=1)

    return np.count_nonzero(dst == True) / (np.count_nonzero(dst == False) + np.count_nonzero(dst == True))


# -----------------------------------------------some tools ------------------------------------------------------------
def get_all_labels(species, img_path):
    species_to_id = dict((c, i) for i, c in enumerate(species))
    all_labels = []
    # 对所有图片路径进行迭代
    for img in img_path:
        # 区分出每个img，应该属于什么类别
        for i, c in enumerate(species):
            if c in img:
                all_labels.append(i)
    return all_labels, species_to_id


def run_circle(img_dir, f_resh, str_dir):
    img = cv2.imread(img_dir)
    h, w = img.shape[:2]
    r0 = h / 25.4
    r1 = 3 * r0
    r2 = 6 * r0
    # thickness是根据中间的线条在两边进行延伸做线条粗细的
    cv2.circle(img, (int(f_resh[1]), int(f_resh[0])), int(r0), (255, 255, 255), thickness=5)
    cv2.circle(img, (int(f_resh[1]), int(f_resh[0])), int(r1), (255, 255, 255), thickness=5)
    cv2.circle(img, (int(f_resh[1]), int(f_resh[0])), int(r2), (255, 255, 255), thickness=5)
    cv2.imwrite(str_dir, img)
