#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 15, 10

import torch
import torch.nn as nn
from torchvision import transforms

from datasets import get_transforms, get_dataloader, get_dataset_info
from models import get_net

device = torch.device('cuda:2')

root = '/home/iip/Vanilla_OOD_Detection'
data_root = '/home/iip/datasets'
tmp_root = root + '/tmp'
train_set = 'cifar10'
num_classes = len(get_dataset_info(train_set, 'classes'))
mean, std = get_dataset_info(train_set, 'mean_and_std')
mean = torch.tensor(mean).view(3, 1, 1).to(device)
std = torch.tensor(std).view(3, 1, 1).to(device)

arch = 'resnet50_vae'
network_path = root + '/outputs/wp-resnet50_vae-cifar10-reconstruct-lambda_kld_0.00256-lambda_reconstruct_1.0/' + 'best.pth'

# ---------- init dataloader ----------
data_transform = get_transforms(train_set, 'test')
data_loader = get_dataloader(data_root, train_set, 'train', data_transform, 64, False, 8)

# ---------- init network ----------
net = get_net(arch, num_classes)

# ---------- load trained parameters ----------
if os.path.isfile(network_path):
    network_params = torch.load(network_path)
    net.load_state_dict(network_params['state_dict'])
else:
    raise RuntimeError("---> no ckpt found at '{}'".format(network_path))

net.to(device)

# ---------- reconstruct ----------
origin_recons_pairs = []
for data in data_loader:
    if type(data) == list:
        # labeled
        img_batch, target_batch = data
    else:
        # unlabeled
        img_batch = data
    
    img_batch = img_batch.to(device)
    pred_batch, (reconstruction_batch, _, _) = net(img_batch)
    
    for img, reconstruction in zip(img_batch, reconstruction_batch):
        original_img = img.detach() * std + mean
        reconstruction_img = reconstruction.detach() * std + mean
        origin_recons_pairs.append((original_img, reconstruction_img))

# ---------- save to local ----------
toPIL = transforms.ToPILImage()
for idx, (original, reconstruction) in enumerate(origin_recons_pairs):
    if idx > 35:
        exit()
    original_pic = toPIL(original.cpu())
    reconstruction_pic = toPIL(reconstruction.cpu())
    original_pic.save(os.path.join(tmp_root, 'o-' + str(idx) +'.jpg'))
    reconstruction_pic.save(os.path.join(tmp_root, 'r-' + str(idx) + '.jpg'))