import os
import cv2
import csv
import glob
import torch
import kornia
import random
import torchvision
import skimage.io
import numpy as np
import torch.nn as nn
from osgeo import gdal
from PIL import Image
from torch.utils import data as data
from torchvision.transforms import Normalize, Compose, Lambda, Resize, InterpolationMode, ToTensor
from torchvision.transforms import functional as trans_fn
from basicsr.utils.registry import DATASET_REGISTRY

totensor = torchvision.transforms.ToTensor()


@DATASET_REGISTRY.register()
class OLI2MSIDataset(data.Dataset):
    """
    Args:
        opt (dict): Config for train datasets. It contains the following keys:
    """

    def __init__(self, opt):
        super(OLI2MSIDataset, self).__init__()
        self.opt = opt

        self.split = opt['phase']
        self.data_root = opt['data_root']

        self.use_3d = opt['use_3d'] if 'use_3d' in opt else False

        if self.split == 'train':
            hr_fps = glob.glob(self.data_root + 'train_hr/*.TIF')
            lr_fps = [hr_fp.replace('train_hr', 'train_lr') for hr_fp in hr_fps]
        else:
            hr_fps = hr_fps = glob.glob(self.data_root + 'test_hr/*.TIF')
            lr_fps = [hr_fp.replace('test_hr', 'test_lr') for hr_fp in hr_fps]
        
        self.datapoints = []
        for i,hr_fp in enumerate(hr_fps):
            self.datapoints.append([hr_fp, lr_fps[i]])

        self.data_len = len(self.datapoints)
        print("Loaded ", self.data_len, " data pairs for split ", self.split)

    def __getitem__(self, index):
        hr_path, lr_path = self.datapoints[index]

        hr_ds = gdal.Open(hr_path)
        hr_arr = np.array(hr_ds.ReadAsArray())
        hr_tensor = torch.tensor(hr_arr).float()

        lr_ds = gdal.Open(lr_path)
        lr_arr = np.array(lr_ds.ReadAsArray())
        lr_tensor = torch.tensor(lr_arr).float()

        if self.use_3d:
            lr_tensor = lr_tensor.unsqueeze(0)

        img_HR = hr_tensor
        img_LR = lr_tensor

        return {'gt': img_HR, 'lq': img_LR, 'Index': index}

    def __len__(self):
        return self.data_len
