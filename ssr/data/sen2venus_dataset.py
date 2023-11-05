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
class Sen2VenusDataset(data.Dataset):
    """
    Args:
        opt (dict): Config for train datasets. It contains the following keys:
    """

    def __init__(self, opt):
        super(Sen2VenusDataset, self).__init__()
        self.opt = opt

        self.split = opt['phase']
        self.data_root = opt['data_root']

        hr_fps = glob.glob(self.data_root + '**/*_05m_b2b3b4b8.pt')
        print("Number of hr patch fps:", len(hr_fps))
        lr_fps = [hr.replace('05m', '10m') for hr in hr_fps]
        print("Number of lr fps:", len(lr_fps))

        self.datapoints = []
        for i,hr_fp in enumerate(hr_fps):
            load_tensor = torch.load(hr_fp)
            num_patches = load_tensor.shape[0]
            self.datapoints.extend([[hr_fp, lr_fps[i], patch] for patch in range(num_patches)])

        self.data_len = len(self.datapoints)
        print("Loaded ", self.data_len, " data pairs.")

    def __getitem__(self, index):
        hr_path, lr_paths, patch_num = self.datapoints[index]

        hr_tensor = torch.load(hr_path)[patch_num, :, :, :]
        lr_tensor = torch.load(lr_path)[patch_num, :, :, :]
        print("shapes of input:", hr_tensor.shape, lr_tensor.shape)

        img_HR = hr_tensor
        img_LR = lr_tensor

        return {'gt': img_HR, 'lq': img_LR, 'Index': index, 'PatchNum': patch_num}

    def __len__(self):
        return self.data_len
