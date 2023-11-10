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
import torch.nn.functional as F
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

        self.use_3d = opt['use_3d'] if 'use_3d' in opt else False

        hr_fps = glob.glob(self.data_root + '**/*_05m_b2b3b4b8.pt')
        
        # Filter filepaths based on if the split is train or validation.
        if self.split == 'train':
            hr_fps = [hr_fp for hr_fp in hr_fps if not ('JAM2018' in hr_fp or 'BENGA' in hr_fp or 'SO2' in hr_fp)]
        else:
            hr_fps = [hr_fp for hr_fp in hr_fps if ('JAM2018' in hr_fp or 'BENGA' in hr_fp or 'SO2' in hr_fp)]

        lr_fps = [hr.replace('05m', '10m') for hr in hr_fps]

        self.datapoints = []
        for i,hr_fp in enumerate(hr_fps):
            load_tensor = torch.load(hr_fp)
            num_patches = load_tensor.shape[0]
            self.datapoints.extend([[hr_fp, lr_fps[i], patch] for patch in range(num_patches)])

        self.data_len = len(self.datapoints)
        print("Loaded ", self.data_len, " data pairs for split ", self.split)

    def __getitem__(self, index):
        hr_path, lr_path, patch_num = self.datapoints[index]

        hr_tensor = torch.load(hr_path)[patch_num, :3, :, :].float()
        lr_tensor = torch.load(lr_path)[patch_num, :3, :, :].float()
        hr_tensor = F.interpolate(hr_tensor.unsqueeze(0), (128,128)).squeeze(0)
        lr_tensor = F.interpolate(lr_tensor.unsqueeze(0), (64,64)).squeeze(0)

        if self.use_3d:
            lr_tensor = lr_tensor.unsqueeze(0)

        img_HR = hr_tensor
        img_LR = lr_tensor

        return {'gt': img_HR, 'lq': img_LR, 'Index': index, 'PatchNum': patch_num}

    def __len__(self):
        return self.data_len
