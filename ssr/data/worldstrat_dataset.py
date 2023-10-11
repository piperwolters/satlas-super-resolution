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

# WorldStrat constants
SPOT_MAX_EXPECTED_VALUE_12_BIT = 10000
S2_ALL_BANDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
JIF_S2_MEAN = torch.Tensor(
    [
        0.0008,
        0.0117,
        0.0211,
        0.0198,
        0.0671,
        0.1274,
        0.1449,
        0.1841,
        0.1738,
        0.1803,
        0.0566,
        -0.0559,
    ]
).to(torch.float64)
JIF_S2_STD = torch.Tensor(
    [
        0.0892,
        0.0976,
        0.1001,
        0.1169,
        0.1154,
        0.1066,
        0.1097,
        0.1151,
        0.1122,
        0.1176,
        0.1298,
        0.1176,
    ]
).to(torch.float64)


@DATASET_REGISTRY.register()
class WorldStratDataset(data.Dataset):
    """
    Args:
        opt (dict): Config for train datasets. It contains the following keys:
    """

    def __init__(self, opt):
        super(WorldStratDataset, self).__init__()
        self.opt = opt

        self.split = opt['phase']
        if self.split == 'validation':
            self.split = 'val'

        self.lr_path = opt['lr_path']
        self.hr_path = opt['hr_path']
        self.splits_csv = '/data/piperw/worldstrat/dataset/stratified_train_val_test_split.csv'

        # Flags whether to use all bands (13) or just rgb (3).
        self.all_bands = opt['all_bands'] if 'all_bands' in opt else False

        # Flags whether the model being used expects [b, n_images, channels, h, w] or [b, n_images*channels, h, w].
        self.use_3d = opt['use_3d'] if 'use_3d' in opt else False

        if self.all_bands:
            lr_bands_to_use = np.array(S2_ALL_BANDS) - 1
            # Normalization code copied from WorldStrat codebase.
            normalize = Normalize(
                mean=JIF_S2_MEAN[lr_bands_to_use], std=JIF_S2_STD[lr_bands_to_use]
            )
            self.lr_transform = Compose(
                [
                    #Lambda(lambda lr_revisit: torch.as_tensor(lr_revisit)),
                    ToTensor(),
                    normalize,
                    Resize(size=(160,160), interpolation=InterpolationMode.BICUBIC, antialias=True),
                ]
            )
            self.hr_transform = Compose(
                [
                    Lambda(
                        lambda hr_revisit: torch.as_tensor(hr_revisit.astype(np.int32))
                        / SPOT_MAX_EXPECTED_VALUE_12_BIT
                    ),
                    Resize(size=(156,156), interpolation=InterpolationMode.BICUBIC, antialias=True),
                    Lambda(lambda high_res_revisit: high_res_revisit.clamp(min=0, max=1)),
                ]
            )

        # Read in the csv file containing splits and filter out non-relevant images for this split.
        # Build a list of [hr_path, [lr_paths]] lists. 
        self.datapoints = []
        with open(self.splits_csv, newline='') as csvfile:
            read = csv.reader(csvfile, delimiter=' ')
            for i,row in enumerate(read):
                # Skip the row with columns.
                if i == 0:
                    continue

                row = row[0].split(',')
                tile = row[1]
                split = row[-1]
                if split != self.split:
                    continue

                # A few paths are missing even though specified in the split csv, so skip them.
                if not os.path.exists((os.path.join(self.lr_path, tile, 'L2A', tile+'-'+str(1)+'-L2A_data.tiff'))):
                    continue

                # HR image for the current datapoint.
                #if not self.all_bands:
                hr_img_path = os.path.join(self.hr_path, tile, tile+'_rgb.png')
                #else:
                #hr_img_path = os.path.join(self.hr_path, tile, tile+'_ps.tiff')

                # Each HR image has 16 corresponding LR images.
                lrs = []
                for img in range(1, int(opt['n_s2_images'])+1):
                    lr_img_path = os.path.join(self.lr_path, tile, 'L2A', tile+'-'+str(img)+'-L2A_data.tiff')
                    lrs.append(lr_img_path)

                self.datapoints.append([hr_img_path, lrs])

        self.data_len = len(self.datapoints)
        print("Loaded ", self.data_len, " data pairs.")

    def __getitem__(self, index):
        hr_path, lr_paths = self.datapoints[index]

        hr_im = skimage.io.imread(hr_path)[:, :, 0:3]
        hr_im = cv2.resize(hr_im, (640, 640)) # NOTE: temporarily downsizing the HR image to match the SR image
        hr_im = totensor(hr_im)

        # NOTE: do we want to run on all bands or on pansharpened? tbd
        # The following commented out code tries to load pansharpened but was gettin inf psnr when using this as gt.
        #hr_raster = gdal.Open(hr_path)
        #hr_im = np.transpose(hr_raster.ReadAsArray()[0:3, :, :], (1,2,0))
        #hr_im = cv2.resize(hr_im, (640,640)).astype(np.float32)
        #hr_im = totensor(hr_im)

        img_HR = hr_im

        # Load each of the LR images with gdal, since they're tifs.
        lr_ims = []
        for lr_path in lr_paths:
            raster = gdal.Open(lr_path)
            array = raster.ReadAsArray()

            # If all_bands is specified, trying to replicate exact WorldStrat methodology,
            # otherwise have option to run on RGB.
            if self.all_bands:
                lr_im = array.transpose(1, 2, 0)
                lr_im = self.lr_transform(lr_im)
            else:
                lr_im = array.transpose(1, 2, 0)[:, :, 1:4]

            lr_ims.append(lr_im)

        if not self.all_bands:
            # Resize each Sentinel-2 image to the same spatial dimension.
            lr_ims = [totensor(cv2.resize(im, (160,160))) for im in lr_ims]

        img_LR = torch.stack(lr_ims, dim=0)
        if not self.use_3d:
            img_LR = torch.reshape(img_LR, (-1, 160, 160))

        return {'gt': img_HR, 'lq': img_LR, 'Index': index}

    def __len__(self):
        return self.data_len
