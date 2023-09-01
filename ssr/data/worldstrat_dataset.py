import os
import cv2
import csv
import glob
import torch
import random
import torchvision
import skimage.io
import numpy as np
import torch.nn as nn
from osgeo import gdal
from PIL import Image
from torch.utils import data as data
from torchvision.transforms import functional as trans_fn
from basicsr.utils.registry import DATASET_REGISTRY

totensor = torchvision.transforms.ToTensor()

def pad(im, pad_to):
    if im.shape[2] < pad_to:
        half = (pad_to - im.shape[2]) // 2
        im = nn.functional.pad(im, [pad_to - im.shape[2] - half, half, 0, 0])
    else:
        start = (im.shape[2] - pad_to) // 2
        im = im[:, :, start:start+pad_to]

    if im.shape[1] < pad_to:
        half = (pad_to - im.shape[1]) // 2
        im = nn.functional.pad(im, [0, 0, pad_to - im.shape[1] - half, half])
    else:
        start = (im.shape[1] - pad_to) // 2
        im = im[:, start:start+pad_to, :]
    return im

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

        # Flags whether the model being used expects [b, n_images, channels, h, w] or [b, n_images*channels, h, w].
        self.use_3d = opt['use_3d'] if 'use_3d' in opt else False

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
                if not os.path.exists((os.path.join(self.lr_path, tile, 'L1C', tile+'-'+str(1)+'-L1C_data.tiff'))):
                    continue

                # HR image for the current datapoint.
                hr_img_path = os.path.join(self.hr_path, tile, tile+'_rgb.png')

                # Each HR image has 16 corresponding LR images.
                lrs = []
                for img in range(1, int(opt['n_s2_images'])+1):
                    lr_img_path = os.path.join(self.lr_path, tile, 'L1C', tile+'-'+str(img)+'-L1C_data.tiff')
                    lrs.append(lr_img_path)

                self.datapoints.append([hr_img_path, lrs])

        self.data_len = len(self.datapoints)
        print("Loaded ", self.data_len, " data pairs.")

    def __getitem__(self, index):
        hr_path, lr_paths = self.datapoints[index]

        # Load the HR image with skimage, since it's a png.
        hr_im = skimage.io.imread(hr_path)[:, :, 0:3]
        hr_im = cv2.resize(hr_im, (640, 640)) # NOTE: temporarily downsizing the HR image to match the SR image
        img_HR = totensor(hr_im)

        # Load each of the LR images with gdal, since they're tifs.
        lr_ims = []
        for lr_path in lr_paths:
            raster = gdal.Open(lr_path)
            array = raster.ReadAsArray()
            array = np.clip(array*700, 0, 255).astype(np.uint8)
            lr_im = array.transpose(1, 2, 0)[:, :, 1:4]
            lr_ims.append(lr_im)

        # Pad to some desired spatial dimensions so all images are consistent.
        #lr_ims = [pad(totensor(im), 160) for im in lr_ims]

        # Resize each Sentinel-2 image to the same spatial dimension.
        lr_ims = [totensor(cv2.resize(im, (160,160))) for im in lr_ims]

        img_LR = torch.stack(lr_ims, dim=0)
        if not self.use_3d:
            img_LR = torch.reshape(img_LR, (-1, 160, 160))

        return {'gt': img_HR, 'lq': img_LR, 'Index': index}

    def __len__(self):
        return self.data_len
