import os
import json
import glob
import torch
import random
import rasterio
import torchvision
import numpy as np
import torch.nn.functional as F
from torch.utils import data as data
from torch.utils.data import WeightedRandomSampler

from basicsr.utils.registry import DATASET_REGISTRY

from ssr.utils.data_utils import *

random.seed(123)


@DATASET_REGISTRY.register()
class S2NAIPv2Dataset(data.Dataset):
    """
    Dataset object for the S2NAIP data. Builds a list of Sentinel-2 time series and NAIP image pairs.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            sentinel2_path (str): Data path for Sentinel-2 imagery.
            naip_path (str): Data path for NAIP imagery.
            n_sentinel2_images (int): Number of Sentinel-2 images to use as input to model.
            scale (int): Upsample amount, only 4x is supported currently.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(S2NAIPv2Dataset, self).__init__()
        self.opt = opt

        self.split = opt['phase']
        train = True if self.split == 'train' else False

        self.rand_crop = opt['rand_crop'] if 'rand_crop' in opt else False  # random crop augmentation (training only)
        self.n_s2_images = int(opt['n_s2_images'])  # number of sentinel-2 images to be used as input
        self.scale = int(opt['scale'])  # upsample factor
        self.s2_bands = opt['s2_bands'] if 's2_bands' in opt else 'rgb'  # bands to be used as input. Either rgb, 10m, 20m, or 60m

        # TODO: code for worldcover, osm, and sentinel1?
        # High-res images at older timestamps than the training set. For S2NAIP this is 2016-2018 NAIP images.
        self.old_naip_chips = get_old_naip(opt['old_naip_path']) if 'old_naip_path' in opt else None

        # Paths to Sentinel-2 and NAIP imagery. Assert that at least the LR path is provided.
        self.s2_path = opt['sentinel2_path']
        self.naip_path = opt['naip_path'] if 'naip_path' in opt else None
        if not os.path.exists(self.s2_path):
            raise Exception("Please make sure the paths to the data directories are correct.")

        s2_tiles = glob.glob(self.s2_path + '*_8.tif')  # list of all tif files containing 10m bands

        # Reduce the training set down to a specified number of samples. If not specified, whole set is used.
        if 'train_samples' in opt and train:
            s2_tiles = random.sample(s2_tiles, opt['train_samples'])

        self.datapoints = []
        self.indexes = []
        for i,s2_tile in enumerate(s2_tiles):

            s2_paths = []
            s2_paths.append(s2_tile)
            if self.s2_bands in ['20m', '60m']:
                s2_paths.append(s2_tile.replace('_8.tif', '_16.tif'))
                if self.s2_bands == '60m':
                    s2_paths.append(s2_tile.replace('_8.tif', '_32.tif'))

            naip_path = None
            if self.naip_path is not None:
                naip_path = s2_tile.replace('sentinel2', 'naip').replace('_8.tif', '.png')

            old_naip_path = None
            if self.old_naip_chips is not None:
                old_naip_path = old_naip_chips[chip][0]

            for r in range(0, 512, 64):
                for c in range(0, 512, 64):
                    self.datapoints.append([s2_paths, naip_path, old_naip_path])
                    self.indexes.append([r, c])  # when we want to run inference on larger image, have to chunk

        # Split train vs val
        #rand_dps = random.sample(self.datapoints, 100)
        #if train:
        #    self.datapoints = [item for item in self.datapoints if item not in rand_dps]
        #else:
        #    self.datapoints = rand_dps

        self.data_len = len(self.datapoints)
        print("Number of datapoints for split ", self.split, ": ", self.data_len)

    def __getitem__(self, index):

        datapoint = self.datapoints[index]
        s2_paths, naip_path, old_naip_path = datapoint

        r, c = self.indexes[index]  # index into the S2 image

        # Load the S2 tif files (will either be 1,2,or 3 files to load)
        s2_tensor = None
        # 4 bands:  B02, B03, B04, and B08 formatted like [B02B03B04B08B02B03B04B08...]
        if self.s2_bands == 'rgb':
            s2_rgb = load_and_extract_bands(s2_paths[0], desired_bands=[2,1,0], n_bands_per_image=4)
            #s2_rgb = s2_rgb.repeat(8, 1, 1)
            s2_rgb = s2_rgb[:, r:r+64, c:c+64]
            s2_tensor = torch.reshape(s2_rgb, (-1, 3, 64, 64))   # shape [n_imgs*3, 64, 64] -> [n_imgs, 3, 64, 64]
        elif self.s2_bands == '10m':
            s2_10m = load_and_extract_bands(s2_paths[1], desired_bands=[2,1,0,3], n_bands_per_image=4)
        #elif self.s2_bands == '20m':
        #    s2_20m = load_and_extract_bands(s2_paths[0], desired_bands=[2,1,0], n_bands_per_image=6)
        """
        # 6 bands: B05, B06, B07, B8A, B11, and B12
        if self.s2_bands in ['20m', '60m'] and len(s2_paths) >= 2:
            s2_20m = rasterio.open(s2_paths[1]).read()
        # 3 bands: B01, B09, and B10
        if self.s2_bands == '60m' and len(s2_paths) == 3:
            s2_60m = rasterio.open(s2_paths[2]).read()
        """

        # Load the NAIP png file
        img_HR = None
        if naip_path is not None:
            # Load the NAIP chip in as a tensor of shape [channels, height, width].
            naip_chip = torchvision.io.read_image(naip_path)  # shape [4, 512, 512]

            # Downsample naip to be just x4 the resolution of sentinel2 (could change later?)
            downsampled = torch.nn.functional.interpolate(naip_chip.unsqueeze(0), size=(256,256), mode='bilinear')
            rgb = downsampled.squeeze(0)[:3]
            img_HR = rgb
        else:
            img_HR = torch.zeros((s2_tensor.shape[0], 256, 256))

        # Iterate through the 64x64 tci chunks at each timestep, separating them into "good" (valid)
        # and "bad" (partially black, invalid). Will use these to pick best collection of S2 images.
        tci_chunks = s2_tensor[:, :3, :, :]
        goods, bads = [], []
        for i,ts in enumerate(tci_chunks):
            if has_black_pixels(ts):
                bads.append(i)
            else:
                goods.append(i)

        # Pick self.n_s2_images random indices of S2 images to use. Skip ones that are partially black.
        if len(goods) >= self.n_s2_images:
            rand_indices = random.sample(goods, self.n_s2_images)
        else:
            need = self.n_s2_images - len(goods)
            rand_indices = goods + random.sample(bads, need)
        rand_indices_tensor = torch.as_tensor(rand_indices)

        # Extract the self.n_s2_images from the first dimension.
        img_S2 = s2_tensor[rand_indices_tensor]
        # Reshape to stack n_imgsxbands into 1 dimension
        img_S2 = torch.reshape(img_S2, (-1, 64, 64))

        return {'hr': img_HR, 'lr': img_S2, 'Index': index, 'Phase': self.split}

    def __len__(self):
        return self.data_len
