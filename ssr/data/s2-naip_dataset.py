import os
import cv2
import glob
import torch
import random
import skimage.io
import torchvision
import numpy as np
from torch.utils import data as data
from torch.utils.data import WeightedRandomSampler
from torchvision.transforms import functional as trans_fn

from basicsr.utils.registry import DATASET_REGISTRY

totensor = torchvision.transforms.ToTensor()


class CustomWeightedRandomSampler(WeightedRandomSampler):
    """
    WeightedRandomSampler except allows for more than 2^24 samples to be sampled.
    Source code: https://github.com/pytorch/pytorch/issues/2576#issuecomment-831780307
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        rand_tensor = np.random.choice(range(0, len(self.weights)),
                                       size=self.num_samples,
                                       p=self.weights.numpy() / torch.sum(self.weights).numpy(),
                                       replace=self.replacement)
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())

@DATASET_REGISTRY.register()
class S2NAIPDataset(data.Dataset):
    """
    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            sentinel2_path (str): Data path for Sentinel-2 imagery.
            naip_path (str): Data path for NAIP imagery.
            n_sentinel2_images (int): Number of Sentinel-2 images to use as input to model.
            scale (int): Upsample amount, only 4x is supported currently.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(S2NAIPDataset, self).__init__()
        self.opt = opt

        self.split = opt['phase']
        self.n_s2_images = int(opt['n_s2_images'])
        self.scale = int(opt['scale'])

        # Flags whether the model being used expects [b, n_images, channels, h, w] or [b, n_images*channels, h, w].
        # The L2-based models expect the first shape, while the ESRGAN models expect the latter.
        self.use_3d = opt['use_3d'] if 'use_3d' in opt else False

        # Path to high-res images of older timestamps and corresponding locations to training data.
        # In the case of the S2NAIP dataset, that means NAIP images from 2016-2018.
        self.old_naip_path = opt['old_naip_path'] if 'old_naip_path' in opt else None

        # If a path to older NAIP imagery is provided, build dictionary of each chip:path to png.
        if self.old_naip_path is not None:
            old_naip_chips = {}
            for old_naip in glob.glob(self.old_naip_path + '/**/*.png', recursive=True):
                old_chip = old_naip.split('/')[-1][:-4]

                if not old_chip in old_naip_chips:
                    old_naip_chips[old_chip] = []
                old_naip_chips[old_chip].append(old_naip)

        # Paths to Sentinel-2 and NAIP imagery.
        self.s2_path = opt['sentinel2_path']
        self.naip_path = opt['naip_path']
        if not (os.path.exists(self.s2_path) and os.path.exists(self.naip_path)):
            raise Exception("Please make sure the paths to the data directories are correct.")

        self.naip_chips = glob.glob(self.naip_path + '/**/*.png', recursive=True)

        self.datapoints = []
        for n in self.naip_chips:

            # Extract the X,Y tile from this NAIP image filepath.
            split_path = n.split('/')
            chip = split_path[-1][:-4]
            tile = int(chip.split('_')[0]) // 16, int(chip.split('_')[1]) // 16

            # If old_hr_path is specified, grab an old high-res image (NAIP) for the current datapoint.
            if self.old_naip_path is not None:
                old_chip = old_naip_chips[chip][0]

            # Now compute the corresponding Sentinel-2 tiles.
            s2_left_corner = tile[0] * 16, tile[1] * 16
            diffs = int(chip.split('_')[0]) - s2_left_corner[0], int(chip.split('_')[1]) - s2_left_corner[1]

            s2_path = os.path.join(self.s2_path, str(tile[0])+'_'+str(tile[1]), str(diffs[1])+'_'+str(diffs[0])+'.png')

            # Return the low-res, high-res, and [optionally] older high-res image paths. 
            if self.old_naip_path:
                self.datapoints.append([n, s2_path, old_chip])
            else:
                self.datapoints.append([n, s2_path])

        self.data_len = len(self.datapoints)
        print("Number of datapoints for split ", self.split, ": ", self.data_len)

    def get_tile_weight_sampler(self, tile_weights):
        weights = []
        for dp in self.datapoints:
            # Extract the NAIP chip from this datapoint's NAIP path.
            # With the chip, we can index into the tile_weights dict (naip_chip : weight)
            # and then weight this datapoint pair in self.datapoints based on that value.
            naip_path = dp[0]
            split = naip_path.split('/')[-1]
            chip = split[:-4]

            # If the chip isn't in the tile weights dict, then there weren't any OSM features
            # in that chip, so we can set the weight to be relatively low (ex. 1).
            if not chip in tile_weights:
                weights.append(1)
            else:
                weights.append(tile_weights[chip])

        print('Using tile_weight_sampler, min={} max={} mean={}'.format(min(weights), max(weights), np.mean(weights)))
        return CustomWeightedRandomSampler(weights, len(self.datapoints))

    def __getitem__(self, index):

        # A while loop and try/excepts to catch a few images that we want to ignore during 
        # training but do not necessarily want to remove from the dataset, such as the
        # ground truth NAIP image being partially invalid (all black). 
        counter = 0
        while True:
            index += counter  # increment the index based on what errors have been caught
            if index >= self.data_len:
                index = 0

            datapoint = self.datapoints[index]

            if self.old_naip_path:
                naip_path, s2_path, old_naip_path = datapoint[0], datapoint[1], datapoint[2]
            else:
                naip_path, s2_path = datapoint[0], datapoint[1]

            # Load the 128x128 NAIP chip.
            naip_chip = skimage.io.imread(naip_path)

            # Check for black pixels (almost certainly invalid) and skip if found.
            if [0, 0, 0] in naip_chip:
                counter += 1
                continue

            # Load the T*32x32 S2 file.
            # There are a few rare cases where loading the Sentinel-2 image fails, skip if found.
            try:
                s2_images = skimage.io.imread(s2_path)
            except:
                counter += 1
                continue

            # Reshape to be Tx32x32.
            s2_chunks = np.reshape(s2_images, (-1, 32, 32, 3))

            # Iterate through the 32x32 chunks at each timestep, separating them into "good" (valid)
            # and "bad" (partially black, invalid). Will use these to pick best collection of S2 images.
            goods, bads = [], []
            for i,ts in enumerate(s2_chunks):
                if [0, 0, 0] in ts:
                    bads.append(i)
                else:
                    goods.append(i)

            # Pick 18 random indices of s2 images to use. Skip ones that are partially black.
            if len(goods) >= self.n_s2_images:
                rand_indices = random.sample(goods, self.n_s2_images)
            else:
                need = self.n_s2_images - len(goods)
                rand_indices = goods + random.sample(bads, need)

            s2_chunks = [s2_chunks[i] for i in rand_indices]
            s2_chunks = np.array(s2_chunks)
            s2_chunks = [totensor(img) for img in s2_chunks]
            img_HR = totensor(naip_chip)

            if self.use_3d:
                img_S2 = torch.stack(s2_chunks)
            else:
                img_S2 = torch.cat(s2_chunks)

            if self.old_naip_path is not None:
                old_naip_chip = skimage.io.imread(old_naip_path)
                img_old_HR = totensor(old_naip_chip)
                return {'hr': img_HR, 'lr': img_S2, 'old_hr': img_old_HR, 'Index': index}

            return {'hr': img_HR, 'lr': img_S2, 'Index': index}

    def __len__(self):
        return self.data_len