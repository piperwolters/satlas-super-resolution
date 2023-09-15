"""
Adapted from: https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/models/esrgan_model.py
Authors: XPixelGroup
"""
import cv2
import json
import kornia
import random
import torch
import torchvision
import torch.nn as nn
from collections import OrderedDict

from basicsr.models.srgan_model import SRGANModel
from basicsr.utils import USMSharp
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class OSMESRGANModel(SRGANModel):
    """
    SSR ESRGAN Model: Training Satellite Imagery Super Resolution with Paired Training Data.

    The input to the generator is a time series of Sentinel-2 images, and it learns to generate
    a higher resolution image. The discriminator then sees the generated images and NAIP images. 
    """

    def __init__(self, opt):
        super(OSMESRGANModel, self).__init__(opt)
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening

        # Load in the big json containing maps from chips to OSM object bounds.
        osm_file = open(opt['datasets']['train']['osm_path'])
        self.osm_data = json.load(osm_file)

    @torch.no_grad()
    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            self.gt_usm = self.usm_sharpener(self.gt)

        self.old_naip = None
        if 'old_naip' in data:
            self.old_naip = data['old_naip'].to(self.device)

        # List of dictionaries of objects for each chip in this batch.
        self.chip_objs = [self.osm_data[data['Chip'][c]] for c in range(len(data['Chip']))]

        self.feed_disc_s2 = True if ('feed_disc_s2' in self.opt and self.opt['feed_disc_s2']) else False
        self.diff_mod_layers = self.opt['network_d']['diff_mod_layers'] if 'diff_mod_layers' in self.opt['network_d'] else None

    def optimize_parameters(self, current_iter):
        # usm sharpening
        l1_gt = self.gt_usm
        percep_gt = self.gt_usm
        gan_gt = self.gt_usm
        if self.opt['l1_gt_usm'] is False:
            l1_gt = self.gt
        if self.opt['percep_gt_usm'] is False:
            percep_gt = self.gt
        if self.opt['gan_gt_usm'] is False:
            gan_gt = self.gt

        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        # Extract OSM objects in this chip and resize them to standard size.
        gt_extracted_objs = []
        gen_extracted_objs = []
        for b in range(gan_gt.shape[0]):
            batch_gt, batch_gen = [], []
            for k,v in self.chip_objs[b].items():
                for i,o in enumerate(v):
                    x1, y1, x2, y2 = o

                    # Account for some edge cases that should've been handled in preprocessing script...
                    if x1 == x2:
                        x1, x2 = (x1, x2+1) if x2 < 128 else (x1-1, x2)
                    if y1 == y2:
                        y1, y2 = (y1, y2+1) if y2 < 128 else (y1-1, y2)

                    gt_extract = gan_gt[b, :, y1:y2, x1:x2]
                    gen_extract = self.output[b, :, y1:y2, x1:x2]
                    batch_gt.append(torchvision.transforms.functional.resize(gt_extract, (32,32)))
                    batch_gen.append(torchvision.transforms.functional.resize(gen_extract, (32,32)))

            # list of lists of objects for each batch
            gt_extracted_objs.append(batch_gt)
            gen_extracted_objs.append(batch_gen)

        # We want to randomly pick subset of objects (for now). Each batch has a different 
        # set of random indices, since there will be a different number of objects in each.
        gt_objs, gen_objs = [], []
        for b in range(gan_gt.shape[0]):
            gts = gt_extracted_objs[b]
            gens = gen_extracted_objs[b]

            # Randomly pick a subset of objects for the current image.
            rand_idxs = random.sample([i for i in range(0, len(gts))], 5)

            gt_objs.append(torch.stack([gts[g] for g in range(len(gts)) if g in rand_idxs]))
            gen_objs.append(torch.stack([gens[g] for g in range(len(gens)) if g in rand_idxs]))
        gt_objs = torch.cat(gt_objs, dim=0)
        gen_objs = torch.cat(gen_objs, dim=0)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, l1_gt)
                l_g_total += l_g_pix 
                loss_dict['l_g_pix'] = l_g_pix
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, percep_gt)

                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style

            lq_shp = self.lq.shape
            lq_resized = nn.functional.interpolate(self.lq, scale_factor=4)

            # Special case when we want to pass in both sentinel-2 images and old naip to discriminator.
            if (self.old_naip is not None) and self.feed_disc_s2:
                if self.diff_mod_layers is None:
                    output_copy = torch.cat((self.output, lq_resized, self.old_naip), dim=1)          
                    fake_g_pred, obj_pred = self.net_d(output_copy, gen_objs)
                else:
                    print("diff mod layers not implemented for this case yet")
            # If old naip was provided, cat it onto the channel dimension of discriminator input.
            elif self.old_naip is not None:
                output_copy = torch.cat((self.output, self.old_naip), dim=1)
                fake_g_pred, obj_pred = self.net_d(output_copy, gen_objs)
            # If we want to feed the discriminator the sentinel-2 time series.
            elif self.feed_disc_s2:
                if self.diff_mod_layers is None:
                    output_copy = torch.cat((self.output, lq_resized), dim=1)
                else:
                    output_copy = [self.output, self.lq]
                fake_g_pred, obj_pred = self.net_d(output_copy, gen_objs)
            else:
                # gan loss
                fake_g_pred, obj_pred = self.net_d(self.output, gen_objs)

            print("fake_g_pred:", fake_g_pred.shape, " & obj_pred:", obj_pred.shape)

            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            print("l_g_gan with fake_g_pred:", l_g_gan)
            print("obj_pred l_g_gan?:", self.cri_gan(obj_pred, True, is_disc=False))
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        # If both old naip and sentinel-2 time series were provided to discriminator.
        if (self.old_naip is not None) and self.feed_disc_s2: 
           gan_gt = torch.cat((gan_gt, lq_resized, self.old_naip), dim=1)
           self.output = torch.cat((self.output, lq_resized, self.old_naip), dim=1)
        # If old naip was provided, stack it onto the channel dimension of discriminator input.
        elif self.old_naip is not None:
            gan_gt = torch.cat((gan_gt, self.old_naip), dim=1)
            self.output = torch.cat((self.output, self.old_naip), dim=1)
        # If we want to feed the discriminator the sentinel-2 time series.
        elif self.feed_disc_s2:
            if self.diff_mod_layers is None:
                self.output = torch.cat((self.output, lq_resized), dim=1)
                gan_gt = torch.cat((gan_gt, lq_resized), dim=1)
            else:
                self.output = [self.output, self.lq]
                gan_gt = [gan_gt, self.lq]

        self.optimizer_d.zero_grad()

        # real
        real_d_pred, real_obj_pred = self.net_d(gan_gt, gt_objs)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()

        # fake
        if self.feed_disc_s2 and self.diff_mod_layers is not None:
            fake_d_pred, fake_obj_pred = self.net_d([self.output[0], self.output[1].detach().clone()])
        else:
            fake_d_pred, fake_obj_pred = self.net_d(self.output.detach().clone(), gen_objs)  # clone for pt1.9
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()
        self.optimizer_d.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)
