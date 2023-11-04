"""
Adapted from: https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/models/esrgan_model.py
Authors: XPixelGroup
"""
import cv2
import kornia
import torch
import torchvision
import torch.nn as nn
from collections import OrderedDict

from basicsr.models.srgan_model import SRGANModel
from basicsr.utils import USMSharp
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.archs.rrdbnet_arch import RRDBNet


@MODEL_REGISTRY.register()
class ESRGAN16xModel(SRGANModel):
    """
    SSR ESRGAN Model: Training Satellite Imagery Super Resolution with Paired Training Data.

    The input to the generator is a time series of Sentinel-2 images, and it learns to generate
    a higher resolution image. The discriminator then sees the generated images and NAIP images. 
    """

    def __init__(self, opt):
        super(ESRGAN16xModel, self).__init__(opt)
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening

        self.ssim_loss = opt['ssim_loss'] if 'ssim_loss' in opt else False

        # Could try passing in HighResNet outputs someday.
        #highresnet_weights = '/data/piperw/super-res/satlas-super-resolution/weights/worldstrat_weights/highresnet.pth'
        #self.highresnet = HighResNet()

        # ESRGAN that was trained to upsample Sentinel-2 to 1/4 the resolution of NAIP.
        esrgan_4x_weights = '/data/piperw/super-res/satlas-super-resolution/experiments/satlas32_baseline/models/net_g_925000.pth'
        self.esrgan_4x = RRDBNet(num_in_ch=24, num_out_ch=3, num_feat=128, num_block=23, num_grow_ch=64, scale=4).to(self.device)
        state_dict = torch.load(esrgan_4x_weights)
        self.esrgan_4x.load_state_dict(state_dict['params_ema'])
        self.esrgan_4x.eval()

    @torch.no_grad()
    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)

        # First feed the sentinel-2 images through our trained 4x ESRGAN model,
        # This output will be passed into the 16x model along with the sentinel-2 images.
        lq_4x = self.esrgan_4x(self.lq)

        # Will want to combine the S2 images with the 4x output, which requires upsampling the S2 by 4x.
        lq_resized = torchvision.transforms.functional.resize(self.lq, (128,128))
        self.lq = torch.cat((lq_4x, lq_resized), dim=1)

        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            self.gt_usm = self.usm_sharpener(self.gt)

        self.old_naip = None
        if 'old_naip' in data:
            self.old_naip = data['old_naip'].to(self.device)

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

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, l1_gt)

                ssim = 0.0
                if self.ssim_loss:
                    ssim = torch.mean(kornia.losses.ssim_loss(self.output, l1_gt, window_size=5, reduction="none").mean(dim=(-1,-2,-3)))
                if not ssim == 0.0:
                    l_g_total += ssim
                    loss_dict['l_ssim'] = ssim

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
                    fake_g_pred = self.net_d(output_copy)
                else:
                    print("diff mod layers not implemented for this case yet")
            # If old naip was provided, cat it onto the channel dimension of discriminator input.
            elif self.old_naip is not None:
                output_copy = torch.cat((self.output, self.old_naip), dim=1)
                fake_g_pred = self.net_d(output_copy)
            # If we want to feed the discriminator the sentinel-2 time series.
            elif self.feed_disc_s2:
                if self.diff_mod_layers is None:
                    output_copy = torch.cat((self.output, lq_resized), dim=1)
                else:
                    output_copy = [self.output, self.lq]
                fake_g_pred = self.net_d(output_copy)
            else:
                # gan loss
                fake_g_pred = self.net_d(self.output)

            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
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
        real_d_pred = self.net_d(gan_gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()

        # fake
        if self.feed_disc_s2 and self.diff_mod_layers is not None:
            fake_d_pred = self.net_d([self.output[0], self.output[1].detach().clone()])
        else:
            fake_d_pred = self.net_d(self.output.detach().clone())  # clone for pt1.9
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()
        self.optimizer_d.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)
