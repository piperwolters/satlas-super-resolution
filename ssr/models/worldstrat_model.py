import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from basicsr.models.sr_model import SRModel
from basicsr.utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class WorldStratModel(SRModel):
    """
    Wrapper model code to run the architectures from WorldStrat codebase.
    """

    def __init__(self, opt):
        super(WorldStratModel, self).__init__(opt)

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                print(k, v)
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        print("len(optim_params):", len(optim_params))
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        print("list(model.parameters())[0].grad:", list(self.net_g.parameters())[0].grad)

    @torch.no_grad()
    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        loss_dict = OrderedDict()

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq).squeeze()

        # total loss = 0.3*w_mse + 0.4*w_mae + 0.3*w_ssim
        #mse = F.mse_loss(self.output, self.gt, reduction="none").mean(dim=(-1, -2, -3))
        #mae = F.l1_loss(self.output, self.gt, reduction="none").mean(dim=(-1, -2, -3))
        #ssim = kornia.losses.ssim_loss(self.output, self.gt, window_size=5, reduction="non").mean(dim=(-1,-2,-3))
        #loss = torch.mean((0.3*mse) + (0.4*mae) + (0.3*ssim))

        # Trying to replicate the SRModel loss calculation to fix model updates?
        loss = self.cri_pix(self.output, self.gt)
        loss_dict['l_g_pix'] = loss
        print("loss .grad_fn:", loss.grad_fn, loss.requires_grad, type(loss))

        loss.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
