import os
import cv2
import yaml
import torch
import skimage.io
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ssr.archs.srcnn_arch import SRCNN
from ssr.archs.highresnet_arch import HighResNet
from basicsr.archs.rrdbnet_arch import RRDBNet
from ssr.data.worldstrat_dataset import WorldStratDataset 

device = torch.device('cuda')
save_dir = '/data/piperw/data/mturk/worldstrat/'

# Use just one option file to make the dataset, hardcode model-specifiy details
opt_file = 'ssr/options/cvpr-baselines/highresnet_worldstrat.yml'
with open(opt_file, 'r') as opt_f:
    opt = yaml.safe_load(opt_f)
    print("opt:", opt)
    opt['datasets']['val']['phase'] = 'val'
    dataset = WorldStratDataset(opt['datasets']['val'])

dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Initialize the 3 satlas-super-resolution supported models
srcnn_model = SRCNN(
    in_channels=3,
    mask_channels=0,
    revisits=8,
    hidden_channels=128,
    out_channels=3,
    kernel_size=3,
    residual_layers=1,
    output_size=(640,640),
    zoom_factor=4,
    sr_kernel_size=1
)

high_model = HighResNet(
    in_channels=3, 
    mask_channels=0, 
    revisits=8, 
    hidden_channels=128, 
    out_channels=3, 
    kernel_size=3, 
    residual_layers=1, 
    output_size=(640,640), 
    zoom_factor=4, 
    sr_kernel_size=1
)

esrgan_model = RRDBNet(num_in_ch=24, num_out_ch=3, num_feat=128, num_block=23, num_grow_ch=64, scale=4)

# Load weights into respective models
srcnn_weights = torch.load('/data/piperw/super-res/satlas-super-resolution/experiments/srcnn_worldstrat/models/net_g_95000.pth')
srcnn_model.load_state_dict(srcnn_weights['params'], strict=True)
srcnn_model.eval().to(device)

high_weights = torch.load('/data/piperw/super-res/satlas-super-resolution/experiments/highresnet_worldstrat/models/net_g_90000.pth')
high_model.load_state_dict(high_weights['params'], strict=True)
high_model.eval().to(device)

esrgan_weights = torch.load('/data/piperw/super-res/satlas-super-resolution/experiments/esrgan_worldstrat/models/net_g_550000.pth')
esrgan_model.load_state_dict(esrgan_weights['params_ema'], strict=True)
esrgan_model.eval().to(device)

model_names = ['srcnn', 'highresnet', 'esrgan']
for i,data in enumerate(dataloader):
    print("Processing...", i)

    hr = data['gt'].to(device)
    lr = data['lq'].to(device)

    resized_hr = hr

    srcnn_output = srcnn_model(lr)
    high_output = high_model(lr)
    esrgan_output = esrgan_model(torch.reshape(lr, (1, 24, 160, 160)))
    esrgan_output = F.interpolate(esrgan_output, (640,640))
    outputs = [srcnn_output, high_output, esrgan_output]

    for o,output in enumerate(outputs):
        # bias adjust brightness using the ground truth
        output = output + ((resized_hr - output).mean(dim=(-1, -2), keepdim=True))
        print("range of output :", torch.min(output), torch.max(output))

        # format and save outputs
        output = torch.clamp(output, 0, 1)
        save_output = np.transpose(output.squeeze().detach().cpu().numpy(), (1, 2, 0))
        save_output = (save_output * 255).astype(np.uint8)

        save_path = save_dir + str(i) + '/' + model_names[o] + '.png'
        os.makedirs(save_dir + str(i) + '/', exist_ok=True)
        skimage.io.imsave(save_path, save_output)

        mse_loss = F.mse_loss(output.squeeze(), resized_hr.squeeze(), reduction="none").mean(dim=(-1, -2, -3))  # over C, H, W
        psnr = -10.0 * torch.log10(mse_loss)
        print(model_names[o], " psnr:", psnr)

    if not os.path.exists(save_dir + str(i) + '/hr.png'):
        save_hr = np.transpose(hr.squeeze().detach().cpu().numpy(), (1, 2, 0))
        save_hr = (save_hr * 255).astype(np.uint8)
        skimage.io.imsave(save_dir + str(i) + '/hr.png', save_hr)



