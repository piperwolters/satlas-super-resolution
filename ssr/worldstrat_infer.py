import cv2
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ssr.archs.highresnet_arch import HighResNet
from ssr.data.worldstrat_dataset import WorldStratDataset 

device = torch.device('cuda')

opt_file = 'ssr/options/worldstrat/testing_highresnet.yml'
with open(opt_file, 'r') as opt_f:
    opt = yaml.safe_load(opt_f)
    print("opt:", opt)
    opt['datasets']['val']['phase'] = 'val'
    dataset = WorldStratDataset(opt['datasets']['val'])

dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

model = HighResNet(
    in_channels=12, 
    mask_channels=1, 
    revisits=8, 
    hidden_channels=128, 
    out_channels=3, 
    kernel_size=3, 
    residual_layers=1, 
    output_size=(156,156), 
    zoom_factor=2, 
    sr_kernel_size=1
)
weights = torch.load('weights/worldstrat_weights/highresnet.pth')
model.load_state_dict(weights, strict=True)
model.eval().to(device)

psnrs = []
for i,data in enumerate(dataloader):
    hr = data['gt'].to(device)
    lr = data['lq'].to(device)
    print("lr:", lr.shape, " & hr:", hr.shape)

    resized_hr = F.interpolate(hr, (156,156))
    resized_hr = resized_hr / 10000 
    print("resized:", resized_hr.shape)

    output = model(lr)
    print("output:", output.shape)

    # bias adjust brightness using the ground truth
    output = output + ((resized_hr - output).mean(dim=(-1, -2), keepdim=True))

    mse_loss = F.mse_loss(output.squeeze(), resized_hr.squeeze(), reduction="none").mean(dim=(-1, -2, -3))  # over C, H, W
    psnr = -10.0 * torch.log10(mse_loss)
    print("psnr:", psnr)

    psnrs.append(psnr.item())

print("Average Test Set PSNR:", sum(psnrs) / len(psnrs))
