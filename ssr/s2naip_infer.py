import os
import glob
import torch
import random
import rasterio
import argparse
import skimage.io
import numpy as np

from ssr.utils.data_utils import load_and_extract_bands
from ssr.utils.infer_utils import format_s2naip_data, tensor2img
from ssr.utils.options import yaml_load
from ssr.utils.model_utils import build_network


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help="Path to the options file.")
    args = parser.parse_args()

    device = torch.device('cuda')

    # Load the configuration file.
    opt = yaml_load(args.opt)

    data_dir = opt['data_dir']  # root directory containing the low-res images you want to super-resolve
    n_lr_images = opt['n_lr_images']  # number of low-res images as input to the model; must be the same as when the model was trained
    save_path = opt['save_path']  # directory where model outputs will be saved

    # Define the generator model, based on the type and parameters specified in the config.
    model = build_network(opt)

    # Load the pretrained weights into the model
    if not 'pretrain_network_g' in opt['path']:
        print("WARNING: Model weights are not specified in configuration file.")
    else:
        weights = opt['path']['pretrain_network_g']  # path to the generator weights
        state_dict = torch.load(weights)
        model.load_state_dict(state_dict[opt['path']['param_key_g']], strict=opt['path']['strict_load_g'])
    model = model.to(device).eval()

    s2_tiles = glob.glob(data_dir + '*_8.tif')  # list of all tif files containing 10m bands
    print("Running inference on ", len(s2_tiles), " images.")

    for i,s2_tile in enumerate(s2_tiles):
        print("S2 TILE:", s2_tile)

        s2_rgb = load_and_extract_bands(s2_tile, desired_bands=[2,1,0], n_bands_per_image=4)
        s2_tensor = torch.reshape(s2_rgb, (-1, 3, 512, 512))   # shape [n_imgs*3, 64, 64] -> [n_imgs, 3, 64, 64]
        s2_tensor = s2_tensor[n_lr_images:n_lr_images+8, :, :, :]
        s2_tensor = torch.reshape(s2_tensor, (-1, 512, 512)).unsqueeze(0).to(device)

        for r in range(0, 512, 64):
            for c in range(0, 512, 64):
                save_dir = os.path.join(save_path, str(i), str(r) + '_' + str(c))
                os.makedirs(save_dir, exist_ok=True)

                s2_img = torch.permute(s2_tensor[:, :3, :, :].squeeze(0), (1, 2, 0)).cpu().detach().numpy()
                s2_img = (s2_img * 255).astype(np.uint8)

                # Feed the low-res images through the super-res model.
                input_tensor = s2_tensor[:, :, r:r+64, c:c+64]
                output = model(input_tensor)

                # Save the low-res input image in the same dir as the super-res image so
                # it is easy for the user to compare.
                skimage.io.imsave(save_dir + '/lr.png', s2_img)

                # Convert the model output back to a numpy array and adjust shape and range.
                output = torch.clamp(output, 0, 1)
                output = output.squeeze().cpu().detach().numpy()
                output = np.transpose(output, (1, 2, 0))  # transpose to [h, w, 3] to save as image
                output = (output * 255).astype(np.uint8)

                # Save the super-res output image
                skimage.io.imsave(save_dir + '/sr.png', output, check_contrast=False)

        exit()
