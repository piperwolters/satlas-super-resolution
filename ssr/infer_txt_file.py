import os
import glob
import torch
import shutil
import random
import argparse
import torchvision
import skimage.io
import numpy as np

from basicsr.archs.rrdbnet_arch import RRDBNet
from ssr.archs.highresnet_arch import HighResNet
from ssr.archs.srcnn_arch import SRCNN

totensor = torchvision.transforms.ToTensor()


def infer(s2_data, n_s2_images, use_3d, device, extra_res=None):
    # Reshape to be Tx32x32x3.
    s2_chunks = np.reshape(s2_data, (-1, 32, 32, 3))

    # Iterate through the 32x32 chunks at each timestep, separating them into "good" (valid)
    # and "bad" (partially black, invalid). Will use these to pick best collection of S2 images.
    goods, bads = [], []
    for i,ts in enumerate(s2_chunks):
        if [0, 0, 0] in ts:
            bads.append(i)
        else:
            goods.append(i)

    # Pick {n_s2_images} random indices of s2 images to use. Skip ones that are partially black.
    if len(goods) >= n_s2_images:
        rand_indices = random.sample(goods, n_s2_images)
    else:
        need = n_s2_images - len(goods)
        rand_indices = goods + random.sample(bads, need)

    s2_chunks = [s2_chunks[i] for i in rand_indices]
    s2_chunks = np.array(s2_chunks)

    # Convert to torch tensor.
    s2_chunks = [totensor(img) for img in s2_chunks]
    if use_3d:
        s2_tensor = torch.stack(s2_chunks).unsqueeze(0).to(device)
    else:
        s2_tensor = torch.cat(s2_chunks).unsqueeze(0).to(device)

    # Feed input of shape [batch, n_s2_images * channels, 32, 32] through model.
    output = model(s2_tensor)

    # If extra_res is specified, run output through the 4x->16x model after the 4x model.
    if extra_res is not None:
        output = extra_res(output)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_txt', type=str, help="Path to a txt file with a list of naip filepaths.")
    parser.add_argument('-w', '--weights_path', type=str, default="weights/esrgan_orig_6S2.pth", help="Path to the model weights.")
    parser.add_argument('--n_s2_images', type=int, default=8, help="Number of Sentinel-2 images as input, must correlate to correct model weights.")
    parser.add_argument('--save_path', type=str, default="outputs", help="Directory where generated outputs will be saved.")
    parser.add_argument('--extra_res_weights', help="Weights to a trained 4x->16x model. Doesn't currently work with stitch I don't think.")
    args = parser.parse_args()

    device = torch.device('cuda')
    n_s2_images = args.n_s2_images
    save_path = 'outputs' #args.save_path
    extra_res_weights = args.extra_res_weights
    data_txt = args.data_txt

    # Initialize generator model and load in specified weights.
    state_dict = torch.load(args.weights_path)
    model_type = 'RRDBNet'  # RRDBNet, HighResNet, SRCNN
    if model_type == 'RRDBNet':
        use_3d = False
        model = RRDBNet(num_in_ch=24, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4).to(device)
        model.load_state_dict(state_dict['params_ema'])
    elif model_type == 'HighResNet':
        use_3d = True
        model = HighResNet(in_channels=3, mask_channels=0, hidden_channels=128, out_channels=3, kernel_size=3,
                            residual_layers=1, output_size=(128,128), revisits=8, zoom_factor=4, sr_kernel_size=1).to(device)
        model.load_state_dict(state_dict['params'])
    elif model_type == 'SRCNN':
        use_3d = True
        model = SRCNN(in_channels=3, mask_channels=0, hidden_channels=128, out_channels=3, kernel_size=3,
                            residual_layers=1, output_size=(128,128), revisits=8, zoom_factor=4, sr_kernel_size=1).to(device)
        model.load_state_dict(state_dict['params'])
    model.eval()

    txt = open(args.data_txt)
    fps = txt.readlines()
    for i,png in enumerate(fps):
        print("Processing....", i)

        png = png.replace('\n', '')

        file_info = png.split('/')
        chip = file_info[-1][:-4]
        save_dir = os.path.join(save_path, chip)
        os.makedirs(save_dir, exist_ok=True)
        print('saving to ...', save_dir)

        # Uncomment if you want to save NAIP images
        #naip_im = skimage.io.imread(png)
        #skimage.io.imsave(save_dir + '/naip.png', naip_im)

        chip = chip.split('_')
        tile = int(chip[0]) // 16, int(chip[1]) // 16

        s2_left_corner = tile[0] * 16, tile[1] * 16
        diffs = int(chip[0]) - s2_left_corner[0], int(chip[1]) - s2_left_corner[1]

        s2_path = '/data/piperw/data/val_set/s2_condensed/' + str(tile[0])+'_'+str(tile[1]) + '/' + str(diffs[1])+'_'+str(diffs[0]) + '.png'
        #s2_path = '/data/piperw/data/full_dataset/s2_condensed/' + str(tile[0])+'_'+str(tile[1]) + '/' + str(diffs[1])+'_'+str(diffs[0]) + '.png'

        s2_im = skimage.io.imread(s2_path)

        output = infer(s2_im, n_s2_images, use_3d, device, None)

        output = output.squeeze().cpu().detach().numpy()
        print("range befreo *255 and uint8:", np.min(output), np.max(output))
        output = np.transpose(output*255, (1, 2, 0)).astype(np.uint8)  # transpose to [h, w, 3] to save as image
        skimage.io.imsave(save_dir + '/esrgan_1perc.png', output, check_contrast=False)

