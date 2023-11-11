import os
import cv2
import glob
import torch
import shutil
import random
import argparse
import torchvision
import skimage.io
import numpy as np
import torch.nn.functional as F

from osgeo import gdal

#from basicsr.archs.rrdbnet_arch import RRDBNet
from ssr.archs.rrdbnet_arch import SSR_RRDBNet
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
    extra_res_weights = args.extra_res_weights
    data_txt = args.data_txt

    if 'oli2msi' in data_txt:
        datatype = 'oli2msi'
        base_path = '/data/piperw/data/OLI2MSI/'
        save_path = '/data/piperw/cvpr_outputs/oli2msi/'
    elif 'sen2venus' in data_txt:
        sen2venus_counter = 0
        datatype = 'sen2venus'
        base_path = '/data/piperw/data/sen2venus/'
        save_path = '/data/piperw/cvpr_outputs/sen2venus/'
    elif 'worldstrat' in data_txt:
        datatype == 'worldstrat'
        base_path = '/data/piperw/worldstrat/dataset/lr_dataset/'
        save_path = '/data/piperw/cvpr_outputs/worldstrat/'
    elif 'held_out_set' in data_txt:
        datatype= 'naip-s2'
        base_path = '/data/piperw/data/held_out_set/'
        save_path = '/data/piperw/cvpr_outputs/held_out_set/'
    elif 'probav' in data_txt:
        probav_counter = 0
        datatype = 'probav'
        base_path = '/data/piperw/data/PROBA-V/train/NIR/val/'
        save_path = '/data/piperw/cvpr_outputs/probav/'
    else:
        datatype = 'naip-s2'
        base_path = '/data/piperw/data/val_set/'
        save_path = '/data/piperw/cvpr_outputs/naip-s2/'
    print("Datatype:", datatype)

    # Initialize generator model and load in specified weights.
    state_dict = torch.load(args.weights_path)
    model_type = 'esrgan'  # srcnn, highresnet, esrgan
    if model_type == 'esrgan':
        esrgan_savename = 'testing.png'
        use_3d = False
        model = SSR_RRDBNet(num_in_ch=24, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4).to(device)
        model.load_state_dict(state_dict['params_ema'])
    elif model_type == 'highresnet':
        use_3d = True
        model = HighResNet(in_channels=3, mask_channels=0, hidden_channels=128, out_channels=3, kernel_size=3,
                            residual_layers=1, output_size=(128,128), revisits=9, zoom_factor=4, sr_kernel_size=1).to(device)
        model.load_state_dict(state_dict['params'])
    elif model_type == 'srcnn':
        use_3d = True
        model = SRCNN(in_channels=3, mask_channels=0, hidden_channels=128, out_channels=3, kernel_size=3,
                            residual_layers=1, output_size=(128,128), revisits=9, zoom_factor=4, sr_kernel_size=1).to(device)
        model.load_state_dict(state_dict['params'])
    model.eval()

    txt = open(args.data_txt)
    fps = txt.readlines()
    for i,png in enumerate(fps):
        print("Processing....", i)

        png = png.replace('\n', '')

        # NAIP-S2 inference
        if datatype == 'naip-s2':
            file_info = png.split('/')
            chip = file_info[-1][:-4]
            save_dir = os.path.join(save_path, chip)
            os.makedirs(save_dir, exist_ok=True)
            print('saving to ...', save_dir)

            # Uncomment if you want to save NAIP images
            #naip_im = skimage.io.imread(base_path + 'naip_128/' + png)
            #skimage.io.imsave(save_dir + '/naip.png', naip_im)

            chip = chip.split('_')
            tile = int(chip[0]) // 16, int(chip[1]) // 16 
            s2_left_corner = tile[0] * 16, tile[1] * 16
            diffs = int(chip[0]) - s2_left_corner[0], int(chip[1]) - s2_left_corner[1]

            s2_path = base_path + 's2_condensed/' + str(tile[0])+'_'+str(tile[1]) + '/' + str(diffs[1])+'_'+str(diffs[0]) + '.png'

            s2_im = skimage.io.imread(s2_path)

            output = infer(s2_im, n_s2_images, use_3d, device, None)

            output = torch.clamp(output, 0, 1)

            output = output.squeeze().cpu().detach().numpy()
            output = np.transpose(output*255, (1, 2, 0)).astype(np.uint8)  # transpose to [h, w, 3] to save as image
            skimage.io.imsave(save_dir + '/' + esrgan_savename, output, check_contrast=False)

        elif datatype == 'oli2msi':
            save_dir = os.path.join(save_path, str(i))
            os.makedirs(save_dir, exist_ok=True)

            lr_path = base_path + png # not actually a png oops
            lr_ds = gdal.Open(lr_path)
            lr_arr = np.array(lr_ds.ReadAsArray())
            lr_tensor = torch.tensor(lr_arr).float()
            lr_tensor = lr_tensor.unsqueeze(0).to(device)

            if use_3d:
                lr_tensor = lr_tensor.unsqueeze(0)

            hr_path = lr_path.replace('test_lr', 'test_hr')
            hr_ds = gdal.Open(hr_path)
            hr_arr = np.array(hr_ds.ReadAsArray())

            # Uncomment if you want to save high-res image.
            hr_save = (np.transpose(hr_arr, (1, 2, 0)) * 500).astype(np.uint8)
            hr_save = cv2.resize(hr_save, (320,320))
            cv2.imwrite(save_dir + '/hr.png', hr_save)

            output = model(lr_tensor)
            output = output.squeeze().cpu().detach().numpy()
            output = np.transpose(output*500, (1, 2, 0)).astype(np.uint8)  # transpose to [h, w, 3] to save as image

            cv2.imwrite(save_dir + '/' + model_type + '.png', output)

        elif datatype == 'sen2venus':
            # Only grabbing the RGB S2 image
            if not '10m_b2b3b4b8' in png:
                continue

            lr_path = base_path + png
            hr_path = lr_path.replace('10m', '05m')

            hr_tensor = torch.load(hr_path)[:, :3, :, :].float().to(device)
            lr_tensor = torch.load(lr_path)[:, :3, :, :].float().to(device)

            hr_tensor = F.interpolate(hr_tensor, (128,128))
            lr_tensor = F.interpolate(lr_tensor, (64,64))
                
            lr_tensor = (lr_tensor - torch.min(lr_tensor)) / torch.max(lr_tensor)

            for patch in range(hr_tensor.shape[0]):

                save_dir = os.path.join(save_path, str(sen2venus_counter))
                os.makedirs(save_dir, exist_ok=True)
                sen2venus_counter += 1

                lr_patch = lr_tensor[patch, :, :, :].unsqueeze(0)
                hr_patch = lr_tensor[patch, :, :, :]

                if use_3d:
                    lr_patch = lr_patch.unsqueeze(0)

                print("input range:", torch.min(lr_patch), torch.max(lr_patch))
                output = model(lr_patch)

                output = output.squeeze().cpu().detach().numpy()
                output = np.transpose(output * 255, (1, 2, 0)).astype(np.uint8)  # transpose to [h, w, 3] to save as image
                print("range of output save:", np.min(output), np.max(output))
                cv2.imwrite(save_dir + '/' + model_type + '.png', output)

                # Uncomment if you want to save high-res images.
                hr_arr = hr_patch.detach().cpu().numpy()
                hr_save = (np.transpose(hr_arr * 255, (1, 2, 0))).astype(np.uint8)  # / 3000 ?
                print("range of hr save:", np.min(hr_save), np.max(hr_save))
                cv2.imwrite(save_dir + '/hr.png', hr_save)

        elif datatype == 'probav':
            hr_path = base_path + png

            lr_paths = []
            for i in range(n_s2_images):
                if i < 10:
                    lr = hr_path.replace('HR', 'LR00' + str(i))
                else:
                    lr = hr_path.replace('HR', 'LR0' + str(i))
                lr_paths.append(lr)

            hr_im = cv2.imread(hr_path)
            hr_tensor = totensor(hr_im)

            lr_ims = []
            for lr_path in lr_paths:
                lr_im = cv2.imread(lr_path)
                lr_tensor = totensor(lr_im)
                lr_ims.append(lr_tensor)

            if use_3d:
                img_LR = torch.stack(lr_ims).unsqueeze(0)
            else:
                img_LR = torch.cat(lr_ims).unsqueeze(0)
            img_HR = hr_tensor

            for i in range(3):
                for j in range(3):
                    save_dir = os.path.join(save_path, str(probav_counter))
                    os.makedirs(save_dir, exist_ok=True)
                    probav_counter += 1

                    start_x, start_y = i * 32, j * 32
                    hstart_x, hstart_y = i * 128, j * 128
                    hr_chunk = img_HR[:, hstart_x:hstart_x+128, hstart_y:hstart_y+128]

                    if use_3d:
                        lr_chunk = img_LR[:, :, :, start_x:start_x+32, start_y:start_y+32].to(device)
                    else:
                        lr_chunk = img_LR[:, :,start_x:start_x+32, start_y:start_y+32].to(device)

                    print("saving to:", save_dir + '/' + model_type + '.png')
                    print(hr_chunk.shape, lr_chunk.shape)

                    output = model(lr_chunk)

                    output = output.squeeze().cpu().detach().numpy()
                    output = np.transpose(output*1000, (1, 2, 0)).astype(np.uint8)  # transpose to [h, w, 3] to save as image
                    print("range of output save:", np.min(output), np.max(output))
                    cv2.imwrite(save_dir + '/' + model_type + '.png', output)

                    hr_save = hr_chunk.squeeze().cpu().detach().numpy()
                    hr_save = np.transpose(hr_save*1000, (1, 2, 0)).astype(np.uint8)
                    cv2.imwrite(save_dir + '/hr.png', hr_save)

        elif datatype == 'worldstrat':
            print("wordlstrat")

        else:
            print("dataset not supported")
