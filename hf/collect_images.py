import os
import sys
import json
import glob
import torch
import random
import torchvision
import skimage.io
import numpy as np

#from basicsr.archs.rrdbnet_arch import RRDBNet
sys.path.append('/data/piperw/super-res/satlas-super-resolution/')
from ssr.archs.rrdbnet_arch import SSR_RRDBNet
totensor = torchvision.transforms.ToTensor()

# Load the dict that tells us OSM features for each NAIP chip
osm_path = '/data/piperw/data/urban_set/osm_chips_to_masks.json'
osm_file = open(osm_path)
osm_data = json.load(osm_file)

# Imagery paths
s2_path = '/data/piperw/data/urban_set/s2_condensed'
naip_path = '/data/piperw/data/urban_set/naip'
naip_chips = glob.glob(naip_path + '/**/*.png', recursive=True)

# Load the pretrained SR model
weights_path = '/data/piperw/super-res/satlas-super-resolution/experiments/satlas32_baseline/models/net_g_905000.pth' 
device = torch.device('cuda') 
n_s2_images = 8
model = SSR_RRDBNet(num_in_ch=24, num_out_ch=3, num_feat=128, num_block=23, num_grow_ch=64, scale=4).to(device)
state_dict = torch.load(weights_path)
model.load_state_dict(state_dict['params_ema'])
model.eval()

# Load the second, 16x model
extra_res_weights = '/data/piperw/super-res/satlas-super-resolution/weights/esrgan_4x_2_16x.pth'
model2 = SSR_RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=128, num_block=23, num_grow_ch=32, scale=4).to(device)
state_dict = torch.load(extra_res_weights)
model2.load_state_dict(state_dict['params_ema'])
model2.eval()

# Process each datapoint
save_building_images = 'building_outputs/'
counter = 0
for n in naip_chips:
    split_path = n.split('/')
    chip = split_path[-1][:-4]

    # Check if this NAIP chip contains an OSM building.
    if not (chip in osm_data and 'building' in osm_data[chip] and len(osm_data[chip]['building']) >= 1):
        continue

    whole_img = skimage.io.imread(n)
    if [0, 0, 0] in whole_img:
        continue

    # Find corresponding S2 images for this NAIP chip.
    tile = int(chip.split('_')[0]) // 16, int(chip.split('_')[1]) // 16 
    s2_left_corner = tile[0] * 16, tile[1] * 16
    diffs = int(chip.split('_')[0]) - s2_left_corner[0], int(chip.split('_')[1]) - s2_left_corner[1]

    # Assuming just TCI for now. Load the image and reshape to expected format.
    s2_paths = [os.path.join(s2_path, str(tile[0])+'_'+str(tile[1]), str(diffs[1])+'_'+str(diffs[0])+'.png')]
    s2_images = skimage.io.imread(s2_paths[0])

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
    if len(goods) >= n_s2_images:
        rand_indices = random.sample(goods, n_s2_images)
    else:
        need = n_s2_images - len(goods)
        rand_indices = goods + random.sample(bads, need)

    s2_chunks = [s2_chunks[i] for i in rand_indices]
    s2_chunks = np.array(s2_chunks)
    first_s2 = s2_chunks[0]  # extract just 1 S2 image of shape (h,w,3)
    s2_chunks = [totensor(img) for img in s2_chunks]
    img_S2 = torch.cat(s2_chunks).unsqueeze(0).to(device)

    # Feed the S2 images through the pretrained model.
    output = model(img_S2)
    output = model2(output)

    save_out = np.transpose(output.squeeze(0).cpu().detach().numpy() * 255, (1,2,0)).astype(np.uint8)
    skimage.io.imsave(save_building_images+chip+'.png', save_out)

    #skimage.io.imsave(save_building_images+chip+'.png', whole_img)
    avg_dims = []
    for i,building in enumerate(osm_data[chip]['building']):
        x1, y1, x2, y2 = [b*4 for b in building]

        # Pad the chunks a bit.
        # If you want to save the NAIP building chunks, bound is 512 and multiply by 4. Use whole_img.
        # If you want to save the S2 building chunks, bound is 32 and divide by 4. Use first_s2.
        x1 = max(x1 - 3, 0) 
        x2 = min(x2 + 3, 512)
        y1 = max(y1 - 3, 0)
        y2 = min(y2 + 3, 512)

        output = output.squeeze(0)
        build = output[:, y1:y2, x1:x2]

        avg_dims.append(build.shape)

        build = np.transpose(build.cpu().detach().numpy() * 255, (1,2,0)).astype(np.uint8)
        skimage.io.imsave(save_building_images+chip+'_'+str(i)+'.png', build)

    counter += 1
    if counter >= 20:
        break

