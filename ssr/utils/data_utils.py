import glob
import torch
import rasterio
import numpy as np


def has_black_pixels(tensor):
    # Sum along the channel dimension to get a 2D tensor [height, width]
    channel_sum = torch.sum(tensor, dim=0)

    # Check if any pixel has a sum of 0, indicating black
    black_pixels = (channel_sum.view(-1) == 0).any()

    return black_pixels

def get_old_naip(old_naip_path):
    old_naip_chips = {}
    for old_naip in glob.glob(old_naip_path + '*.png', recursive=True):
        old_chip = old_naip.split('/')[-1][:-4]

        if not old_chip in old_naip_chips:
            old_naip_chips[old_chip] = []
        old_naip_chips[old_chip].append(old_naip)
    return old_naip_chips

def load_and_extract_bands(filename, desired_bands=[0, 1], n_bands_per_image=4):
    """
    Load a Sentinel-2 image file and extract specified bands into a PyTorch tensor.

    Parameters:
    - filename: path to the Sentinel-2 image file.
    - desired_bands: indices of the bands to extract, based on their order in each group.
    - n_bands_per_image: total number of bands for each image in the file.

    Returns:
    - A PyTorch tensor containing the extracted bands for all images.
    """

    with rasterio.open(filename) as src:
        # Read the entire file into a numpy array
        data = src.read()
        # data shape is expected to be [n_images*n_bands, height, width]

        # Initialize an empty list to store extracted band data
        extracted_bands = []

        # Calculate the total number of images based on the data shape and number of bands per image
        n_images = data.shape[0] // n_bands_per_image

        # Loop over each image
        for i in range(n_images):
            # For each image, extract the desired bands and append to the list
            for band_index in desired_bands:
                # Calculate the global band index for the current image
                global_band_index = i * n_bands_per_image + band_index
                extracted_bands.append(data[global_band_index])

        # Stack the extracted bands along a new dimension
        extracted_bands_array = np.stack(extracted_bands, axis=0)

        # Normalize to be compatible with satlaspretrain (could change this later?)
        normalized = np.clip((extracted_bands_array / 8160), 0, 1)

        # Convert the numpy array to a PyTorch tensor
        bands_tensor = torch.from_numpy(normalized).float()

        return bands_tensor
