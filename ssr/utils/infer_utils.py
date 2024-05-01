import cv2
import torch
import random
import skimage.io
import numpy as np

def format_s2naip_data(s2_data, n_s2_images, device):
    # Reshape to be Tx32x32x3.
    s2_chunks = np.reshape(s2_data, (-1, 32, 32, 3))

    # Save one image to a variable for later saving.
    s2_image = s2_chunks[0]

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
    s2_chunks = [torch.as_tensor(img).permute(2, 0, 1) for img in s2_chunks]
    s2_tensor = torch.cat(s2_chunks).unsqueeze(0)
    s2_tensor = s2_tensor.to(device).float()/255

    # Return input of shape [batch, n_s2_images * channels, 32, 32].
    # Also return an S2 image that can be saved for reference.
    return s2_tensor, s2_image

def stitch(chunks_dir, img_size, save_path, scale=4, grid_size=16, sentinel2=False):
    chunk_size = int(img_size / grid_size)

    # Create an empty numpy array that is the shape of the desired output.
    empty = np.zeros((img_size, img_size, 3))
    for i in range(grid_size):
        for j in range(grid_size):
            path = chunks_dir + '/' + str(i) + '_' + str(j) + '.png'
            load = skimage.io.imread(path)

            # Because Sentinel-2 images are saved as [n_s2_images*32, 32, 3], need
            # to reshape it and just take the first image for the purpose of the stitch.
            if sentinel2:
                load = np.reshape(load, (-1, 32, 32, 3))[0]

            # Fill in the numpy array with the respective chunks.
            empty[i*chunk_size:i*chunk_size+chunk_size, j*chunk_size:j*chunk_size+chunk_size, :] = load

    empty = empty.astype(np.uint8)
    skimage.io.imsave(save_path, empty, check_contrast=False)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result
