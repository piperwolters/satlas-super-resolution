"""
Adapted from: https://https://github.com/worldstrat/worldstrat/blob/main/src/modules.py
Authors: Ivan Oršolić, Julien Cornebise, Ulf Mertens, Freddie Kalaitzis
"""
import torch
import numpy as np
from kornia.geometry.transform import Resize
from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import *

S2_ALL_12BANDS = {
    "true_color": [4, 3, 2],
    "false_color": [8, 4, 3],
    "swir": [12, 8, 4],
    "agriculture": [11, 8, 2],
    "geology": [12, 11, 2],
    "bathimetric": [4, 3, 1],
    "true_color_zero_index": [3, 2, 1],
}

# Precomputed on training data by running: src/datasets.py
JIF_S2_MEAN = torch.Tensor(
    [
        0.0008,
        0.0117,
        0.0211,
        0.0198,
        0.0671,
        0.1274,
        0.1449,
        0.1841,
        0.1738,
        0.1803,
        0.0566,
        -0.0559,
    ]
).to(torch.float64) 

JIF_S2_STD = torch.Tensor(
    [
        0.0892,
        0.0976,
        0.1001,
        0.1169,
        0.1154,
        0.1066,
        0.1097,
        0.1151,
        0.1122,
        0.1176,
        0.1298,
        0.1176,
    ]
).to(torch.float64)


@ARCH_REGISTRY.register()
class Bicubic(nn.Module):
    """ Bicubic upscaled single-image baseline. """

    def __init__(
        self, input_size, output_size, chip_size, interpolation="bicubic", device=None, **kws
    ):
        """ Initialize the BicubicUpscaledBaseline.

        Parameters
        ----------
        input_size : tuple of int
            The input size.
        output_size : tuple of int
            The output size.
        chip_size : tuple of int
            The chip size.
        interpolation : str, optional
            The interpolation method, by default 'bicubic'.
            Available methods: 'nearest', 'bilinear', 'bicubic'.
        """
        super().__init__()
        assert interpolation in ["bilinear", "bicubic", "nearest"]
        self.resize = Resize(output_size, interpolation=interpolation)
        self.output_size = output_size
        self.input_size = input_size
        self.chip_size = chip_size
        self.lr_bands = np.array(S2_ALL_12BANDS["true_color"]) - 1
        self.mean = JIF_S2_MEAN[self.lr_bands]
        self.std = JIF_S2_STD[self.lr_bands]
        self.device = device

    def forward(self, x: Tensor) -> Tensor:
        """ Forward pass of the BicubicUpscaledBaseline.

        Parameters
        ----------
        x : Tensor
            The input tensor (a batch of low-res revisits).
            Shape: (batch_size, revisits, channels, height, width).

        Returns
        -------
        Tensor
            The output tensor (a single upscaled low-res revisit).
        """
        # If all bands are used, get only the RGB bands for WandB image logging
        if x.shape[2] > 3:
            x = x[:, :, S2_ALL_12BANDS["true_color_zero_index"]]
        # Select the first revisit
        x = x[:, 0, :]

        # Pad with empty revisit dimension: (batch_size, 1, channels, height, width)
        x = x[:, None, :]

        # Normalisation on the channel axis:
        # Add the mean and multiply by the standard deviation
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if self.device is None else self.device
        x += torch.as_tensor(self.mean[None, None, ..., None, None]).to(device)
        x *= torch.as_tensor(self.std[None, None, ..., None, None]).to(device)

        # Convert to float, and scale to [0, 1]:
        x = x.float()
        x /= torch.max(x)
        torch.clamp_(x, 0, 1)

        # Upscale to the output size:
        x = self.resize(x)  # upscale (..., T, C, H_o, W_o)
        return x
