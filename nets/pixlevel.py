# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


class PixLevelModule(nn.Module):
    """
    Deformable Convolution v2 replacing PLAM.

    Instead of fixed-grid convolution, DCNv2 learns:
      - OFFSETS: where to sample (handles irregular lesion shapes)
      - MASKS:   how much to weight each sample point (modulation)

    This is ideal for medical images where lesion boundaries
    are blurred and shapes vary significantly across patients.

    Interface is identical to original PLAM:
        Input:  (B, C, H, W)
        Output: (B, C, H, W)  ← same shape, drop-in replacement
    """

    def __init__(self, in_channels, kernel_size=3):
        super(PixLevelModule, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2  # keeps spatial size same

        # --- Learnable deformable conv weights and bias ---
        # Acts as the main feature transformation (like a normal conv)
        self.weight = nn.Parameter(
            torch.Tensor(in_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.Tensor(in_channels))

        # --- Offset predictor ---
        # Predicts 2 * kernel_size^2 values per spatial location
        # (x-offset and y-offset for each of the k*k kernel points)
        self.offset_conv = nn.Conv2d(
            in_channels,
            out_channels=2 * kernel_size * kernel_size,  # 18 for k=3
            kernel_size=kernel_size,
            padding=self.padding,
            bias=True
        )

        # --- Modulation mask predictor (DCNv2 specific) ---
        # Predicts importance weight [0,1] for each kernel point
        self.mask_conv = nn.Conv2d(
            in_channels,
            out_channels=kernel_size * kernel_size,  # 9 for k=3
            kernel_size=kernel_size,
            padding=self.padding,
            bias=True
        )

        # --- BN + activation after deformable conv ---
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        # --- Weight initialization ---
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        nn.init.zeros_(self.bias)
        # Init offsets to 0 → starts as regular conv, learns deformation
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        # Init mask to 0.5 → uniform importance at start
        nn.init.zeros_(self.mask_conv.weight)
        nn.init.constant_(self.mask_conv.bias, 0.5)

    def forward(self, x):
        # Step 1: Predict where to sample (offsets)
        # offset shape: (B, 2*k*k, H, W)
        offset = self.offset_conv(x)

        # Step 2: Predict how much to weight each point (modulation)
        # mask shape: (B, k*k, H, W), sigmoid → values in [0, 1]
        mask = torch.sigmoid(self.mask_conv(x))

        # Step 3: Apply deformable conv with learned offsets and mask
        # torchvision.ops.deform_conv2d handles the grid sampling internally
        out = deform_conv2d(
            input=x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            padding=self.padding,
            mask=mask       # DCNv2: pass mask for modulation
        )

        # Step 4: Normalize and activate
        out = self.relu(self.bn(out))

        return out