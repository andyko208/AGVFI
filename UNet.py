import math
import numpy as np
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet_3D import SEGating

__all__ = ['UNet_3D_3D']


def joinTensors(X1, X2, type="concat"):
    """
    Joins two tensors along the channel dimension.

    Args:
        X1 (torch.Tensor): The first tensor.
        X2 (torch.Tensor): The second tensor.
        type (str, optional): The type of join operation.
                              "concat" for concatenation, "add" for element-wise addition.
                              Defaults to "concat".

    Returns:
        torch.Tensor: The joined tensor. Returns X1 if type is not "concat" or "add".
    """
    if type == "concat":
        return torch.cat([X1, X2], dim=1)
    elif type == "add":
        return X1 + X2
    else:
        return X1


class Conv_2d(nn.Module):
    """
    2D Convolutional layer wrapper.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Defaults to 0.
        bias (bool, optional): If True, adds a learnable bias to the output. Defaults to False.
        batchnorm (bool, optional): If True, adds BatchNorm2d after convolution. Defaults to False.
    """

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=False, batchnorm=False):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                            stride=stride, padding=padding, bias=bias)]
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_ch))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass for the 2D convolution."""
        return self.conv(x)


class upConv3D(nn.Module):
    """
    3D Up-Convolutional layer for UNet decoder.

    Supports two upsampling modes:
    - "transpose": Uses ConvTranspose3d.
    - "upsample": Uses trilinear upsampling followed by a 1x1x1 Conv3d.

    Includes an SEGating module after the up-convolution.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        kernel_size (int or tuple): Size of the convolving kernel for ConvTranspose3d.
        stride (int or tuple): Stride for ConvTranspose3d.
        padding (int or tuple): Padding for ConvTranspose3d.
        upmode (str, optional): "transpose" or "upsample". Defaults to "transpose".
        batchnorm (bool, optional): If True, adds BatchNorm3d after SEGating. Defaults to False.
    """

    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, upmode="transpose", batchnorm=False):
        super().__init__()
        self.upmode = upmode

        if self.upmode == "transpose":
            self.upconv = nn.ModuleList(
                [nn.ConvTranspose3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
                 SEGating(out_ch),
                 ]
            )

        else:
            self.upconv = nn.ModuleList(
                [nn.Upsample(mode='trilinear', scale_factor=(1, 2, 2), align_corners=False),
                 nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1),
                 SEGating(out_ch)
                 ]
            )

        if batchnorm:
            self.upconv += [nn.BatchNorm3d(out_ch)]

        self.upconv = nn.Sequential(*self.upconv)

    def forward(self, x):
        """Forward pass for the 3D up-convolution."""
        return self.upconv(x)


class Conv_3d(nn.Module):
    """
    3D Convolutional layer with SEGating.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Zero-padding added to all sides of the input. Defaults to 0.
        bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.
        batchnorm (bool, optional): If True, adds BatchNorm3d after SEGating. Defaults to False.
    """

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True, batchnorm=False):
        super().__init__()
        layers = [
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias),
            SEGating(out_ch)
        ]
        if batchnorm:
            layers.append(nn.BatchNorm3d(out_ch))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass for the 3D convolution."""
        return self.conv(x)


class upConv2D(nn.Module):
    """
    2D Up-Convolutional layer.

    Supports two upsampling modes:
    - "transpose": Uses ConvTranspose2d.
    - "upsample": Uses bilinear upsampling followed by a 1x1 Conv2d.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        kernel_size (int or tuple): Size of the convolving kernel for ConvTranspose2d.
        stride (int or tuple): Stride for ConvTranspose2d.
        padding (int or tuple): Padding for ConvTranspose2d.
        upmode (str, optional): "transpose" or "upsample". Defaults to "transpose".
        batchnorm (bool, optional): If True, adds BatchNorm2d. Defaults to False.
    """

    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, upmode="transpose", batchnorm=False):
        super().__init__()
        self.upmode = upmode

        if self.upmode == "transpose":
            self.upconv = [nn.ConvTranspose2d(
                in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)]

        else:
            self.upconv = [
                nn.Upsample(mode='bilinear', scale_factor=2,
                            align_corners=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)
            ]

        if batchnorm:
            self.upconv += [nn.BatchNorm2d(out_ch)]

        self.upconv = nn.Sequential(*self.upconv)

    def forward(self, x):
        """Forward pass for the 2D up-convolution."""
        return self.upconv(x)


class UNet_3D_3D(nn.Module):
    """
    3D UNet architecture for video processing, specifically designed as a decoder part
    of a larger model (e.g., for video frame interpolation or inpainting).

    This network takes 3D feature maps from an encoder and upsamples them through
    a series of 3D transposed convolutions (or upsampling + 1x1 conv) and 2D convolutions.
    It's designed to output one or more video frames.

    Args:
        block: This argument is unused in the current implementation but might have
               been intended for specifying an encoder block.
        n_inputs (int): Number of input frames/features (used to determine `t_size` if not hardcoded).
                        Currently, `t_size` is hardcoded to 17.
        n_outputs (int): Number of output frames to generate.
        batchnorm (bool, optional): Whether to use batch normalization in convolutional layers.
                                   Defaults to False.
        joinType (str, optional): How to join features from skip connections if they were used
                                  (e.g., "concat" or "add"). This affects the `growth` factor.
                                  Defaults to "concat".
        upmode (str, optional): The upsampling mode for `upConv3D` and `upConv2D` layers.
                                Can be "transpose" or "upsample". Defaults to "transpose".
    """

    def __init__(self, block, n_inputs, n_outputs, batchnorm=False, joinType="concat", upmode="transpose"):
        super().__init__()

        # Number of features at different decoder stages
        nf = [512, 256, 128, 64]
        out_channels = 3 * n_outputs  # 3 channels (RGB) per output frame
        self.joinType = joinType
        self.n_outputs = n_outputs

        growth = 2 if joinType == "concat" else 1
        self.t_size = 17
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.decoder = nn.Sequential(
            upConv3D(nf[0], nf[1], kernel_size=(3, 3, 3), stride=(
                1, 1, 1), padding=(1, 1, 1), upmode=upmode, batchnorm=batchnorm),
            upConv3D(nf[1]*growth, nf[2], kernel_size=(3, 4, 4), stride=(1,
                     2, 2), padding=(1, 1, 1), upmode=upmode, batchnorm=batchnorm),
            upConv3D(nf[2]*growth, nf[3], kernel_size=(3, 4, 4), stride=(1,
                     2, 2), padding=(1, 1, 1), upmode=upmode, batchnorm=batchnorm),
            upConv3D(nf[3]*growth, nf[3], kernel_size=(3, 3, 3), stride=(1,
                     1, 1), padding=(1, 1, 1), upmode=upmode, batchnorm=batchnorm),
            upConv3D(nf[3]*growth, nf[3], kernel_size=(3, 4, 4), stride=(1,
                     2, 2), padding=(1, 1, 1), upmode=upmode, batchnorm=batchnorm),

        )

        self.convs = nn.Sequential(
            nn.Conv2d(nf[1]*self.t_size, nf[1]*self.t_size,
                      kernel_size=3, stride=1, padding=1),
            nn.Conv2d(nf[2]*self.t_size, nf[2]*self.t_size,
                      kernel_size=3, stride=1, padding=1),
            nn.Conv2d(nf[3]*self.t_size, nf[3]*self.t_size,
                      kernel_size=3, stride=1, padding=1),
            nn.Conv2d(nf[3]*self.t_size, nf[3]*self.t_size,
                      kernel_size=3, stride=1, padding=1),
            nn.Conv2d(nf[3]*self.t_size, nf[3]*self.t_size,
                      kernel_size=3, stride=1, padding=1),
        )

        self.feature_fuse = Conv_2d(
            nf[3]*self.t_size, nf[3], kernel_size=1, stride=1, batchnorm=batchnorm)

        self.outconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(nf[3], out_channels, kernel_size=7, stride=1, padding=0)
        )

    def forward(self, lth, vid_feat):
        """
        Forward pass of the UNet_3D_3D decoder.

        Args:
            lth (int): Layer index, indicating which part of the decoder to use.
                       If lth is 5, it's the final output stage.
            vid_feat (torch.Tensor): Video features from the previous stage or encoder.
                                     Shape varies depending on the layer.

        Returns:
            torch.Tensor or list[torch.Tensor]:
                - If lth == 5, returns a list of output image tensors (split by channels).
                - Otherwise, returns the processed 3D feature map for the next decoder stage.
        """
        if lth == 5:
            # Final output stage
            # Concatenate along channel dim after unbinding temporal dim
            dx_out = torch.cat(torch.unbind(vid_feat, 2), 1)
            out = self.lrelu(self.feature_fuse(dx_out))
            out = self.outconv(out)
            # Split into N output images
            out = torch.split(out, dim=1, split_size_or_sections=3)
            return out
        else:
            # Intermediate decoder stage
            # Apply 3D transposed convolution (upsampling) and Feature Gating
            feat_map_3d = self.lrelu(self.decoder[lth](vid_feat))

            # Reshape and apply 2D convolution
            # Combines temporal and channel dimensions, then processes with 2D conv
            batch, v_dim, t_size, h, w = feat_map_3d.shape
            combined_feat = feat_map_3d.reshape(batch, v_dim * t_size, h, w)
            feat_map_2d = self.convs[lth](combined_feat)
            feat_map_2d = feat_map_2d.reshape(
                batch, v_dim, t_size, h, w)  # Reshape back
            return feat_map_2d
