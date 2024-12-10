import math
import numpy as np
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet_3D import SEGating

__all__ = ['UNet_3D_3D']

def joinTensors(X1 , X2 , type="concat"):

    if type == "concat":
        return torch.cat([X1 , X2] , dim=1)
    elif type == "add":
        return X1 + X2
    else:
        return X1


class Conv_2d(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=False, batchnorm=False):

        super().__init__()
        self.conv = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]

        if batchnorm:
            self.conv += [nn.BatchNorm2d(out_ch)]

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):

        return self.conv(x)

class upConv3D(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, upmode="transpose" , batchnorm=False):

        super().__init__()

        self.upmode = upmode

        if self.upmode=="transpose":
            self.upconv = nn.ModuleList(
                [nn.ConvTranspose3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
                SEGating(out_ch),
                ]
            )

        else:
            self.upconv = nn.ModuleList(
                [nn.Upsample(mode='trilinear', scale_factor=(1,2,2), align_corners=False),
                nn.Conv3d(in_ch, out_ch , kernel_size=1 , stride=1),
                SEGating(out_ch)
                ]
            )

        if batchnorm:
            self.upconv += [nn.BatchNorm3d(out_ch)]

        self.upconv = nn.Sequential(*self.upconv)
        # self.conv2d = Conv_2d(out_ch, out_ch, kernel_size=3, stride=1, batchnorm=batchnorm)
        

    def forward(self, x):

        return self.upconv(x)
        # return self.conv2d(self.upconv(x))

class Conv_3d(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True, batchnorm=False):

        super().__init__()
        self.conv = [nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                    SEGating(out_ch)
                    ]

        if batchnorm:
            self.conv += [nn.BatchNorm3d(out_ch)]

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):

        return self.conv(x)

class upConv2D(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, upmode="transpose" , batchnorm=False):

        super().__init__()

        self.upmode = upmode

        if self.upmode=="transpose":
            self.upconv = [nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)]

        else:
            self.upconv = [
                nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
                nn.Conv2d(in_ch, out_ch , kernel_size=1 , stride=1)
            ]

        if batchnorm:
            self.upconv += [nn.BatchNorm2d(out_ch)]

        self.upconv = nn.Sequential(*self.upconv)

    def forward(self, x):

        return self.upconv(x)


class UNet_3D_3D(nn.Module):
    def __init__(self, block , n_inputs, n_outputs, batchnorm=False , joinType="concat" , upmode="transpose"):
        super().__init__()

        nf = [512 , 256 , 128 , 64]        
        out_channels = 3*n_outputs
        self.joinType = joinType
        self.n_outputs = n_outputs

        growth = 2 if joinType == "concat" else 1
        self.t_size = 17
        self.lrelu = nn.LeakyReLU(0.2, True)

        # unet_3D = importlib.import_module(".resnet_3D" , "model")
        # unet_3D = importlib.import_module("resnet_3D" , "model")
        # if n_outputs > 1:
        #     unet_3D.useBias = True
        # self.encoder = getattr(unet_3D , block)(pretrained=False , bn=batchnorm)            

        self.decoder = nn.Sequential(
            upConv3D(nf[0], nf[1], kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1) , upmode=upmode, batchnorm=batchnorm),
            upConv3D(nf[1]*growth, nf[2], kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1) , upmode=upmode, batchnorm=batchnorm),
            upConv3D(nf[2]*growth, nf[3], kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1) , upmode=upmode, batchnorm=batchnorm),
            upConv3D(nf[3]*growth, nf[3], kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1) , upmode=upmode, batchnorm=batchnorm),
            upConv3D(nf[3]*growth, nf[3], kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1) , upmode=upmode, batchnorm=batchnorm),
            
        )

        self.convs = nn.Sequential(
            nn.Conv2d(nf[1]*self.t_size, nf[1]*self.t_size, kernel_size=3 , stride=1, padding=1),
            nn.Conv2d(nf[2]*self.t_size, nf[2]*self.t_size , kernel_size=3 , stride=1, padding=1),
            nn.Conv2d(nf[3]*self.t_size, nf[3]*self.t_size , kernel_size=3 , stride=1, padding=1),
            nn.Conv2d(nf[3]*self.t_size, nf[3]*self.t_size , kernel_size=3 , stride=1, padding=1),
            nn.Conv2d(nf[3]*self.t_size, nf[3]*self.t_size , kernel_size=3 , stride=1, padding=1),
        )

        # self.feature_fuse = Conv_2d(nf[3]*n_inputs , nf[3] , kernel_size=1 , stride=1, batchnorm=batchnorm)
        self.feature_fuse = Conv_2d(nf[3]*self.t_size , nf[3] , kernel_size=1 , stride=1, batchnorm=batchnorm)

        self.outconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(nf[3], out_channels, kernel_size=7 , stride=1, padding=0) 
        )         

    def forward(self, lth, vid_feat):

        if lth == 5:
            
            dx_out = torch.cat(torch.unbind(vid_feat, 2), 1)

            out = self.lrelu(self.feature_fuse(dx_out))
            out = self.outconv(out)

            out = torch.split(out, dim=1, split_size_or_sections=3)
            # mean_ = mean_.squeeze(2)
            # out = [o+mean_ for o in out]
            return out
        else:
            """
            Feed vid into the decoder layer (3D transpose convolution and Feature Gating)
            to get 3D feature maps.
                Input: batch, v_i_dim, t_size, h, w
                Output: batch, v_i-1_dim, t_size, h, w
            """
            feat_map_3d = self.lrelu(self.decoder[lth](vid_feat))
            # print(f'feat_map_3d: {feat_map_3d.shape}')
            """
            In order to get the feature of I_t, we combine the feature map's 
            temporal and channel dimensions and further process it by 2D convolution.
                Input: batch, v_i-1_dim, t_size, h, w
                Output: batch, v_i-1_dim, t_size, h, w
            """
            batch, v_dim, t_size, h, w = feat_map_3d.shape
            combined_feat = feat_map_3d.reshape(batch, v_dim*t_size, h, w)
            feat_map_2d = self.convs[lth](combined_feat)
            # print(f'feat_map_2d: {feat_map_2d.shape}')
            feat_map_2d = feat_map_2d.reshape(batch, v_dim, t_size, h, w)
            # print(f'feat_map_2d: {feat_map_2d.shape}')

        return feat_map_2d

      