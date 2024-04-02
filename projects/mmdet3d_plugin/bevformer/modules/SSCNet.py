#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: Xu Yan
@File: SSCNet.py
@Time: 2023/3/28 11:49
'''
import torch
import torch.nn as nn


class PixelShuffle3D(nn.Module):
    """
    3D pixelShuffle
    """
    def __init__(self, upscale_factor):
        """
        :param upscale_factor: int
        """
        super(PixelShuffle3D, self).__init__()

        self.upscale_factor = upscale_factor

    def forward(self, inputs):

        batch_size, channels, in_depth, in_height, in_width = inputs.size()

        channels //= self.upscale_factor ** 3

        out_depth = in_depth * self.upscale_factor
        out_height = in_height * self.upscale_factor
        out_width = in_width * self.upscale_factor

        input_view = inputs.contiguous().view(
            batch_size, channels, self.upscale_factor, self.upscale_factor, self.upscale_factor,
            in_depth, in_height, in_width)

        shuffle_out = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return shuffle_out.view(batch_size, channels, out_depth, out_height, out_width)


class SSCNet(nn.Module):
    def __init__(self, input_dim, nPlanes, classes):
        super().__init__()
        # Block 1
        self.b1_conv1 = nn.Sequential(nn.Conv3d(input_dim, 16, 3, 1, padding=1), nn.BatchNorm3d(16), nn.ReLU())
        self.b1_conv2 = nn.Sequential(nn.Conv3d(16, nPlanes[0], 3, 1, padding=1), nn.BatchNorm3d(nPlanes[0]), nn.ReLU())
        self.b1_conv3 = nn.Sequential(nn.Conv3d(nPlanes[0], nPlanes[0], 3, 1, padding=1), nn.BatchNorm3d(nPlanes[0]),
                                      nn.ReLU())
        self.b1_res = nn.Sequential(nn.Conv3d(16, nPlanes[0], 3, 1, padding=1), nn.BatchNorm3d(nPlanes[0]), nn.ReLU())
        self.pool1 = nn.Sequential(nn.MaxPool3d(2, 2))

        # Block 2
        self.b2_conv1 = nn.Sequential(nn.Conv3d(nPlanes[0], nPlanes[1], 3, 1, padding=1), nn.BatchNorm3d(nPlanes[1]),
                                      nn.ReLU())
        self.b2_conv2 = nn.Sequential(nn.Conv3d(nPlanes[1], nPlanes[1], 3, 1, padding=1), nn.BatchNorm3d(nPlanes[1]),
                                      nn.ReLU())
        self.b2_res = nn.Sequential(nn.Conv3d(nPlanes[0], nPlanes[1], 3, 1, padding=1), nn.BatchNorm3d(nPlanes[1]),
                                    nn.ReLU())

        # Block 3
        self.b3_conv1 = nn.Sequential(nn.Conv3d(nPlanes[1], nPlanes[2], 3, 1, padding=1), nn.BatchNorm3d(nPlanes[2]),
                                      nn.ReLU())
        self.b3_conv2 = nn.Sequential(nn.Conv3d(nPlanes[2], nPlanes[2], 3, 1, padding=1), nn.BatchNorm3d(nPlanes[2]),
                                      nn.ReLU())

        # Block 4
        self.b4_conv1 = nn.Sequential(nn.Conv3d(nPlanes[2], nPlanes[3], 3, 1, dilation=2, padding=2),
                                      nn.BatchNorm3d(nPlanes[3]), nn.ReLU())
        self.b4_conv2 = nn.Sequential(nn.Conv3d(nPlanes[3], nPlanes[3], 3, 1, dilation=2, padding=2),
                                      nn.BatchNorm3d(nPlanes[3]), nn.ReLU())

        # Block 5
        self.b5_conv1 = nn.Sequential(nn.Conv3d(nPlanes[3], nPlanes[4], 3, 1, dilation=2, padding=2),
                                      nn.BatchNorm3d(nPlanes[4]), nn.ReLU())
        self.b5_conv2 = nn.Sequential(nn.Conv3d(nPlanes[4], nPlanes[4], 3, 1, dilation=2, padding=2),
                                      nn.BatchNorm3d(nPlanes[4]), nn.ReLU())

        # Prediction
        self.pre_conv1 = nn.Sequential(
            nn.Conv3d(nPlanes[2] + nPlanes[3] + nPlanes[4], int((nPlanes[2] + nPlanes[3] + nPlanes[4]) / 3 * 2), 1, 1), \
            nn.BatchNorm3d(int((nPlanes[2] + nPlanes[3] + nPlanes[4]) / 3 * 2)), nn.ReLU())
        self.pre_conv2 = nn.Sequential(nn.Conv3d(int((nPlanes[2] + nPlanes[3] + nPlanes[4]) / 3 * 2), classes, 1, 1))

        # # Upsample
        # self.upsample = nn.Sequential(nn.Conv3d(in_channels=classes, out_channels=classes * 8, kernel_size=1, stride=1),
        #                               nn.BatchNorm3d(classes * 8), nn.ReLU(),
        #                               PixelShuffle3D(upscale_factor=2))

    def forward(self, x):
        # Block 1
        x = self.b1_conv1(x)
        res_x = self.b1_res(x)
        x = self.b1_conv2(x)
        x = self.b1_conv3(x)
        x = x + res_x

        # Block 2
        res_x = self.b2_res(x)
        x = self.b2_conv1(x)
        x = self.b2_conv2(x)
        x = x + res_x

        # Block 3
        b3_x1 = self.b3_conv1(x)
        b3_x2 = self.b3_conv2(b3_x1)
        b3_x = b3_x1 + b3_x2

        # Block 4
        b4_x1 = self.b4_conv1(b3_x)
        b4_x2 = self.b4_conv2(b4_x1)
        b4_x = b4_x1 + b4_x2

        # Block 5
        b5_x1 = self.b5_conv1(b4_x)
        b5_x2 = self.b5_conv2(b5_x1)
        b5_x = b5_x1 + b5_x2

        # Concat b3,b4,b5
        x = torch.cat((b3_x, b4_x, b5_x), dim=1)

        # Prediction
        x = self.pre_conv1(x)
        x = self.pre_conv2(x)

        # Upsample
        # x = self.upsample(x)
        return x