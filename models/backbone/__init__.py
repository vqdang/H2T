import enum
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .convnext import ConvNeXt, LayerNorm
from .convnext import Block as ConvNeXtBlock

from torchvision.models.resnet import Bottleneck as ResNetBottleneck
from torchvision.models.resnet import ResNet


def load_weights(model, pretrained):
    checkpoint = torch.load(pretrained, map_location="cpu")
    (
        missing_keys, unexpected_keys
    ) = model.load_state_dict(checkpoint["model"], strict=False)
    print('Missing: ', missing_keys)
    print('Unexpected: ', unexpected_keys)
    return model


class ResNetExt(ResNet):
    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x0 = x = self.conv1(x)
        x0 = x = self.bn1(x)
        x0 = x = self.relu(x)
        x1 = x = self.maxpool(x)
        x1 = x = self.layer1(x)
        x2 = x = self.layer2(x)
        x3 = x = self.layer3(x)
        x4 = x = self.layer4(x)
        return [x0, x1, x2, x3, x4]

    @staticmethod
    def resnet50(num_input_channels, pretrained=None):
        model = ResNetExt(ResNetBottleneck, [3, 4, 6, 3])
        if pretrained:
            model = load_weights(model, pretrained)
        if num_input_channels != 3:
            model.conv1 = nn.Conv2d(
                num_input_channels, 64, 7, stride=2, padding=3)
        return model

    @staticmethod
    def resnet50M(num_input_channels, pretrained=None):
        model = ResNetExt(ResNetBottleneck, [2, 2, 2, 2])
        if pretrained:
            model = load_weights(model, pretrained)
        if num_input_channels != 3:
            model.conv1 = nn.Conv2d(
                num_input_channels, 64, 7, stride=2, padding=3)
        return model


class ConvNeXtExt(ConvNeXt):
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=2, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer

        # strip off classifier
        # self.head = nn.Linear(dims[-1], num_classes)
        # self.apply(self._init_weights)
        # self.head.weight.data.mul_(head_init_scale)
        # self.head.bias.data.mul_(head_init_scale)

    def forward(self, x):
        features = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            features.append(x)
        return features

    @staticmethod
    def convnext_base(pretrained: str = None, **kwargs):
        model = ConvNeXtExt(
            depths=[3, 3, 27, 3],
            dims=[128, 256, 512, 1024],
            drop_path_rate=0.5,
            **kwargs
        )
        if pretrained:
            model = load_weights(model, pretrained)
        return model
