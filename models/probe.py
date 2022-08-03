
import numpy as np
import torch
import torch.nn as nn

from .backbone import ResNetExt
from .encoding import OnehotEncoding, SineEncoding


class ColocalModel(nn.Module):
    def __init__(
        self,
        encode=None,
        encode_kwargs=None,
        pretrained_backbone=None,
    ):
        super().__init__()
        # Normalize over last dimension

        if encode is None:
            num_input_channels = 1
            self.input_encoder = (
                lambda x: (x / encode_kwargs['max_value'])[:, None]  # HW1
            )
        elif encode == 'onehot':
            self.input_encoder = OnehotEncoding(**encode_kwargs)
            num_input_channels = encode_kwargs['max_value']
        elif encode == 'fourier':
            self.input_encoder = SineEncoding(**encode_kwargs)
            num_input_channels = encode_kwargs['num_embedded']
        else:
            assert False, f'Unknown encode mode `{encode}`'

        self.backbone = ResNetExt.resnet50M(num_input_channels, pretrained_backbone)

        # sort from lores to hires
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        return

    def forward(self, img_list):
        with torch.no_grad():
            img_list = self.input_encoder(img_list)

        features = self.backbone(img_list)[-1]
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        return features


class Probe(nn.Module):
    def __init__(
        self,
        num_input_channels=2048,
        num_output_channels=2,
        mode='linear',
        colocal=None,
    ):
        super().__init__()
        assert mode in ['linear', 'non-linear']

        self.colocal = None
        if colocal is not None:
            self.colocal = ColocalModel(**colocal)
            num_input_channels += 2048

        if mode == 'linear':
            self.clf = nn.Sequential(
                nn.BatchNorm1d(num_input_channels),
                nn.Linear(num_input_channels, num_output_channels)
            )
        else:
            self.clf = nn.Sequential(
                nn.BatchNorm1d(num_input_channels),
                nn.Linear(num_input_channels, 128),
                nn.Dropout(p=0.5),
                nn.ReLU(),
                nn.Linear(num_input_channels, num_output_channels)
            )
        return

    def forward(self, feat_list=None, img_list=None):
        if self.colocal is not None and feat_list is not None:
            colocal = self.colocal(img_list)
            feat_list = torch.cat([feat_list, colocal], 1)
        elif self.colocal is not None:
            feat_list = self.colocal(img_list)
        output = self.clf(feat_list)
        return output
