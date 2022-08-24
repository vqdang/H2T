import enum
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)


from .hopfield import HopfieldPooling, HopfieldLayer, Hopfield
from .resnet_swav import resnet50 as swav_resnet50
from .utils import PositionalEncoding2D

# ! internal debug import
# from backbone.resnet import resnet50
# from hopfield import HopfieldPooling, HopfieldLayer, Hopfield
# from resnet_swav import resnet50 as swav_resnet50
# from utils import PositionalEncoding2D

from torchvision.models.resnet import Bottleneck as ResNetBottleneck
from torchvision.models.resnet import ResNet
from torch.utils.checkpoint import checkpoint as mem_checkpoint


def upsample2x(feat):
    return F.interpolate(
        feat, scale_factor=2, mode="bilinear", align_corners=False
    )


def crop_op(x, cropping, data_format="NCHW"):
    """Center crop image
    Args:
        x: input image
        cropping: the substracted amount
        data_format: choose either `NCHW` or `NHWC`
    """
    crop_t = cropping[0] // 2
    crop_b = cropping[0] - crop_t
    crop_l = cropping[1] // 2
    crop_r = cropping[1] - crop_l
    if data_format == "NCHW":
        x = x[:, :, crop_t:-crop_b, crop_l:-crop_r]
    else:
        x = x[:, crop_t:-crop_b, crop_l:-crop_r, :]
    return x


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
        if pretrained is not None:
            (
                missing_keys, unexpected_keys
            ) = model.load_state_dict(pretrained, strict=False)
        if num_input_channels != 3:
            model.conv1 = nn.Conv2d(
                num_input_channels, 64, 7, stride=2, padding=3)
        return model

    @staticmethod
    def resnet50M(num_input_channels, pretrained=None):
        model = ResNetExt(ResNetBottleneck, [2, 2, 2, 2])
        if pretrained is not None:
            (
                missing_keys, unexpected_keys
            ) = model.load_state_dict(pretrained, strict=False)
        if num_input_channels != 3:
            model.conv1 = nn.Conv2d(
                num_input_channels, 64, 7, stride=2, padding=3)
        return model


class UpSample2x(nn.Module):
    """Upsample input by a factor of 2.

    Assume input is of NCHW, port FixedUnpooling from TensorPack.
    """

    def __init__(self):
        super(UpSample2x, self).__init__()
        # correct way to create constant within module
        self.register_buffer(
            "unpool_mat", torch.from_numpy(np.ones((2, 2), dtype="float32"))
        )
        self.unpool_mat.unsqueeze(0)

    def forward(self, x):
        input_shape = list(x.shape)
        # unsqueeze is expand_dims equivalent
        # permute is transpose equivalent
        # view is reshape equivalent
        x = x.unsqueeze(-1)  # bchwx1
        mat = self.unpool_mat.unsqueeze(0)  # 1xshxsw
        ret = torch.tensordot(x, mat, dims=1)  # bxcxhxwxshxsw
        ret = ret.permute(0, 1, 2, 4, 3, 5)
        ret = ret.reshape((-1, input_shape[1], input_shape[2] * 2, input_shape[3] * 2))
        return ret


class AttentionBlock(nn.Module):
    def __init__(
                self,
                num_qkv_channels,
                num_output_channels,
                num_projection_channels,
                num_heads=8,
                dropout=0.5
            ):
        super().__init__()
        self.conv_1x1_list = nn.ModuleList([
            nn.Conv2d(v, num_projection_channels, (1, 1), bias=False)
            for v in num_qkv_channels
        ])
        self.conv_1x1_out = nn.Conv2d(
            num_projection_channels, num_output_channels, (1, 1), bias=False)

        self.attention = Hopfield(
            input_size=num_projection_channels,
            output_size=num_output_channels,
            hidden_size=num_projection_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True)
        self.pe = PositionalEncoding2D(ch=num_projection_channels)

    def forward(self, q, k, v):
        # ! check linear transformer so that the attention
        # ! can be scale linearly wrt to input sequence
        def flatten_hw(x):
            # NxSxC
            n, c, h, w = x.shape
            x = x.permute(0, 2, 3, 1)
            x = torch.reshape(x, (n, h*w, c))
            return x

        def retrieve_hw(x):
            n, hw, c = x.size()
            h = w = np.sqrt(hw).astype(np.int32)
            x = torch.reshape(x, (n, h, w, c))
            x = x.permute(0, 3, 1, 2)
            return x

        q, k, v = [
            flatten_hw(self.pe(self.conv_1x1_list[i](d)))
            for i, d in enumerate([q, k, v])
        ]
        x = self.attention((q, k, v))
        q = self.conv_1x1_out(retrieve_hw(q))
        x = retrieve_hw(x) + q
        return x


class FCN_SA(nn.Module):
    def __init__(
                self,
                nr_output_ch=2,
                freeze_encoder=True
            ):
        super().__init__()
        self.freeze_encoder = freeze_encoder
        state_dict = torch.load('pretrained/[SWAV]_800ep_pretrain.tar')
        state_dict = {
            k.replace('module.', ''): v for k, v in state_dict.items()
        }
        self.backbone = ResNetExt.resnet50(
            num_input_channels=3, pretrained=state_dict)

        # *
        img_list = torch.rand([1, 3, 256, 256])
        out_list = self.backbone(img_list)
        # orderd from lores hires
        down_ch_list = [v.shape[1] for v in out_list][::-1]

        self.conv1x1 = None
        if down_ch_list[0] != down_ch_list[1]:  # channel mapping for shortcut
            self.conv1x1 = nn.Conv2d(
                down_ch_list[0], down_ch_list[1], (1, 1), bias=False)

        self.up_list = nn.ModuleList()
        for ch_idx, ch in enumerate(down_ch_list[1:]):
            next_up_ch = ch
            if ch_idx + 2 < len(down_ch_list):
                next_up_ch = down_ch_list[ch_idx+2]

            if ch_idx > 0:
                self.up_list.append(
                    nn.Sequential(
                        nn.BatchNorm2d(ch), nn.ReLU(),
                        nn.Conv2d(ch, next_up_ch, (3, 3), padding=1, bias=False),
                    )
                )
            else:
                self.up_list.append(
                    AttentionBlock(
                        num_qkv_channels=[ch, ch, ch],
                        num_output_channels=next_up_ch,
                        num_projection_channels=int(ch/16),
                    )
                )

        self.clf = nn.Conv2d(next_up_ch, nr_output_ch, (1, 1), bias=True)
        self.upsample2x = UpSample2x()
        return


    @property
    def preproc_func(self):
        return lambda x: x


    def forward(self, img_list):
        n, c, h, w = img_list.shape

        is_freeze = not self.training or self.freeze_encoder
        with torch.set_grad_enabled(not is_freeze):
            # assume output is after each down-sample resolution
            en_list = self.backbone(img_list)

        if self.conv1x1 is not None:
            x = self.conv1x1(en_list[-1])

        en_list = en_list[:-1]
        for idx in range(1, len(en_list)+1):
            y = en_list[-idx]
            x = self.upsample2x(x) + y
            if idx > 1:
                x = self.up_list[idx-1](x)
            else:
                x = self.up_list[idx-1](x, y, y)
        output = self.clf(x)
        return output


    @staticmethod
    def infer_batch(model, batch_data, on_gpu):

        ####
        model.eval()
        device = 'cuda' if on_gpu else 'cpu'

        ####
        img_list = batch_data

        img_list = img_list.to(device).type(torch.float32)
        img_list = img_list.permute(0, 3, 1, 2)  # to NCHW

        # --------------------------------------------------------------
        with torch.no_grad():
            logit_list = model(img_list)
            logit_list = logit_list.permute(0, 2, 3, 1)  # to NHWC
            prob_list = F.softmax(logit_list, -1)

            prob_list = prob_list.permute(0, 3, 1, 2)  # to NCHW
            prob_list = upsample2x(prob_list)
            prob_list = crop_op(prob_list, [512, 512])
            prob_list = prob_list.permute(0, 2, 3, 1)  # to NHWC

        prob_list = prob_list.cpu().numpy()
        return [prob_list]


class FCN_Model(nn.Module):
    def __init__(
        self,
        nr_output_ch=2,
        freeze_encoder=True
    ):
        super(FCN_Model, self).__init__()

        self.freeze_encoder = freeze_encoder
        self.backbone = resnet50(pretrained=False)
        state_dict = torch.load('pretrained/[SWAV]_800ep_pretrain.tar')
        state_dict = {
            k.replace('module.', ''): v for k, v in state_dict.items()}
        (
            missing_keys, unexpected_keys
        ) = self.backbone.load_state_dict(state_dict, strict=False)
        print('Missing: ', missing_keys)
        print('Unexpected: ', unexpected_keys)

        img_list = torch.rand([1, 3, 256, 256])
        out_list = self.backbone(img_list)
        # orderd from lores hires
        down_ch_list = [v.shape[1] for v in out_list][::-1]

        self.conv1x1 = None
        if down_ch_list[0] != down_ch_list[1]:  # channel mapping for shortcut
            self.conv1x1 = nn.Conv2d(down_ch_list[0], down_ch_list[1], (1, 1), bias=False)

        self.uplist = nn.ModuleList()
        for ch_idx, ch in enumerate(down_ch_list[1:]):
            next_up_ch = down_ch_list[ch_idx+2] if ch_idx + 2 < len(down_ch_list) else ch
            self.uplist.append(
                nn.Sequential(
                    nn.BatchNorm2d(ch), nn.ReLU(),
                    nn.Conv2d(ch, next_up_ch, (3, 3), padding=1, bias=False),
                )
            )

        self.clf = nn.Conv2d(next_up_ch, nr_output_ch, (1, 1), bias=True)
        self.upsample2x = UpSample2x()
        return

    def forward(self, img_list):        
        img_list = img_list / 255.0 # scale to 0-1

        is_freeze = not self.training or self.freeze_encoder
        with torch.set_grad_enabled(not is_freeze):
            # assume output is after each down-sample resolution
            en_list = self.backbone(img_list)

        if self.conv1x1 is not None:
            x = self.conv1x1(en_list[-1])

        en_list = en_list[:-1]
        for idx in range(1, len(en_list)+1):
            y = en_list[-idx]
            x = self.upsample2x(x) + y
            x = self.uplist[idx-1](x)
        output = self.clf(x)
        return output

####
def create_model(model_code=None, **kwargs):
    return FCN_SA(**kwargs)
        
if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # for manually testing batch version
    torch.manual_seed(5)
    batch = 1
    #
    img_list = torch.rand([batch, 3, 1024, 1024]).to('cuda')
    model = FCN_Model()
    model.to('cuda')
    output = model(img_list)
    print(output.shape)
    print('here')
