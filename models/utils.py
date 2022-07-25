import math
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from run_utils.callbacks.serialize import fig2data


def upsample2x(feat):
    return F.interpolate(
        feat, scale_factor=2, mode="bilinear", align_corners=False
    )


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


class PositionalEncoding2D(nn.Module):
    def __init__(self, ch=None):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super().__init__()
        channels = int(np.ceil(ch/2))
        self.channels = channels
        inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (NCHW)
        """
        tensor = tensor.permute(0,2,3,1) # to NHWC

        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((x,y,self.channels*2),device=tensor.device).type(tensor.type())
        emb[:,:,:self.channels] = emb_x
        emb[:,:,self.channels:2*self.channels] = emb_y
        emb[None,:,:,:orig_ch].repeat(batch_size, 1, 1, 1)

        enc = tensor + emb
        return enc.permute(0,3,1,2)