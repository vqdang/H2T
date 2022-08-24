import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding2DImage(nn.Module):
    def __init__(self, ch=None):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super().__init__()
        channels = int(np.ceil(ch / 2))
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (NHWC)
        """
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y
        emb[None, :, :, :orig_ch].repeat(batch_size, 1, 1, 1)
        return emb


class PositionalEncoding2DList(nn.Module):
    def __init__(self, ch=None):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super().__init__()
        channels = int(np.ceil(ch / 2))
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, positions):
        """
        :param tensor: A 3d tensor of size (Batch Size x Num Positions x 2)
        """
        batch_size = positions.shape[0]
        inv_freq = torch.stack([self.inv_freq] * batch_size)
        sin_inp_x = torch.einsum("bi,bj->bij", positions[:, :, 0], inv_freq)
        sin_inp_y = torch.einsum("bi,bj->bij", positions[:, :, 1], inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.cat([emb_x, emb_y], dim=-1)
        return emb



class OnehotEncoding(nn.Module):
    """Assume input is NHWC"""
    def __init__(self, max_value):
        super().__init__()
        self.max_value = max_value

    def forward(self, x):
        x = x.type(torch.int64)
        x = F.one_hot(x, self.max_value)
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return x.type(torch.float32)


class SineEncoding(nn.Module):

    def __init__(self, num_embedded, max_values):
        super().__init__()
        position = torch.arange(max_values).unsqueeze(1)
        div_term = torch.arange(0, num_embedded, 2)
        div_term = torch.exp(div_term * (-math.log(10000.0) / num_embedded))
        pe = torch.zeros(max_values, num_embedded)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        n, h, w, c = x.shape
        assert c == 1
        x = x.type(torch.int64)
        x = 1 + self.pe[x[...,0]]
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return x
