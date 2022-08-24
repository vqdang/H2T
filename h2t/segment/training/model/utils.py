import math
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# from run_utils.callbacks.serialize import fig2data

####
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
        
####
def find_optimal_roc_cutoff(fpr, tpr, thresholds):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    Returns
    -------     
    list type, with optimal cutoff value
        
    """
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold

####
def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure(figsize=(8, 6))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate' , fontsize=18)
    plt.title('Receiver operating characteristic example', fontsize=20)
    plt.legend(loc="lower right")
    ax = plt.gca()
    ax.xaxis.set_tick_params(labelsize='x-large')
    ax.yaxis.set_tick_params(labelsize='x-large')
    img = fig2data(plt.gcf())
    plt.close()
    return img
