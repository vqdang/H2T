import math
import pathlib
import tarfile
from io import BytesIO

####
import os
import cv2
import imgaug as ia
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from imgaug import augmenters as iaa
from torch.nn.utils.rnn import pad_sequence

from torch.nn.utils.rnn import pad_sequence

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self,
        root_dir,
        sample_info_list,
        run_mode='train',
        setup_augmentor=True,
        selection_dir=None,
        **kwargs
    ):
        """
        """
                        
        self.kwargs = kwargs

        self.root_dir = root_dir
        self.selection_dir = selection_dir

        self.run_mode = run_mode    
        self.sample_info_list = sample_info_list

        self.id = 0
        if setup_augmentor:
            self.setup_augmentor(0, 0)
        return

    def setup_augmentor(self, worker_id, seed):
        self.id = self.id + worker_id
        self.rng = np.random.default_rng(seed)
        return

    def __len__(self):
        return len(self.sample_info_list)

    def load_sequence(self, subject_info):
        subset_code, slide_code = subject_info
        slide_path = f'{self.root_dir}/{subset_code}/{slide_code}'

        features_list = np.load(f'{slide_path}.features.npy', mmap_mode='r')
        position_list = np.load(f'{slide_path}.position.npy', mmap_mode='r')

        if self.selection_dir:
            selection_path = f'{self.selection_dir}/{subset_code}/{slide_code}.npy'
            selections = np.load(selection_path)
            selections = np.array(selections > 0)
            features_list = features_list[selections]
            position_list = position_list[selections]
        features_list = np.squeeze(features_list)
        position_list = np.squeeze(position_list)

        assert len(features_list.shape) == 2
        # * sampling to prevent feeding overlapping patches
        norm_position_list = (
            (position_list - np.min(position_list, axis=0, keepdims=True)) / 256)
        norm_position_list = norm_position_list.astype(np.int32)
        norm_top_left_list = norm_position_list[:, :2]
        w, h = np.max(norm_top_left_list, axis=0)  # montage h, w
        #
        if self.rng.integers(low=0, high=2) == 0:
            sel = (
                (norm_top_left_list[:, 0] % 2 == 0)
                & (norm_top_left_list[:, 1] % 2 == 0)
            )
        else:
            sel = (
                (norm_top_left_list[:, 0] % 2 == 1)
                & (norm_top_left_list[:, 1] % 2 == 1)
            )
        #
        features_list = features_list[sel]
        position_list = norm_position_list[sel]

        # turn this into nr_patch x batch=1 x nr_feat
        return features_list, position_list
        
    def __getitem__(self, idx):
        info, label = self.sample_info_list[idx]
        seq_feat, seq_pos = self.load_sequence(info)
        return seq_feat, seq_pos, float(label)

    def __get_augmentation(self, mode, rng):
        return None, None

    @staticmethod
    def loader_collate_fn(seq_list):  # batch is a list
        # batch first means assuming seq_list has shape batch x time step x dim
        seq_feat_list, seq_pos_list, seq_label_list = zip(*seq_list)
        seq_len_list = [v.shape[0] for v in seq_feat_list]

        seq_label_list = torch.from_numpy(np.array(seq_label_list))

        seq_feat_list = [torch.from_numpy(v) for v in seq_feat_list]
        seq_feat_list = pad_sequence(seq_feat_list, batch_first=True)

        seq_pos_list = [torch.from_numpy(v) for v in seq_pos_list]
        seq_pos_list = pad_sequence(seq_pos_list, batch_first=True)

        seq_len_list = torch.from_numpy(np.array(seq_len_list))

        seq_msk_list = [torch.zeros(v) for v in seq_len_list]
        seq_msk_list = pad_sequence(seq_msk_list, batch_first=True, padding_value=1.0)
        seq_msk_list = seq_msk_list > 0.0 # to bool

        return seq_feat_list, seq_pos_list, seq_len_list, seq_msk_list, seq_label_list
