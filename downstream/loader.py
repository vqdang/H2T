import math
import os
import pathlib

import cv2
import joblib
import numpy as np
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        sample_info_list,
        run_mode="train",
        setup_augmentor=True,
        selection_dir=None,
        **kwargs,
    ):
        """ """

        self.step_shape = 256
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

    def normalize_positions(self, positions):
        """Quantize/Normalize the positions with respect to the step size.

        Example: Consider a list of positions `[[256, 256], [512, 512], [1024, 1024]]`
            in `(x, y)` coordinates, in term of step of shape `(H, W) = (256, 256)`,
            the above list is first converted to `[[1, 1], [2, 2], [4, 4]]`. Afterward,
            they are shifted to the top-left normalized corner and become
            `[[0, 0], [1, 1], [3, 3]]`.

        """
        normalized_positions = (
            positions - np.min(positions, axis=0, keepdims=True)
        ) / self.step_shape
        normalized_positions = normalized_positions.astype(np.int32)
        return normalized_positions

    def sampling_positions(self, positions, mode):
        """Sampling to prevent feeding overlapping patches.

        Note: This is mainly done to reduce the GPU memory usage.

        """
        normalized_positions = self.normalize_positions(positions)
        normalized_top_left_positions = normalized_positions[:, :2]

        def even_locations(positions):
            sel = (positions[:, 0] % 2 == 0) & (positions[:, 1] % 2 == 0)
            return sel

        def odd_locations(positions):
            sel = (positions[:, 0] % 2 == 1) & (positions[:, 1] % 2 == 1)
            return sel

        if mode == "even":
            sampling_func = even_locations
        elif mode == "odd":
            sampling_func = odd_locations
        else:
            sampling_func = (
                odd_locations
                if self.rng.integers(low=0, high=2) == 0
                else even_locations
            )

        sel = sampling_func(normalized_top_left_positions)
        return sel

    def load_sequence(self, subject_info):
        subset_code, slide_code = subject_info
        slide_path = f"{self.root_dir}/{subset_code}/{slide_code}"

        features_list = np.load(f"{slide_path}.features.npy", mmap_mode="r")
        position_list = np.load(f"{slide_path}.position.npy", mmap_mode="r")

        if self.selection_dir:
            selection_path = f"{self.selection_dir}/{subset_code}/{slide_code}.npy"
            selections = np.load(selection_path)
            selections = np.array(selections > 0)
            features_list = features_list[selections]
            position_list = position_list[selections]
        features_list = np.squeeze(features_list)
        position_list = np.squeeze(position_list)
        assert len(features_list.shape) == 2

        sel = self.sampling_positions(
            position_list, "random" if self.run_mode == "train" else "even"
        )
        normalized_positions = self.normalize_positions(position_list)
        features_list = features_list[sel]
        position_list = normalized_positions[sel]

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
        seq_msk_list = seq_msk_list > 0.0  # to bool

        return seq_feat_list, seq_pos_list, seq_len_list, seq_msk_list, seq_label_list
