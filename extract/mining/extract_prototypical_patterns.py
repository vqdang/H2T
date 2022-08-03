import multiprocessing as mp

from tqdm import tqdm

import logging
import os
import tarfile

import joblib
import numpy as np
import torch
import torch.utils.data as torch_data
import yaml


from io import BytesIO

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from misc.utils import mkdir


class FileDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        subject_info_list,
        num_samples,
        setup_augmentor=True,
        l2norm=True,
        scaler=None,
        selection_dir=None,
        **kwargs,
    ):
        """ """

        self.kwargs = kwargs

        self.root_dir = root_dir
        self.selection_dir = selection_dir
        self.subject_info_list = subject_info_list

        self.l2norm = l2norm
        self.scaler = scaler

        self.id = 0
        if setup_augmentor:
            self.setup_augmentor(0, 0)
        return

    def setup_augmentor(self, worker_id, seed):
        self.id = self.id + worker_id
        self.rng = np.random.Generator(np.random.PCG64(seed))
        return

    def __len__(self):
        return len(self.subject_info_list)

    def _load(self, sample_info):
        ds_code, wsi_code = sample_info

        feature_path = f"{self.root_dir}/{ds_code}/{wsi_code}.features.npy"
        patch_features = np.load(feature_path, mmap_mode="r")

        if self.selection is not None:
            selection_path = f"{self.selection_dir}/{ds_code}/{wsi_code}.npy"
            selections = np.load(selection_path)
            patch_features = patch_features[selections > 0]
        return patch_features

    def _getitem(self, idx):
        sample_info = self.subject_info_list[idx]
        patch_features = self._load(sample_info)

        # ! if sampling size > upper range, over-sampling may happen
        sel = self.rng.integers(0, patch_features.shape[0], size=self.num_samples)
        patch_features = np.array(patch_features[sel])
        patch_features = np.squeeze(patch_features)

        if self.scaler:
            patch_features = self.scaler.transform(patch_features)

        # must norm if want to do spherical k-mean
        if self.l2norm:
            patch_features /= np.linalg.norm(patch_features, axis=-1, keepdims=True)

        return patch_features

    def __getitem__(self, idx):
        return self._getitem(idx)


####
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--ATLAS_ROOT_DATA", type=str, default="[rootData=tcga]")
    parser.add_argument(
        "--ATLAS_SOURCE_TISSUE", type=str, default="[sourceTissue=Normal+LUAD+LUSC]"
    )
    parser.add_argument(
        "--FEATURE_CODE", type=str, default="[SWAV]-[mpp=0.50]-[512-256]"
    )
    parser.add_argument("--NUM_EPOCHS", type=int, default=25)
    parser.add_argument("--SCALER", type=bool, default=False)

    args = parser.parse_args()
    print(args)

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["JOBLIB_TEMP_FOLDER"] = "exp_output/storage/a100/cache/"
    # os.environ['JOBLIB_TEMP_FOLDER'] = 'exp_output/storage/nima/cache/'

    FEATURE_CODE = args.FEATURE_CODE
    ATLAS_ROOT_DATA = args.ATLAS_ROOT_DATA
    ATLAS_SOURCE_TISSUE = args.ATLAS_SOURCE_TISSUE

    THRESHOLD = 0.5
    WORKSPACE_DIR = "exp_output/storage/a100/"
    # WORKSPACE_DIR = 'exp_output/storage/nima/'
    FEAT_ROOT_DIR = f"{WORKSPACE_DIR}/features/{FEATURE_CODE}/"
    SELECTION_DIR = (
        f"{WORKSPACE_DIR}/features/mpp=0.25/" f"selections-{THRESHOLD:0.2f}/"
    )
    SAVE_DIR = (
        f"{WORKSPACE_DIR}/cluster_filtered-{THRESHOLD:0.2f}/"
        # f'/ablation/[epochs={args.NUM_EPOCHS}]/'
        f"{ATLAS_SOURCE_TISSUE}/{FEATURE_CODE}/{ATLAS_ROOT_DATA}/"
        # f'{WORKSPACE_DIR}/dump/cluster/'
    )

    mkdir(SAVE_DIR)
    logging.basicConfig(
        level=logging.DEBUG,
        format="|%(asctime)s.%(msecs)03d| [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d|%H:%M:%S",
        handlers=[logging.FileHandler(f"{SAVE_DIR}/log.log"), logging.StreamHandler()],
    )

    seed = 5
    rng_seeder = np.random.Generator(np.random.PCG64(seed))

    num_epochs = 50 if args.NUM_EPOCHS is None else args.NUM_EPOCHS
    if "mpp=0.50" in FEATURE_CODE:
        # 256 for mpp=0.5, *4 for mpp=0.25
        num_samples_per_subject = 256
    else:
        # 256 for mpp=0.5, *4 for mpp=0.25
        num_samples_per_subject = 256 * 4

    num_loader_workers = 8
    num_subjects_per_batch = 16
    batch_size = num_subjects_per_batch * num_samples_per_subject

    atlas_construct_template_kwargs = {
        "random_seed": seed,
        "num_clusters": None,
        "cluster_method": "spherical_kmeans",
        "batch_size": batch_size,
        "reassignment_ratio": 0.1,
    }

    # *
    ds = FileDataset(
        FEAT_ROOT_DIR,
        subject_info_list,
        num_samples_per_subject,
        l2norm=False,
        scaler=None,
        selection_dir=SELECTION_DIR,
    )
    loader = torch_data.DataLoader(
        ds,
        drop_last=True,
        batch_size=num_subjects_per_batch,
        num_workers=num_loader_workers,
    )

    scaler = None
    if args.SCALER:
        assert False
        scaler = StandardScaler(copy=False)
        pbar = tqdm(total=len(loader), ascii=True, position=0)
        for batch_idx, batch_data in enumerate(loader):
            batch_data = batch_data.numpy()
            batch_data = np.reshape(batch_data, [batch_size, 2048])
            scaler.partial_fit(batch_data)
            pbar.update()
        pbar.close()

    # *
    ds = FileDataset(
        FEAT_ROOT_DIR,
        subject_info_list,
        num_samples_per_subject,
        l2norm=True,
        scaler=scaler,
        selection_dir=SELECTION_DIR,
    )
    loader = torch_data.DataLoader(
        ds,
        drop_last=True,
        batch_size=num_subjects_per_batch,
        num_workers=num_loader_workers,
        persistent_workers=num_loader_workers > 0,
    )

    for idx, num_clusters in enumerate([8, 16, 32]):

        paramset = atlas_construct_template_kwargs.copy()
        paramset["num_clusters"] = num_clusters
        model = MiniBatchKMeans(
            n_clusters=paramset["num_clusters"],
            batch_size=paramset["batch_size"],
            reassignment_ratio=paramset["reassignment_ratio"],
            verbose=3,
        )
        convergence_context = {}

        iter_idx = 0
        num_batches = len(loader)
        num_iters = num_batches * num_epochs
        for epoch in range(num_epochs):
            logging.info(f"Epoch {epoch}")
            pbar = tqdm(total=num_batches, ascii=True, position=0)
            for batch_idx, batch_data in enumerate(loader):
                batch_data = batch_data.numpy()
                batch_data = np.reshape(batch_data, [batch_size, 2048])
                model, isconverge = model.partial_fitX(
                    batch_data, iter_idx, num_iters, batch_size
                )
                iter_idx += 1
                pbar.update()
            pbar.close()
            # if isconverge and epoch >= num_epochs:
            #     break
        mkdir(f"{SAVE_DIR}/[method={idx}]/")
        joblib.dump(model, f"{SAVE_DIR}/[method={idx}]/model.dat")
        if scaler:
            joblib.dump(scaler, f"{SAVE_DIR}/[method={idx}]/scaler.dat")
