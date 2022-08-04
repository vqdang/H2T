import logging
import os
import tarfile
from io import BytesIO

import joblib
import numpy as np
import torch
import torch.utils.data as torch_data
import yaml
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from h2t.misc.utils import mkdir, load_yaml, log_info


class FileDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        subject_info_list,
        num_samples_per_subject,
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
        self.num_samples_per_subject = num_samples_per_subject

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

        if self.selection_dir is not None:
            selection_path = f"{self.selection_dir}/{ds_code}/{wsi_code}.npy"
            selections = np.load(selection_path)
            patch_features = patch_features[selections > 0]
        return patch_features

    def _getitem(self, idx):
        sample_info = self.subject_info_list[idx]
        patch_features = self._load(sample_info)

        # ! if sampling size > upper range, over-sampling may happen
        sel = self.rng.integers(
            0, patch_features.shape[0], size=self.num_samples_per_subject
        )
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


def retrieve_dataloader(
    num_samples_per_subject, num_subjects, l2norm, scaler, persistent_workers
):
    ds = FileDataset(
        FEATURE_ROOT_DIR,
        sample_info_list,
        num_samples_per_subject=num_samples_per_subject,
        l2norm=l2norm,
        scaler=scaler,
        selection_dir=None,
    )
    loader = torch_data.DataLoader(
        ds, drop_last=True, batch_size=num_subjects, num_workers=0,
    )
    return loader


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
    parser.add_argument("--SCALER", type=bool, default=True)
    parser.add_argument("--NUM_CLUSTERS", type=int, default=8)

    args = parser.parse_args()
    print(args)

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["JOBLIB_TEMP_FOLDER"] = "exp_output/storage/a100/cache/"
    # os.environ['JOBLIB_TEMP_FOLDER'] = 'exp_output/storage/nima/cache/'

    # * ----
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
    # * ----

    # mkdir(SAVE_DIR)

    from h2t.data.utils import retrieve_dataset_slide_info, retrieve_subset

    # PWD = "/root/local_storage/storage_0/workspace/h2t/h2t/"
    # FEATURE_ROOT_DIR = "/root/dgx_workspace/h2t/features/[SWAV]-[mpp=0.50]-[512-256]/"

    PWD = "/mnt/storage_0/workspace/h2t/h2t/"
    FEATURE_ROOT_DIR = f"{PWD}/experiments/local/features/[SWAV]-[mpp=0.50]-[512-256]/"
    CLINICAL_ROOT_DIR = f"{PWD}/data/clinical/"

    dataset_identifiers = [
        "tcga/lung/ffpe/lscc",
        "tcga/lung/frozen/lscc",
        "tcga/lung/ffpe/luad",
        "tcga/lung/frozen/luad",
        # "cptac/lung/luad",
        # "cptac/lung/lscc",
        # "tcga/breast/ffpe",
        # "tcga/breast/frozen",
        # "tcga/kidney/ffpe",
        # "tcga/kidney/frozen",
    ]

    DATASET_CODE = "tcga-lung-luad-lusc"
    DATASET_CONFIG = load_yaml(
        "/mnt/storage_0/workspace/h2t/h2t/extract/mining/config.yaml"
    )
    DATASET_CONFIG = DATASET_CONFIG[DATASET_CODE]

    dataset_sample_info = retrieve_dataset_slide_info(
        CLINICAL_ROOT_DIR, FEATURE_ROOT_DIR, dataset_identifiers
    )
    sample_info_list, _ = retrieve_subset(DATASET_CONFIG, dataset_sample_info)

    # * Parsing the config
    default_seed = 5
    recipe = load_yaml(
        "/mnt/storage_0/workspace/h2t/h2t/extract/mining/params/spherical_kmean.yaml"
    )
    recipe["batch_size"] = (
        recipe["num_samples_per_subject"] * recipe["num_subjects_per_batch"]
    )
    recipe["random_seed"] = (
        default_seed if recipe["random_seed"] is None else recipe["random_seed"]
    )
    recipe["num_patterns"] = (
        args.NUM_CLUSTERS if recipe["num_patterns"] is None else recipe["num_patterns"]
    )

    # * Instantiate related objects
    rng_seeder = np.random.PCG64(recipe["random_seed"])
    rng_seeder = np.random.Generator(rng_seeder)

    from h2t.extract.mining.mine import (
        ExtendedMiniBatchKMeans as BatchKmeans,
        ExtendedMiniBatchDictionaryLearning as BatchDictionaryLearning,
    )

    if recipe["clustering_method"] == "spherical_kmeans":
        model = BatchKmeans(
            verbose=3,
            n_clusters=recipe["num_patterns"],
            batch_size=recipe["batch_size"],
            **recipe["clustering_kwargs"],
        )
    elif recipe["clustering_method"] == "dictionary":
        model = BatchDictionaryLearning(
            verbose=3,
            n_components=recipe["num_clusters"],
            batch_size=recipe["batch_size"],
            **recipe["clustering_kwargs"],
        )

    # *

    scaler = None
    if args.SCALER:
        batch_size = recipe["batch_size"]
        loader = retrieve_dataloader(
            num_samples_per_subject=recipe["num_samples_per_subject"],
            num_subjects=recipe["num_subjects_per_batch"],
            l2norm=False,
            scaler=None,
            persistent_workers=False,
        )
        scaler = StandardScaler(copy=False)
        loader_pbar = tqdm(iterable=loader, total=len(loader), ascii=True, position=0)
        for batch_idx, batch_data in enumerate(loader_pbar):
            batch_data = batch_data.numpy()
            batch_data = np.reshape(batch_data, [batch_size, -1])
            scaler.partial_fit(batch_data)
        print("here")

    loader = retrieve_dataloader(
        num_samples_per_subject=recipe["num_samples_per_subject"],
        num_subjects=recipe["num_subjects_per_batch"],
        l2norm=True,
        scaler=scaler,
        persistent_workers=True,
    )

    iter_idx = 0
    batch_size = recipe["batch_size"]
    num_epochs = recipe["num_epochs"]
    num_batches = len(loader)
    num_iters = num_batches * num_epochs

    for epoch in range(num_epochs):
        log_info(f"Epoch {epoch}")
        pbar = tqdm(iterable=loader, total=num_batches, ascii=True, position=0)
        for batch_idx, batch_data in enumerate(loader):
            batch_data = batch_data.numpy()
            batch_data = np.reshape(batch_data, [batch_size, -1])
            model, isconverge = model.partial_fit(
                batch_data, check_convergence=[iter_idx, num_iters, batch_size]
            )
            iter_idx += 1

        if isconverge and epoch >= num_epochs:
            break

        mkdir(f"{SAVE_DIR}/[method={idx}]/")
        joblib.dump(model, f"{SAVE_DIR}/[method={idx}]/model.dat")
        if scaler:
            joblib.dump(scaler, f"{SAVE_DIR}/[method={idx}]/scaler.dat")
