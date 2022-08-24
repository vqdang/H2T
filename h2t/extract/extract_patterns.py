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

from h2t.misc.utils import mkdir, load_yaml, log_info, setup_logger, save_yaml
from h2t.data.utils import retrieve_dataset_slide_info, retrieve_subset

from h2t.extract.utils import load_sample_with_info


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

    def _getitem(self, idx):
        sample_info = self.subject_info_list[idx]
        patch_features = load_sample_with_info(
            self.root_dir,
            sample_info,
            load_positions=False,
            selection_dir=self.selection_dir,
        )

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
    sample_info_list,
    num_samples_per_subject,
    num_subjects,
    l2norm,
    scaler,
    persistent_workers,
    selection_dir,
):
    ds = FileDataset(
        FEATURE_ROOT_DIR,
        sample_info_list,
        num_samples_per_subject=num_samples_per_subject,
        l2norm=l2norm,
        scaler=scaler,
        selection_dir=selection_dir,
    )
    loader = torch_data.DataLoader(
        ds,
        drop_last=True,
        batch_size=num_subjects,
        num_workers=0,
    )
    return loader


def retrieve_scaler(sample_info_list, recipe, selection_dir):
    batch_size = recipe["batch_size"]
    loader = retrieve_dataloader(
        sample_info_list,
        num_samples_per_subject=recipe["num_samples_per_subject"],
        num_subjects=recipe["num_subjects_per_batch"],
        l2norm=False,
        scaler=None,
        persistent_workers=False,
        selection_dir=SELECTION_DIR,
    )
    scaler = StandardScaler(copy=False)
    loader_pbar = tqdm(iterable=loader, total=len(loader), ascii=True, position=0)
    for batch_idx, batch_data in enumerate(loader_pbar):
        batch_data = batch_data.numpy()
        batch_data = np.reshape(batch_data, [batch_size, -1])
        scaler.partial_fit(batch_data)
    return scaler


####
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--METHOD_CODE", type=str)
    parser.add_argument("--SOURCE_DATASET", type=str)
    parser.add_argument("--FEATURE_CODE", type=str)
    parser.add_argument("--NUM_EPOCHS", type=int, default=50)
    parser.add_argument("--SCALER", type=bool, default=False)
    parser.add_argument("--NUM_CLUSTERS", type=int, default=16)

    args = parser.parse_args()
    print(args)

    os.environ["OMP_NUM_THREADS"] = "1"

    # *
    METHOD_CODE = args.METHOD_CODE
    SOURCE_DATASET = args.SOURCE_DATASET
    FEATURE_CODE = args.FEATURE_CODE

    # * ----

    os.environ["JOBLIB_TEMP_FOLDER"] = f"/root/cache/"
    PWD = "/root/local_storage/storage_0/workspace/h2t/h2t/"

    SELECTION_DIR = None
    FEATURE_ROOT_DIR = "/root/dgx_workspace/h2t/features/[SWAV]-[mpp=0.50]-[512-256]/"
    CLINICAL_ROOT_DIR = f"{PWD}/data/clinical/"
    RECIPE_PATH = f"{PWD}/extract/params/spherical_kmean.yaml"

    SAVE_DIR = (
        f"/root/lsf_workspace/projects/atlas/media-v1/clustering/"
        f"{METHOD_CODE}/{SOURCE_DATASET}/{FEATURE_CODE}/"
    )

    mkdir(SAVE_DIR)
    setup_logger(f"{SAVE_DIR}/debug.log")
    # * ---

    dataset_identifiers = [
        # "tcga/lung/ffpe/lscc",
        # "tcga/lung/frozen/lscc",
        # "tcga/lung/ffpe/luad",
        # "tcga/lung/frozen/luad",
        # "cptac/lung/luad",
        # "cptac/lung/lscc",
        "tcga/breast/ffpe",
        "tcga/breast/frozen",
        "tcga/kidney/ffpe",
        "tcga/kidney/frozen",
    ]

    DATASET_CONFIG = load_yaml(f"{PWD}/extract/config.yaml")
    DATASET_CONFIG = DATASET_CONFIG[SOURCE_DATASET]

    dataset_sample_info = retrieve_dataset_slide_info(
        CLINICAL_ROOT_DIR, FEATURE_ROOT_DIR, dataset_identifiers
    )
    sample_info_list, _ = retrieve_subset(DATASET_CONFIG, dataset_sample_info)
    log_info(f"Number of samples: {len(sample_info_list)})")

    # * Parsing the config
    default_seed = 5
    recipe = load_yaml(RECIPE_PATH)
    recipe["batch_size"] = (
        recipe["num_samples_per_subject"] * recipe["num_subjects_per_batch"]
    )
    recipe["random_seed"] = (
        default_seed if recipe["random_seed"] is None else recipe["random_seed"]
    )
    recipe["num_patterns"] = (
        args.NUM_CLUSTERS if recipe["num_patterns"] is None else recipe["num_patterns"]
    )
    recipe["num_epochs"] = (
        args.NUM_EPOCHS if recipe["num_epochs"] is None else recipe["num_epochs"]
    )
    save_yaml(f"{SAVE_DIR}/config.yaml", recipe)

    # * Instantiate related objects
    rng_seeder = np.random.PCG64(recipe["random_seed"])
    rng_seeder = np.random.Generator(rng_seeder)

    from h2t.extract.mine import (
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

    scaler = (
        retrieve_scaler(sample_info_list, recipe, SELECTION_DIR)
        if args.SCALER
        else None
    )

    loader = retrieve_dataloader(
        sample_info_list,
        num_samples_per_subject=recipe["num_samples_per_subject"],
        num_subjects=recipe["num_subjects_per_batch"],
        l2norm=True,
        scaler=scaler,
        persistent_workers=True,
        selection_dir=SELECTION_DIR,
    )

    iter_idx = 0
    batch_size = recipe["batch_size"]
    num_epochs = recipe["num_epochs"]
    num_batches = len(loader)
    num_iters = num_batches * num_epochs

    mkdir(f"{SAVE_DIR}/models/")
    if scaler:
        joblib.dump(scaler, f"{SAVE_DIR}/scaler.dat")

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
        joblib.dump(model, f"{SAVE_DIR}/models/model-{epoch:03d}.dat")
