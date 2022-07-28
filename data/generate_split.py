# %%
import argparse
import os
import pathlib
from collections import OrderedDict

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold

import dataset as ds
from misc.utils import load_yaml, mkdir, rm_n_mkdir, recur_find_ext, flatten_list


def get_cv_split(Y, num_splits=5, random_state=None, shuffle=True):
    Y = np.array(Y)
    X = np.arange(0, Y.shape[0])  # dummy

    skf = StratifiedKFold(
        n_splits=num_splits, random_state=random_state, shuffle=shuffle
    )
    full_split = []
    for ext_idx, (train_idx, valid_idx) in enumerate(skf.split(X, Y)):
        full_split.append([(ext_idx), [train_idx, valid_idx]])
    return full_split


# %%
dataset_identifiers = [
    "tcga/lung/ffpe/lusc",
    "tcga/lung/frozen/lusc",
    "tcga/lung/ffpe/luad",
    "tcga/lung/frozen/luad",
    # "tcga/breast/ffpe",
    # "tcga/breast/frozen",
    # "tcga/kidney/ffpe",
    # "tcga/kidney/frozen",
    # "cptac/luad/",
    # "cptac/lusc/",
]
feature_root_dir = "/mnt/storage_0/workspace/atlas/exp_output/features/[SWAV]-[mpp=0.50]-[512-256]/temp/"
PWD = "/mnt/storage_0/workspace/h2t/"


def retrieve_dataset_slide_info(feature_root_dir):
    sample_info_per_dataset = {}
    for identifier in dataset_identifiers:
        slide_names = recur_find_ext(
            f"{feature_root_dir}/{identifier}/", [".features.npy"]
        )
        slide_names = [pathlib.Path(v).stem for v in slide_names]
        slide_names = [str(v).replace(".features", "") for v in slide_names]

        if "tcga" in identifier:
            organ_identifier = identifier.split("/")[1]
            slide_info_list = ds.TCGA.retrieve_labels(
                slide_names,
                f"{PWD}/data/clinical/tcga-{organ_identifier}/clinical.tsv",
                f"{PWD}/data/clinical/tcga-{organ_identifier}/slides.json",
            )
            slide_info_list = [[[identifier, v[0]], v[1]] for v in slide_info_list]
        elif "cptac" in identifier:
            slide_info_list = ds.CPTAC.retrieve_labels(
                slide_names,
                f"{PWD}/data/clinical/tcga-{organ_identifier}/sample.tsv",
            )
        else:
            assert False
        sample_info_per_dataset[identifier] = slide_info_list
    return sample_info_per_dataset


def per_dataset_stratified_split(config):

    def retrieve_subset(subset_info, sample_info_per_dataset):
        all_subsets = []
        for subset in subset_info:
            label_mapping = subset["labels"]
            label_mapping = {k.lower(): int(v) for k, v in label_mapping.items()}

            samples = sample_info_per_dataset[subset["identifier"]]
            sample_codes, sample_labels = list(zip(*samples))
            sample_labels = [label_mapping[v] for v in sample_labels]
            all_subsets.append([sample_codes, sample_labels])
        return all_subsets

    # stratified split for each sub dataset first
    # then merge the split per each dataset to create the cv splits

    subset_info = config["train-valid"]
    splits = [{"train": [], "valid": []} for _ in range(num_splits)]

    subset_info = retrieve_subset(subset_info, sample_info_per_dataset)
    for sample_codes, sample_labels in subset_info:
        subset_splits = get_cv_split(
            sample_labels, num_splits=num_splits, random_state=5, shuffle=True
        )

        samples = np.array(list(zip(sample_codes, sample_labels)))
        for split_idx, (train_idx, valid_idx) in subset_splits:
            splits[split_idx]["train"].extend(samples[train_idx])
            splits[split_idx]["valid"].extend(samples[valid_idx])

    # stratified split for each sub dataset first
    # then merge the split per each dataset to create the cv splits
    if "test" in config:
        subset_info = config["test"]
        subset_info = retrieve_subset(subset_info, sample_info_per_dataset)
        subset_info = flatten_list(subset_info)
        for split_idx in splits:
            splits[split_idx]["tests"] = subset_info
    return

# %%
# a = joblib.load("splits/subset_info.dat")

num_splits = 5
sample_info_per_dataset = retrieve_dataset_slide_info(feature_root_dir)
config = load_yaml(f"{PWD}/data/config.yaml")
for split_name, split_info in config.items():
    splits = per_dataset_stratified_split(split_info)

print("here")
