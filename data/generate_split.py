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


def retrieve_dataset_slide_info(feature_root_dir, dataset_identifiers):
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
            organ_identifier = [v for v in identifier.split("/") if len(v) > 0]
            organ_identifier = "-".join(organ_identifier)
            slide_info_list = ds.CPTAC.retrieve_labels(
                slide_names,
                f"{PWD}/data/clinical/{organ_identifier}.json",
            )
        else:
            assert False
        sample_info_per_dataset[identifier] = slide_info_list
    return sample_info_per_dataset


def per_dataset_stratified_split(config, dataset_sample_info):
    def retrieve_subset(subset_info, dataset_sample_info):
        all_subsets = []
        for subset in subset_info:
            label_mapping = subset["labels"]
            label_mapping = {k.lower(): int(v) for k, v in label_mapping.items()}

            samples = dataset_sample_info[subset["identifier"]]
            # filter samples with labels that are not within selected
            # set out (contained as keys within `label_mapping`)
            samples = [v for v in samples if v[1] in label_mapping]

            sample_codes, sample_labels = list(zip(*samples))
            sample_labels = [label_mapping[v] for v in sample_labels]
            all_subsets.append([sample_codes, sample_labels])
        return all_subsets

    # stratified split for each sub dataset first
    # then merge the split per each dataset to create the cv splits

    subset_info = config["train-valid"]
    splits = [{"train": [], "valid": []} for _ in range(num_splits)]

    subset_info = retrieve_subset(subset_info, dataset_sample_info)
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
        subset_info = retrieve_subset(subset_info, dataset_sample_info)
        subset_info = [
            list(zip(sample_codes, sample_labels))
            for sample_codes, sample_labels in subset_info
        ]
        subset_info = flatten_list(subset_info)
        for subsplit in splits:
            subsplit["tests"] = subset_info
    return splits


dataset_identifiers = [
    "tcga/lung/ffpe/lscc",
    "tcga/lung/frozen/lscc",
    "tcga/lung/ffpe/luad",
    "tcga/lung/frozen/luad",
    "cptac/lung/luad",
    "cptac/lung/lscc",
    # "tcga/breast/ffpe",
    # "tcga/breast/frozen",
    # "tcga/kidney/ffpe",
    # "tcga/kidney/frozen",
]
PWD = "/mnt/storage_0/workspace/h2t/"
feature_root_dir = "/mnt/storage_0/workspace/h2t/experiments/local/features/[SWAV]-[mpp=0.50]-[512-256]/"

num_splits = 5
dataset_sample_info = retrieve_dataset_slide_info(
    feature_root_dir, dataset_identifiers
)

# %%
config = load_yaml(f"{PWD}/data/config.yaml")
# for split_name, split_info in config.items():
#     splits = per_dataset_stratified_split(split_info)
#     break
# %%
print("here")

old_subset_info = joblib.load("splits/subset_info.dat")
old_mapping = {
    "TCGA-LSCC": "tcga/lung/ffpe/lscc",
    "TCGA-LSCC-Frozen": "tcga/lung/frozen/lscc",
    "TCGA-LUAD": "tcga/lung/ffpe/luad",
    "TCGA-LUAD-Frozen": "tcga/lung/frozen/luad",
    "CPTAC-LUAD": "cptac/lung/luad",
    "CPTAC-LSCC": "cptac/lung/lscc",
}
for k, v in old_mapping.items():
    # print(len(old_subset_info[k]))
    # print(len(dataset_sample_info[v]))
    flags = [
        sample[0][1] == old_subset_info[k][i][0][1]
        for i, sample in enumerate(old_subset_info[k])
    ]
    print(np.sum(flags), len(old_subset_info[k]))

# %%
split_code = "[luad-lusc]_train=tcga_test=cptac"
old_split_root = "/mnt/storage_0/workspace/h2t/data/splits/"
old_splits = joblib.load(f"{old_split_root}/{split_code}.dat")

new_splits = per_dataset_stratified_split(config[split_code])
# %%
for split_idx in range(num_splits):
    old_split = old_splits[split_idx][1]
    new_split = new_splits[split_idx]

    old_ids = [v[0][1] for v in old_split["tcga-train"]]
    new_ids = [v[0][1] for v in new_split["train"]]
    flags = [v in old_ids for v in new_ids]
    print(len(old_ids))
    print(len(new_ids))
    print(np.sum(flags))
    print("=")
# %%