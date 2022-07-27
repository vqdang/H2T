import csv
import glob
import importlib
import os
import pathlib
import re
from collections import OrderedDict

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch.utils.data

import dataset
from misc.utils import mkdir, rm_n_mkdir


####
def get_cv_split(Y, nr_splits=5, random_state=None, shuffle=True):
    from sklearn.model_selection import StratifiedKFold

    Y = np.array(Y)
    X = np.arange(0, Y.shape[0])  # dummy

    skf = StratifiedKFold(
        n_splits=nr_splits, random_state=random_state, shuffle=shuffle
    )
    full_split = []
    for ext_idx, (train_idx, valid_idx) in enumerate(skf.split(X, Y)):
        full_split.append([(ext_idx), [train_idx, valid_idx]])
    return full_split


####
def get_nested_cv_split(
    path_list,
    label_list,
    nr_external_splits=5,
    nr_internal_splits=5,
    random_state=None,
    shuffle=True,
):
    from sklearn.model_selection import StratifiedKFold

    assert len(path_list) == len(label_list)
    Y = np.array(label_list)
    X = np.array(path_list)  # dummy

    external_skf = StratifiedKFold(
        n_splits=nr_external_splits, random_state=random_state, shuffle=shuffle
    )
    internal_skf = StratifiedKFold(
        n_splits=nr_internal_splits, random_state=random_state, shuffle=shuffle
    )

    ext_idx = 0
    full_split = []
    for train_idx, valid_idx in external_skf.split(X, Y):
        X1_train = X[train_idx]
        Y1_train = Y[train_idx]
        X1_valid = X[valid_idx]
        Y1_valid = Y[valid_idx]

        int_idx = 0
        for train_idy, valid_idy in internal_skf.split(X1_train, Y1_train):
            X2_train = X1_train[train_idy]
            X2_valid = X1_train[valid_idy]

            Y2_train = Y1_train[train_idy]
            Y2_valid = Y1_train[valid_idy]

            split_info = {
                "train": np.array(list(zip(X2_train.tolist(), Y2_train.tolist()))),
                "valid": np.array(list(zip(X2_valid.tolist(), Y2_valid.tolist()))),
                "test": np.array(list(zip(X1_valid.tolist(), Y1_valid.tolist()))),
            }
            full_split.append([(ext_idx, int_idx), split_info])

            # [Y[v] for v in split_info] # test indexing
            # split_info = [set(v.tolist()) for v in split_info]
            # assert len(set.intersection(*split_info)) == 0

            int_idx += 1
        ext_idx += 1
    return full_split


####
def get_subject_info(
    root_dir, dataset_code_list, feat_code, shape_code, mpp_code, group_mode=None
):
    """
    Assuming all slide has unique name code!
    """
    slide_code_list = []
    slide_label_list = []
    slide_data_code_list = []
    for dataset_code in dataset_code_list:
        slide_dir = "%s/%s/[%s]-[%s]-[%s]/" % (
            root_dir,
            feat_code,
            dataset_code,
            shape_code,
            mpp_code,
        )
        if "TCGA" in dataset_code:
            sub_slide_path, sub_slide_label = dataset.TCGA.load(slide_dir)
        elif "CPTAC-LUAD" in dataset_code:
            sub_slide_path, sub_slide_label = dataset.CPTAC.load(
                slide_dir, "CPTAC-LUAD"
            )
        elif "CPTAC-LSCC" in dataset_code:
            sub_slide_path, sub_slide_label = dataset.CPTAC.load(
                slide_dir, "CPTAC-LSCC"
            )
        else:
            assert False

        if "LSCC" in dataset_code:
            sub_slide_label = np.array(sub_slide_label)
            sub_slide_label[sub_slide_label > 0] = 1
        elif "LUAD" in dataset_code:
            sub_slide_label = np.array(sub_slide_label)
            sub_slide_label[sub_slide_label > 0] = 2

        sub_slide_code = [pathlib.Path(v).stem for v in sub_slide_path]
        slide_data_code_list.extend([dataset_code] * len(sub_slide_code))
        slide_code_list.extend(sub_slide_code)
        slide_label_list.extend(sub_slide_label)

    def find_unique_idx(slide_code_list):
        seen = {}
        uniq = []  # preserve order
        for idx, x in enumerate(slide_code_list):
            if x not in seen:
                uniq.append(x)
                seen[x] = idx
        return list(seen.values())

    unique_slide_code_idx = find_unique_idx(slide_code_list)
    # get unique counter part
    slide_code_list = [slide_code_list[v] for v in unique_slide_code_idx]
    slide_label_list = [slide_label_list[v] for v in unique_slide_code_idx]
    slide_data_code_list = [slide_data_code_list[v] for v in unique_slide_code_idx]
    print(len(slide_code_list), np.unique(slide_label_list, return_counts=True))

    # for LUAD vs LSCC | cancer type only
    def filter(arr, sel):
        return [arr[idx] for idx, v in enumerate(sel) if v == 1]

    if group_mode == "luad-lscc":
        sel = np.array(slide_label_list) > 0
        slide_code_list = filter(slide_code_list, sel)
        slide_label_list = filter(slide_label_list, sel)
        slide_label_list = [
            int(v) - 1 for v in slide_label_list
        ]  # ! hack to make binary
        slide_data_code_list = filter(slide_data_code_list, sel)
    elif group_mode == "norm-tumor":  # 0: normal - 1: tumor
        slide_label_list = np.array(slide_label_list)
        slide_label_list = np.array(slide_label_list > 0)
        slide_label_list = slide_label_list.tolist()

    slide_code_list = list(zip(slide_data_code_list, slide_code_list))
    slide_info = list(zip(slide_code_list, slide_label_list))
    # print(len(unique_slide_path), np.unique(unique_slide_label, return_counts=True))
    return np.array(slide_info)


####
def generate_split(args):

    subset_code_list = [
        "CPTAC-LUAD",
        "CPTAC-LSCC",
    ]
    data_split = joblib.load("exp_output/split/subset_info.dat")
    subject_info_list = []
    for subset_code in subset_code_list:
        subject_info_list += data_split[subset_code].tolist()
    cptac_slide_info = np.array(subject_info_list)

    subset_code_list = [
        "TCGA-LUAD-Frozen",
        "TCGA-LSCC-Frozen",
    ]
    data_split = joblib.load("exp_output/split/subset_info.dat")
    subject_info_list = []
    for subset_code in subset_code_list:
        subject_info_list += data_split[subset_code].tolist()
    tcga_f_slide_info = np.array(subject_info_list)

    subset_code_list = [
        "TCGA-LUAD",
        "TCGA-LSCC",
    ]
    data_split = joblib.load("exp_output/split/subset_info.dat")
    subject_info_list = []
    for subset_code in subset_code_list:
        subject_info_list += data_split[subset_code].tolist()
    tcga_d_slide_info = np.array(subject_info_list)

    # TODO: create merge split code
    train_set = "cptac"
    nr_splits = 5
    # cptac_split_info  = get_cv_split([v[1] > 0 for v in cptac_slide_info],
    #     nr_splits, random_state=seed, shuffle=True)
    # tcga_d_split_info = get_cv_split([v[1] > 0 for v in tcga_d_slide_info],
    #     nr_splits, random_state=seed, shuffle=True)
    # tcga_f_split_info = get_cv_split([v[1] > 0 for v in tcga_f_slide_info],
    #     nr_splits, random_state=seed, shuffle=True)

    # filtering
    cptac_slide_info = np.array([v for v in cptac_slide_info if v[1] in [1, 2]])
    tcga_d_slide_info = np.array([v for v in tcga_d_slide_info if v[1] in [1, 2]])
    tcga_f_slide_info = np.array([v for v in tcga_f_slide_info if v[1] in [1, 2]])

    cptac_split_info = get_cv_split(
        [v[1] for v in cptac_slide_info], nr_splits, random_state=seed, shuffle=True
    )
    tcga_d_split_info = get_cv_split(
        [v[1] for v in tcga_d_slide_info], nr_splits, random_state=seed, shuffle=True
    )
    tcga_f_split_info = get_cv_split(
        [v[1] for v in tcga_f_slide_info], nr_splits, random_state=seed, shuffle=True
    )

    split_info = []
    for split_idx in range(nr_splits):
        subset_dict = {}
        if train_set == "cptac":
            _, (train_idx, valid_idx) = cptac_split_info[split_idx]
            subset_dict["cptac-train"] = cptac_slide_info[train_idx].tolist()
            subset_dict["cptac-valid"] = cptac_slide_info[valid_idx].tolist()
            subset_dict["tcga-d"] = tcga_d_slide_info.tolist()
            subset_dict["tcga-f"] = tcga_f_slide_info.tolist()
        elif train_set == "tcga":
            _, (train_idx, valid_idx) = tcga_d_split_info[split_idx]
            _, (train_idy, valid_idy) = tcga_f_split_info[split_idx]
            subset_dict["tcga-train"] = (
                tcga_d_slide_info[train_idx].tolist()
                + tcga_f_slide_info[train_idy].tolist()
            )
            subset_dict["tcga-valid"] = (
                tcga_d_slide_info[valid_idx].tolist()
                + tcga_f_slide_info[valid_idy].tolist()
            )
            subset_dict["cptac"] = cptac_slide_info.tolist()
        for name, info in subset_dict.items():
            labels = [v[1] for v in info]
            print(name, np.unique(labels, return_counts=True))
        split_info.append((split_idx, subset_dict))
    if train_set == "cptac":
        joblib.dump(
            split_info, "exp_output/split/[luad-lusc]_train=cptac_test=tcga.dat"
        )
    if train_set == "tcga":
        joblib.dump(
            split_info, "exp_output/split/[luad-lusc]_train=tcga_test=cptac.dat"
        )
    return


# ! TODO: organize the naming !!!!
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--gpu", type=str, default="1")
    args = parser.parse_args()
    print(args)

    seed = 5
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    root_outdir = "../exp_output/downstream/"
    feat_dir = "../exp_output/feat/"

    generate_split(
        args,
    )
    exit()
