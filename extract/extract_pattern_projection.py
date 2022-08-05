import argparse
import os

import joblib
import numpy as np

from scipy.spatial import distance

from h2t.misc.utils import dispatch_processing, mkdir, rm_n_mkdir, load_yaml
from h2t.data.utils import retrieve_dataset_slide_info, retrieve_subset
from h2t.extract.utils import load_sample_with_info


def transform_once(root_dir, save_dir, sample_info, patterns, scaler=None):
    ds_code, wsi_code = sample_info
    patch_features, _ = load_sample_with_info(
        root_dir, sample_info, load_positions=True
    )

    # bring data into memory in case it is memmapped
    patch_features = np.array(patch_features)
    if scaler is not None:
        patch_features = scaler.transform(patch_features)
    patch_features /= np.linalg.norm(patch_features, axis=-1, keepdims=True)

    patch_distances = distance.cdist(patch_features, patterns)
    del patch_features

    save_path = f"{save_dir}/{ds_code}/{wsi_code}.dist.npy"
    np.save(save_path, patch_distances)

    patch_pattern_labels = np.argmin(patch_distances, axis=-1)
    save_path = f"{save_dir}/{ds_code}/{wsi_code}.label.npy"
    np.save(save_path, patch_pattern_labels)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--CLUSTER_CODE", type=str)
    parser.add_argument("--SOURCE_DATASET", type=str)
    parser.add_argument("--TARGET_DATASET", type=str)
    parser.add_argument(
        "--FEATURE_CODE", type=str, default="[SWAV]-[mpp=0.50]-[512-256]"
    )
    args = parser.parse_args()
    print(args)

    WORKSPACE_DIR = "exp_output/storage/a100/"

    CLUSTER_CODE = args.CLUSTER_CODE
    FEATURE_CODE = args.FEATURE_CODE
    TARGET_DATASET = args.TARGET_DATASET
    SOURCE_DATASET = args.SOURCE_DATASET

    # * debug
    CLUSTER_CODE = "sample"
    FEATURE_CODE = args.FEATURE_CODE
    TARGET_DATASET = "tcga/lung/ffpe/lscc"
    SOURCE_DATASET = "tcga-lung-luad-lusc"
    # *

    # * ---
    PWD = "/mnt/storage_0/workspace/h2t/h2t/"
    FEATURE_ROOT_DIR = f"{PWD}/experiments/local/features/{args.FEATURE_CODE}/"
    CLUSTER_DIR = (
        # f"{PWD}/experiments/local/"
        f"{PWD}/experiments/debug/cluster/"
        f"{CLUSTER_CODE}/{SOURCE_DATASET}/{FEATURE_CODE}/"
    )
    # * ---
    SAVE_DIR = f"{CLUSTER_DIR}/transformed/"
    rm_n_mkdir(SAVE_DIR)

    # * ---

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
    CLINICAL_ROOT_DIR = f"{PWD}/data/clinical/"
    dataset_sample_info = retrieve_dataset_slide_info(
        CLINICAL_ROOT_DIR, FEATURE_ROOT_DIR, dataset_identifiers
    )
    sample_info_list = dataset_sample_info[TARGET_DATASET]
    sample_info_list = [v[0] for v in sample_info_list]

    # premade all directories to prevent possible collisions
    ds_codes, _ = list(zip(*sample_info_list))
    ds_codes = np.unique(ds_codes)
    for ds_code in ds_codes:
        mkdir(f"{SAVE_DIR}/{ds_code}")

    # * ---

    model_config = load_yaml(f"{CLUSTER_DIR}/config.yaml")
    model = joblib.load(f"{CLUSTER_DIR}/model.dat")
    patterns = model.prototypical_patterns()

    scaler = None
    if os.path.exists(f"{CLUSTER_DIR}/scaler.dat"):
        scaler = joblib.load(f"{CLUSTER_DIR}/scaler.dat")
    assert scaler is None

    run_list = [
        [transform_once, FEATURE_ROOT_DIR, SAVE_DIR, sample_info, patterns, scaler]
        for sample_info in sample_info_list
    ]
    dispatch_processing(run_list, 0, crash_on_exception=True)
