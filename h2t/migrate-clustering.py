import pathlib
import itertools
import shutil
import os
import numpy as np
from h2t.misc.utils import dispatch_processing, mkdir, recur_find_ext, rm_n_mkdir


if __name__ == "__main__":
    NUM_WORKERS = 0
    ROOT_DIR = "/root/lsf_workspace/projects/atlas/"
    SRC_DIR = "/root/lsf_workspace/projects/atlas/media-v0/cluster_filtered-0.50/"
    DST_DIR = "/root/lsf_workspace/projects/atlas/final/cluster/"

    FEATURE_CODES = [
        "[SUPERVISE]-[mpp=0.25]-[512-256]",
        "[SUPERVISE]-[mpp=0.50]-[512-256]"
        "[SWAV]-[mpp=0.25]-[512-256]",
        "[SWAV]-[mpp=0.50]-[512-256]",
    ]
    SOURCE_TISSUE_MAP = {
        "[sourceTissue=LUAD]/[rootData=cptac]": "cptac-lung-luad",
        "[sourceTissue=LUSC]/[rootData=cptac]": "cptac-lung-lusc",
        "[sourceTissue=Normal]/[rootData=cptac]": "cptac-lung-normal",
        "[sourceTissue=Normal+LUAD+LUSC]/[rootData=cptac]": "cptac-lung-normal-luad-lusc",

        "[sourceTissue=LUAD]/[rootData=tcga]": "tcga-lung-luad",
        "[sourceTissue=LUSC]/[rootData=tcga]": "tcga-lung-lusc",
        "[sourceTissue=Normal]/[rootData=tcga]": "tcga-lung-normal",
        "[sourceTissue=Normal+LUAD+LUSC]/[rootData=tcga]": "tcga-lung-normal-luad-lusc",
    }
    METHOD_MAP = {
        "[method=0]": "spherical-kmean-8",
        "[method=1]": "spherical-kmean-16",
        "[method=2]": "spherical-kmean-32",
    }
    metadata = itertools.product(
        SOURCE_TISSUE_MAP.keys(),
        METHOD_MAP.keys(),
        FEATURE_CODES
    )
    FLATTEN_DIR = "/root/lsf_workspace/projects/atlas/final/cluster/flatten/"

    rm_n_mkdir(DST_DIR)
    mkdir(FLATTEN_DIR)
    for src_tissue, src_method, feature_code in metadata:
        dst_tissue = SOURCE_TISSUE_MAP[src_tissue]
        dst_method = METHOD_MAP[src_method]

        src_tissue = src_tissue.split("/")
        src_path = (
            f"{SRC_DIR}/{src_tissue[0]}/{feature_code}/"
            f"{src_tissue[1]}/{src_method}/model.dat"
        )
        dst_path = (
            f"{DST_DIR}/{dst_method}/{dst_tissue}/{feature_code}/models/model-049.dat"
        )
        if not os.path.exists(src_path):
            continue
        print(src_path)
        print(dst_path)
        print("----")
        mkdir(f"{DST_DIR}/{dst_method}/{dst_tissue}/{feature_code}/models/")
        shutil.copyfile(src_path, dst_path)
        # copy to flattened hierarchy
        flatten_dst_path = (
            f"{FLATTEN_DIR}/{dst_method}_{dst_tissue}_{feature_code}_model-049.dat"
        )
        shutil.copyfile(src_path, flatten_dst_path)
    # ----

    SRC_DIR = "/root/lsf_workspace/projects/atlas/media-v1/clustering/"
    DST_DIR = "/root/lsf_workspace/projects/atlas/final/cluster/"

    FEATURE_CODES = [
        # "[SUPERVISE]-[mpp=0.25]-[512-256]",
        # "[SUPERVISE]-[mpp=0.50]-[512-256]"
        # "[SWAV]-[mpp=0.25]-[512-256]",
        "[SWAV]-[mpp=0.50]-[512-256]",
    ]
    SOURCE_TISSUES = [
        "tcga-breast-idc-lob",
        "tcga-kidney-ccrcc-prcc-chrcc",
    ]
    METHODS = [
        "spherical-kmean-8",
        "spherical-kmean-16",
        "spherical-kmean-32",
    ]
    metadata = itertools.product(
        SOURCE_TISSUES,
        METHODS,
        FEATURE_CODES
    )
    for src_tissue, src_method, feature_code in metadata:
        src_path = (
            f"{SRC_DIR}/{src_method}/{src_tissue}/{feature_code}/model.dat"
        )        
        dst_path = (
            f"{DST_DIR}/{src_method}/{src_tissue}/{feature_code}/models/model-049.dat"
        )
        if not os.path.exists(src_path):
            src_path = (
                f"{SRC_DIR}/{src_method}/{src_tissue}/{feature_code}/models/model-049.dat"
            )        
            dst_path = (
                f"{DST_DIR}/{src_method}/{src_tissue}/{feature_code}/models/model-049.dat"
            )

        print(src_path)
        print(dst_path)
        print("----")
        mkdir(f"{DST_DIR}/{src_method}/{src_tissue}/{feature_code}/models/")
        shutil.copyfile(src_path, dst_path)
        # copy to flattened hierarchy
        flatten_dst_path = (
            f"{FLATTEN_DIR}/{src_method}_{src_tissue}_{feature_code}_model-049.dat"
        )
        shutil.copyfile(src_path, flatten_dst_path)
