# %%

import joblib
import numpy as np
from h2t.data.utils import (per_dataset_stratified_split,
                            retrieve_dataset_slide_info)
from h2t.misc.utils import load_yaml

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

# PWD = "/mnt/storage_0/workspace/h2t/"
# feature_root_dir = "/mnt/storage_0/workspace/h2t/experiments/local/features/[SWAV]-[mpp=0.50]-[512-256]/"

PWD = "/root/local_storage/storage_0/workspace/h2t/h2t/"

FEATURE_ROOT_DIR = "/root/dgx_workspace/h2t/features/[SWAV]-[mpp=0.50]-[512-256]/"
CLINICAL_ROOT_DIR = f"{PWD}/data/clinical/"

dataset_sample_info = retrieve_dataset_slide_info(
    CLINICAL_ROOT_DIR, FEATURE_ROOT_DIR, dataset_identifiers
)

# SPLIT_CODE = "[normal-luad-lusc]_train=tcga_test=cptac"
# SPLIT_CODE = "[idc-lob]_train=tcga_ffpe"
# SPLIT_CODE = "[ccrcc-prcc-chrcc]_train=tcga_ffpe"
SPLIT_CODE = "[ccrcc-prcc-chrcc]_train=tcga"

config = load_yaml(f"{PWD}/data/config.yaml")
splits = per_dataset_stratified_split(config[SPLIT_CODE], dataset_sample_info)
print("train", np.unique([v[1] for v in splits[0]["train"]], return_counts=True))
print("valid", np.unique([v[1] for v in splits[0]["valid"]], return_counts=True))
joblib.dump(splits, f"{SPLIT_CODE}.dat")

# %%
