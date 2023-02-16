import pathlib
import itertools
import shutil
import os
import numpy as np
from h2t.misc.utils import dispatch_processing, mkdir, recur_find_ext


def migrate_once(src_path, dst_path):
    shutil.copyfile(src_path, dst_path)


if __name__ == "__main__":
    NUM_WORKERS = 0
    ROOT_DIR = "/root/lsf_workspace/projects/atlas/"
    SRC_DIR = "/root/lsf_workspace/projects/atlas/media-v0/tissue_segment/FCN-Generic-v0.1/predictions/"
    DST_DIR = "/root/lsf_workspace/projects/atlas/final/segment/fcn-resnet/"

    FEATURE_ROOT = "/root/lsf_workspace/projects/atlas/final/features/[SWAV]-[mpp=0.50]-[512-256]/"
    FOLDER_MAP = {
        "CPTAC-LSCC": "cptac/lung/",
        "CPTAC-LUAD": "cptac/lung/",
        "TCGA-LSCC": "tcga/lung/ffpe/",
        "TCGA-LSCC-Frozen": "tcga/lung/frozen/",
        "TCGA-LUAD": "tcga/lung/ffpe/",
        "TCGA-LUAD-Frozen": "tcga/lung/frozen/",
    }
    for src_dir, dst_dir in FOLDER_MAP.items():
        src_dir_ = f"{SRC_DIR}/{src_dir}/masks/"
        dst_ovl_dir_ = f"{DST_DIR}/{dst_dir}/overlaid/"
        dst_msk_dir_ = f"{DST_DIR}/{dst_dir}/masks/"
        mkdir(dst_ovl_dir_)
        mkdir(dst_msk_dir_)

        src_file_names = recur_find_ext(src_dir_, [".png"])
        src_file_names = [v.split("/")[-1] for v in src_file_names]

        src_ovl_file_names = [v for v in src_file_names if "overlaid" in v]
        src_ovl_path = [f"{src_dir_}/{v}" for v in src_ovl_file_names]

        src_ovl_file_names = [v.replace(".overlaid", "") for v in src_ovl_file_names]
        src_msk_file_names = [v for v in src_file_names if "overlaid" not in v and "thumb" not in v]

        base_names = [v.replace(".png", "") for v in src_msk_file_names]
        flags = [f"{FEATURE_ROOT}/{dst_dir}/{v}.features.npy" for v in base_names]
        flags = [os.path.exists(v) for v in flags]
        assert all([os.path.exists(v) for v in flags])

        dst_ovl_path = [f"{dst_ovl_dir_}/{v}" for v in src_ovl_file_names]
        run_list = [
            [migrate_once, v[0], v[1]] for v in list(zip(src_ovl_path, dst_ovl_path))
        ]
        dispatch_processing(run_list, num_workers=NUM_WORKERS, crash_on_exception=True)

        src_msk_path = [f"{src_dir_}/{v}" for v in src_msk_file_names]
        dst_msk_path = [f"{dst_msk_dir_}/{v}" for v in src_msk_file_names]
        run_list = [
            [migrate_once, v[0], v[1]] for v in list(zip(src_msk_path, dst_msk_path))
        ]
        dispatch_processing(run_list, num_workers=NUM_WORKERS, crash_on_exception=True)
