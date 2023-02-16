import pathlib
import itertools
import numpy as np
from h2t.misc.utils import dispatch_processing, mkdir, recur_find_ext


def migrate_once(src_info, dst_info, sel_path):
    sel = np.load(sel_path) > 0
    src_features = np.load(src_info[0])
    src_positions = np.load(src_info[1])
    assert src_features.shape[0] == src_positions.shape[0]
    assert sel.shape[0] == src_features.shape[0]

    num_sel = np.sum(sel)
    np.save(dst_info[0], np.array(src_features[sel]))
    np.save(dst_info[1], np.array(src_positions[sel]))
    dst_features = np.load(dst_info[0])
    dst_positions = np.load(dst_info[1])
    assert dst_features.shape[0] == dst_positions.shape[0]
    assert num_sel == dst_positions.shape[0], f"{num_sel} vs {dst_positions.shape}"


if __name__ == "__main__":
    MPP_CODES = ["mpp=0.25", "mpp=0.50"]
    BACKBONE_CODES = ["SWAV", "SUPERVISE"]
    ROOT_DIR = "/root/lsf_workspace/projects/atlas/"
    SRC_DIR = "/root/lsf_workspace/projects/atlas/media-v0/features/"
    DST_DIR = "/root/lsf_workspace/projects/atlas/final/features/"

    FOLDER_MAP = {
        "CPTAC-LSCC": "cptac/lung/",
        "CPTAC-LUAD": "cptac/lung/",
        "TCGA-LSCC": "tcga/lung/ffpe/",
        "TCGA-LSCC-Frozen": "tcga/lung/frozen/",
        "TCGA-LUAD": "tcga/lung/ffpe/",
        "TCGA-LUAD-Frozen": "tcga/lung/frozen/",
    }
    metadata = list(itertools.product(BACKBONE_CODES, MPP_CODES))
    for backbone, mpp in metadata:
        for src_dir, dst_dir in FOLDER_MAP.items():
            sel_dir = f"{SRC_DIR}/{mpp}/selections-0.50/{src_dir}/"
            src_dir_ = f"{SRC_DIR}/[{backbone}]-[{mpp}]-[512-256]/{src_dir}/"
            dst_dir_ = f"{DST_DIR}/[{backbone}]-[{mpp}]-[512-256]/{dst_dir}/"
            mkdir(dst_dir_)

            sel_wsi_codes = recur_find_ext(sel_dir, [".npy"])
            src_wsi_codes = recur_find_ext(src_dir_, [".npy"])
            assert len(sel_wsi_codes) == (len(src_wsi_codes) / 2)

            wsi_codes = [pathlib.Path(v).stem for v in sel_wsi_codes]
            src_wsi_paths = [
                [f"{src_dir_}/{v}.features.npy", f"{src_dir_}/{v}.position.npy"]
                for v in wsi_codes
            ]

            dst_wsi_paths = [
                [f"{dst_dir_}/{v}.features.npy", f"{dst_dir_}/{v}.position.npy"]
                for v in wsi_codes
            ]
            sel_wsi_paths = [f"{sel_dir}/{v}.npy" for v in wsi_codes]
            args_list = list(zip(src_wsi_paths, dst_wsi_paths, sel_wsi_paths))
            run_list = [[migrate_once, v[0], v[1], v[2]] for v in args_list]
            dispatch_processing(run_list, num_workers=16, crash_on_exception=True)
