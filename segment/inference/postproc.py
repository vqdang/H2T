import argparse
import os
import pathlib
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology
from tqdm import tqdm

sys.path.append(os.environ["TIATOOLBOX"])

import matplotlib.pyplot as plt
from tiatoolbox.wsicore.wsireader import VirtualWSIReader

from misc.reader import get_reader
from misc.utils import imwrite, mkdir, recur_find_ext, dispatch_processing


def process_tissue(
    wsi_path,
    npy_path,
    save_path,
    threshold=0.5,
    overlaid_path=None,
    overlaid_mask=True,
    resolution={"units": "mpp", "resolution": 8.0},
):
    # for lazy overlaying
    wsi_reader = get_reader(wsi_path)
    thumb_img = wsi_reader.slide_thumbnail(**resolution)

    prediction = np.load(npy_path, mmap_mode="r")
    pred_reader = VirtualWSIReader(prediction, mode="bool")
    pred_reader.info = wsi_reader.info
    thumb_pred = pred_reader.slide_thumbnail(**resolution)[..., 1]

    thumb_pred_ = thumb_pred > threshold
    # to lessen the boundary artifact when tiling prediction
    thumb_pred_ = cv2.GaussianBlur(
        thumb_pred_.astype(np.uint8),
        (17, 17),
        sigmaX=0,
        sigmaY=0,
        borderType=cv2.BORDER_REPLICATE,
    )
    thumb_pred_ = morphology.remove_small_objects(
        thumb_pred_.astype(np.bool), min_size=32 * 32, connectivity=2
    )
    thumb_pred_ = morphology.remove_small_holes(
        thumb_pred_.astype(np.bool), area_threshold=256 * 256
    )
    thumb_pred_ = thumb_pred_.astype(np.uint8)

    imwrite(save_path, thumb_pred_.astype(np.uint8) * 255)
    if overlaid_path is not None:
        if overlaid_mask:
            sel = thumb_pred_ > 0
            thumb_pred = np.stack([thumb_pred_] * 3, axis=-1)
        else:
            sel = thumb_pred > 0.3
            cmap = plt.get_cmap("jet")
            thumb_pred = cmap(thumb_pred)[..., :3] * 255

        overlaid = thumb_img.copy()
        overlaid[sel] = 0.5 * overlaid[sel] + 0.5 * thumb_pred[sel]
        overlaid = np.clip(overlaid, 0, 255).astype(np.uint8)
        imwrite(overlaid_path, overlaid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--WSI_DIR", type=str)
    parser.add_argument("--NPY_DIR", type=str)
    parser.add_argument("--SAVE_MASK_DIR", type=str)
    parser.add_argument("--SAVE_OVERLAID_DIR", type=str, default=None)
    args = parser.parse_args()
    print(args)

    # *
    # WSI_DIR = "/mnt/storage_2/dataset/STAMPEDE/2022/Ki67/2022.07.01_ARM_A/"
    # NPY_DIR = "/mnt/storage_0/workspace/stampede/experiments/segment/new_set/Ki67/2022.07.01_ARM_A/normal-tumor/raw/"
    # SAVE_MASK_DIR = "/mnt/storage_0/workspace/stampede/experiments/segment/new_set/Ki67/2022.07.01_ARM_A/normal-tumor/processed/"
    # SAVE_OVERLAID_DIR = "/mnt/storage_0/workspace/stampede/experiments/segment/new_set/Ki67/2022.07.01_ARM_A/normal-tumor/overlaid/"
    # *
    WSI_DIR = args.WSI_DIR
    NPY_DIR = args.NPY_DIR
    SAVE_MASK_DIR = args.SAVE_MASK_DIR
    SAVE_OVERLAID_DIR = args.SAVE_OVERLAID_DIR

    mkdir(SAVE_MASK_DIR)
    mkdir(SAVE_OVERLAID_DIR)

    wsi_exts = [".svs", ".tif", ".ndpi", ".png"]
    npy_paths = recur_find_ext(f"{NPY_DIR}", [".npy"])
    wsi_paths = recur_find_ext(WSI_DIR, wsi_exts)
    assert len(npy_paths) > 0
    assert len(wsi_paths) > 0

    def get_wsi_with_name(all_paths, base_name):
        for ext in wsi_exts:
            full_name = f"{base_name}{ext}"
            for path in all_paths:
                if full_name in path:
                    return path
        assert False

    run_list = []
    for path in npy_paths:
        base_name = pathlib.Path(path).stem

        wsi_path = get_wsi_with_name(wsi_paths, base_name)
        npy_path = f"{NPY_DIR}/{base_name}.npy"

        mask_path = f"{SAVE_MASK_DIR}/{base_name}.png"
        overlaid_path = f"{SAVE_OVERLAID_DIR}/{base_name}.jpg"
        run_list.append(
            [process_tissue, wsi_path, npy_path, mask_path, 0.5, overlaid_path]
        )
    dispatch_processing(run_list, 32, show_progress=True, crash_on_exception=True)
