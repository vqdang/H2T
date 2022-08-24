import argparse
import os
import pathlib
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology
from skimage.filters import threshold_otsu
from tqdm import tqdm

sys.path.append(os.environ["TIATOOLBOX"])

import matplotlib.pyplot as plt
from tiatoolbox.wsicore.wsireader import VirtualWSIReader

from h2t.misc.reader import get_reader
from h2t.misc.utils import imwrite, mkdir, recur_find_ext, dispatch_processing, rm_n_mkdir


def process_tissue(
    wsi_path,
    npy_path,
    save_path,
    threshold=0.5,
    overlaid_path=None,
    overlaid_mask=True,
    resolution={"units": "mpp", "resolution": 4.0},
):
    # for lazy overlaying
    wsi_reader = get_reader(wsi_path)
    thumb_img = wsi_reader.slide_thumbnail(**resolution)

    prediction = np.load(npy_path, mmap_mode="r")
    pred_reader = VirtualWSIReader(prediction, mode="bool")
    pred_reader.info = wsi_reader.info
    thumb_pred = pred_reader.slide_thumbnail(**resolution)[..., 1]

    thumb_img_gray = cv2.cvtColor(thumb_img, cv2.COLOR_RGB2GRAY)
    thumb_otsu = thumb_img_gray > threshold_otsu(thumb_img_gray)
    thumb_otsu = morphology.binary_erosion(thumb_otsu)
    thumb_otsu = morphology.binary_erosion(thumb_otsu)
    thumb_otsu = morphology.remove_small_objects(
        thumb_otsu, min_size=32 * 32, connectivity=2
    )

    thumb_pred_ = thumb_pred > threshold
    thumb_pred_ = thumb_pred_ & (~thumb_otsu)

    thumb_pred_ = thumb_pred_.astype(np.uint8)
    thumb_pred_ = morphology.binary_erosion(thumb_pred_)
    thumb_pred_ = morphology.remove_small_objects(
        thumb_pred_, min_size=512 * 512, connectivity=2
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
    # WSI_DIR='/root/dgx_workspace/h2t/dataset/tcga//breast/ffpe-he/'
    # NPY_DIR='/root/dgx_workspace/h2t/segment//fcn-convnext/tcga/breast/ffpe/raw/'
    # SAVE_MASK_DIR='/root/local_storage/storage_0/workspace/h2t/experiments/local/segment//fcn-convnext/tcga/breast/ffpe/masks/'
    # SAVE_OVERLAID_DIR='/root/local_storage/storage_0/workspace/h2t/experiments/local/segment//fcn-convnext/tcga/breast/ffpe/overlaid/'
    # *
    WSI_DIR = args.WSI_DIR
    NPY_DIR = args.NPY_DIR
    SAVE_MASK_DIR = args.SAVE_MASK_DIR
    SAVE_OVERLAID_DIR = args.SAVE_OVERLAID_DIR

    rm_n_mkdir(SAVE_MASK_DIR)
    rm_n_mkdir(SAVE_OVERLAID_DIR)

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
