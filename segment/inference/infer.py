
import argparse
import os
import pathlib
import shutil
import sys
import time

import cv2
import joblib
import numpy as np
import torch

sys.path.append(os.environ["TIATOOLBOX"])
from tiatoolbox.models import (IOSegmentorConfig, SemanticSegmentor,
                               WSIStreamDataset)
from tiatoolbox.models.abc import ModelABC
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, WSIMeta, WSIReader

from misc.reader import get_reader
from misc.utils import (convert_pytorch_checkpoint, difference_filename,
                        imread, imwrite, mkdir, recur_find_ext, rm_n_mkdir,
                        rmdir)
from models.utils import crop_op


class XReader(WSIStreamDataset):

    def _get_reader(self, img_path):
        """Get approriate reader for input path."""
        # self.preproc = XReader.preproc_func
        return get_reader(img_path)


class XPredictor(SemanticSegmentor):
    @staticmethod
    def get_reader(
                img_path: str,
                mask_path: str,
                mode: str,
                auto_get_mask: bool
            ):
        """Get reader for mask and source image."""
        img_path = pathlib.Path(img_path)

        reader = get_reader(img_path)
        mask_reader = None
        if mask_path is not None:
            if not os.path.isfile(mask_path):
                raise ValueError("`mask_path` must be a valid file path.")
            # assume to be gray
            mask = cv2.imread(mask_path)
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            mask = np.array(mask > 0, dtype=np.uint8)

            mask_reader = VirtualWSIReader(mask)
            mask_reader.info = reader.info
        elif auto_get_mask and mode == "wsi" and mask_path is None:
            # if no mask provided and `wsi` mode, generate basic tissue
            # mask on the fly
            mask_reader = reader.tissue_mask(resolution=1.25, units="power")
            mask_reader.info = reader.info
        return reader, mask_reader


def get_model_class(arch_name):
    """Instantiate a new class definition.
    
    This will instantiate a new class definition that is a composite of
    class `ModelABC` in `tiatoolbox` and our custom class architecture
    aliased with the input `name`.

    """

    if arch_name == "fcn-convnext":
        from models.fcn import FCN_ConvNext as Arch
    elif arch_name == "fcn-resnet":
        from models.fcn import FCN_ResNet as Arch
    else:
        assert False, f"Unknown class architecture with alias `{arch_name}`."

    class WrapperModel(Arch, ModelABC):
        def __init__(self, num_input_channels=3):
            super().__init__()
            Arch.__init__(self, num_input_channels=num_input_channels)
        
        @staticmethod
        def infer_batch(model, img_list, on_gpu):
            import torch.nn.functional as F
            img_list = img_list.to('cuda').type(torch.float32)
            img_list = img_list.permute(0, 3, 1, 2).contiguous()

            with torch.inference_mode():
                output = model(img_list)
                output = F.softmax(output, 1)
                output = F.interpolate(
                    output, scale_factor=2,
                    mode="bilinear", align_corners=False
                )
                output = crop_op(output, [512, 512])
                output = output.permute(0, 2, 3, 1)

            return [output.cpu().numpy()]
    return WrapperModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--tissue", type=str, default="breast")
    parser.add_argument("--arch_name", type=str, default="fcn-resnet")
    parser.add_argument("--JOB_ID", type=int, default=0)
    parser.add_argument("--START_IDX", type=int, default=0)
    parser.add_argument("--END_IDX", type=int, default=1)
    args = parser.parse_args()
    print(args)

    args = parser.parse_args()

    TISSUE_CODE = args.tissue
    ARCH_NAME = args.arch_name
    # * LSF
    PWD = f"/root/local_storage/storage_0/workspace/h2t/"
    # WSI_DIR = f'/mnt/tia-jalapeno/sources/tcga/{TISSUE_CODE}/'
    WSI_DIR = (
        f'/root/dgx_workspace/h2t/dataset/tcga/{TISSUE_CODE}-he/'
        if "ffpe" in TISSUE_CODE else
        f'/root/dgx_workspace/h2t/dataset/tcga/{TISSUE_CODE}/'
    )
    MSK_DIR = None
    CACHE_DIR = f"/root/dgx_workspace/h2t/cache/{args.JOB_ID}-{ARCH_NAME}_/"
    # *

    # * local
    # PWD = "/mnt/storage_0/workspace/h2t"
    # WSI_DIR = '/mnt/storage_0/dataset/STAMPEDE/2022/Ki67/'
    # MSK_DIR = "/mnt/storage_0/workspace/stampede/experiments/segment/new_set/Ki67/tissue/processed/"
    # CACHE_DIR = f"/mnt/storage_3/cache/STAMPEDE/"
    # *

    SAVE_ROOT_DIR = (
        # f"{PWD}/experiments/local/segment/{ARCH_NAME}/tcga/breast/"
        f"/root/dgx_workspace/h2t/segment/{ARCH_NAME}/tcga/{TISSUE_CODE}/"
    )
    # *
    # ! need to reorganize to pipe config
    if ARCH_NAME == "fcn-convnext":
        PRETRAINED = f"{PWD}/experiments/local/pretrained/tissue-segment-fcn-convnext.tar"
        ioconfig = IOSegmentorConfig(
            input_resolutions=[
                {'units': 'mpp', 'resolution': 4.0},
            ],
            output_resolutions=[
                {'units': 'mpp', 'resolution': 4.0},
            ],
            save_resolution={'units': 'mpp', 'resolution': 8.0},
            patch_input_shape=[1024, 1024],
            patch_output_shape=[512, 512],
            stride_shape=[256, 256],
        )
    elif ARCH_NAME == "fcn-resnet":
        PRETRAINED = f"{PWD}/experiments/local/pretrained/tissue-segment-fcn-resnet.tar"
        ioconfig = IOSegmentorConfig(
            input_resolutions=[
                {'units': 'mpp', 'resolution': 8.0},
            ],
            output_resolutions=[
                {'units': 'mpp', 'resolution': 8.0},
            ],
            save_resolution={'units': 'mpp', 'resolution': 8.0},
            patch_input_shape=[1024, 1024],
            patch_output_shape=[512, 512],
            stride_shape=[256, 256],
        )
    # *

    wsi_paths = recur_find_ext(WSI_DIR, [".svs", ".tif", ".ndpi", ".png"])

    PRETRAINED = torch.load(PRETRAINED, map_location="cpu")["desc"]
    PRETRAINED = convert_pytorch_checkpoint(PRETRAINED)
    model = get_model_class(ARCH_NAME)()
    model.load_state_dict(PRETRAINED)

    segmentor = XPredictor(
        model=model,
        num_loader_workers=16,
        batch_size=32,
        dataset_class=XReader,
    )

    # skip already done
    def filter_already_done(wsi_paths):
        remaining = []
        for wsi_path in wsi_paths:
            wsi_name = pathlib.Path(wsi_path).stem
            save_path = f"{SAVE_ROOT_DIR}/raw/{wsi_name}.npy"
            if not os.path.exists(save_path):
                remaining.append(wsi_path)
        return remaining

    original_paths = wsi_paths
    end_idx = (
        args.END_IDX if args.END_IDX <= len(original_paths) else len(original_paths)
    )
    wsi_paths = wsi_paths[args.START_IDX : end_idx]
    wsi_paths = filter_already_done(wsi_paths)
    print(len(wsi_paths))

    msk_paths = [None] * len(wsi_paths)
    if MSK_DIR is not None:
        wsi_names = [pathlib.Path(v).stem for v in wsi_paths]
        msk_paths = [f"{MSK_DIR}/{v}.png" for v in wsi_names]

    # because the WSIs can be on network storage, to maximize
    # read speed, copying to local
    for wsi_path, msk_path in list(zip(wsi_paths, msk_paths)):

        wsi_ext = wsi_path.split(".")[-1]
        wsi_name = pathlib.Path(wsi_path).stem

        cache_dir = f"{CACHE_DIR}/{wsi_name}/"
        mkdir(cache_dir)

        stime = time.perf_counter()
        # cache_wsi_path = f"{CACHE_DIR}/{wsi_name}.{wsi_ext}"
        # shutil.copyfile(wsi_path, cache_wsi_path)
        cache_wsi_path = wsi_path
        etime = time.perf_counter()
        print(f"Copying to local storage: {etime - stime}")

        rmdir(f'{cache_dir}/')
        output_list = segmentor.predict(
                        [cache_wsi_path],
                        [msk_path],
                        mode='wsi',
                        on_gpu=True,
                        ioconfig=ioconfig,
                        crash_on_exception=False,
                        save_dir=f'{cache_dir}/'
                    )

        output_file = f'{cache_dir}/file_map.dat'
        if not os.path.exists(output_file):
            continue
        output_info = joblib.load(output_file)

        mkdir(f"{SAVE_ROOT_DIR}/raw/")
        for input_file, output_root in output_info:
            file_name = pathlib.Path(input_file).stem
            src_path = f"{output_root}.raw.0.npy"
            dst_path = f"{SAVE_ROOT_DIR}/raw/{file_name}.npy"
            shutil.copyfile(src_path, dst_path)
        rmdir(f'{cache_dir}/')
