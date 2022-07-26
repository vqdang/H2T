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
import torch.nn as nn

sys.path.append(os.environ["TIATOOLBOX"])
from tiatoolbox.models import IOSegmentorConfig, SemanticSegmentor, WSIStreamDataset
from tiatoolbox.models.abc import ModelABC
from tiatoolbox.models.engine.semantic_segmentor import (
    DeepFeatureExtractor,
    IOSegmentorConfig,
)
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, WSIMeta, WSIReader

from misc.reader import get_reader
from misc.utils import (
    convert_pytorch_checkpoint,
    difference_filename,
    dispatch_processing,
    imread,
    imwrite,
    intersection_filename,
    log_info,
    mkdir,
    recur_find_ext,
    rm_n_mkdir,
    rmdir,
)


class XReader(WSIStreamDataset):
    def _get_reader(self, img_path):
        """Get approriate reader for input path."""
        # self.preproc = XReader.preproc_func
        return get_reader(img_path)


class XPredictor(SemanticSegmentor):
    @staticmethod
    def get_reader(img_path: str, mask_path: str, mode: str, auto_get_mask: bool):
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

    if arch_name == "resnet50":
        from models.backbone import ResNetExt

        BackboneModel = ResNetExt.resnet50
    else:
        assert False, f"Unknown class architecture with alias `{arch_name}`."

    class Extractor(nn.Module):
        def __init__(
            self,
            num_input_channels=3,
        ):
            super(Extractor, self).__init__()
            self.backbone = BackboneModel(num_input_channels)
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            return

        def forward(self, img_list):
            img_list = img_list / 255.0  # scale to 0-1

            is_freeze = not self.training or self.freeze_encoder
            with torch.set_grad_enabled(not is_freeze):
                # assume output is after each down-sample resolution
                en_list = self.backbone(img_list)
            last_feat = self.gap(en_list[-1])
            return last_feat

    class WrapperModel(Extractor, ModelABC):
        def __init__(self, num_input_channels=3):
            super().__init__()
            Extractor.__init__(self, num_input_channels=num_input_channels)

        @staticmethod
        def infer_batch(model, img_list, on_gpu):
            import torch.nn.functional as F

            img_list = img_list.to("cuda").type(torch.float32)
            img_list = img_list.permute(0, 3, 1, 2).contiguous()

            model = model.eval()
            with torch.inference_mode():
                output = model(img_list)
            return [output.cpu().numpy()]

    return WrapperModel


def tissue_area(path):
    image = imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return np.mean(image > 0)


# skip:
# - already done
# - those that have no tissue masks
# - those that have little tissue area
def retrieve_input_files(
    wsi_dir,
    msk_dir,
    save_dir,
    index_range,
):
    def find_path_with_name(name, paths):
        names = [v for v in paths if str(name) in v]
        assert len(names) == 1
        return names[0]

    start_idx, end_idx = index_range
    wsi_paths = recur_find_ext(wsi_dir, [".svs", ".tif", ".ndpi", ".png"])
    msk_paths = recur_find_ext(msk_dir, [".png"])

    shared_names = intersection_filename(wsi_paths, msk_paths, return_names=True)

    end_idx = args.END_IDX if args.END_IDX <= len(shared_names) else len(shared_names)

    shared_names = shared_names[start_idx:end_idx]

    existing_output_paths = recur_find_ext(save_dir, [".features.npy"])
    # need to fake name into paths
    remanining_names = [pathlib.Path(f"temp/{v}.features.npy") for v in shared_names]
    remanining_names = difference_filename(remanining_names, existing_output_paths)
    remanining_names = [str(pathlib.Path(v).stem) for v in remanining_names]
    remanining_names = [v.replace(".features", "") for v in remanining_names]

    wsi_paths = np.array([find_path_with_name(v, wsi_paths) for v in remanining_names])
    msk_paths = np.array([find_path_with_name(v, msk_paths) for v in remanining_names])

    tissue_area_list = [[tissue_area, v] for v in msk_paths]
    tissue_area_list = dispatch_processing(
        tissue_area_list, num_workers=32, show_progress=False, crash_on_exception=True
    )
    tissue_area_list = np.array(tissue_area_list)
    sel = tissue_area_list > 0.05
    wsi_paths = list(wsi_paths[sel])
    msk_paths = list(msk_paths[sel])
    return wsi_paths, msk_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--ARCH", type=str, default="resnet50")
    parser.add_argument("--PRETRAINED", type=str)
    parser.add_argument("--WSI_DIR", type=str, default="breast")
    parser.add_argument("--MSK_DIR", type=str, default="fcn-resnet")
    parser.add_argument("--SAVE_DIR", type=str)
    parser.add_argument("--JOB_ID", type=str, default=0)
    parser.add_argument("--START_IDX", type=int, default=0)
    parser.add_argument("--END_IDX", type=int, default=2)
    args = parser.parse_args()
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    num_gpu = len(args.gpu.split(","))
    PWD = os.environ["PWD"]

    # * LSF debug
    # args.PRETRAINED = "/root/local_storage/storage_0/workspace/h2t/experiments/local/pretrained/resnet50-swav.tar"
    # args.WSI_DIR = "/root/dgx_workspace/h2t/dataset/tcga//breast/ffpe/"
    # args.MSK_DIR = (
    #     "/root/dgx_workspace/h2t/segment/fcn-convnext/tcga//breast/ffpe/masks/"
    # )
    # args.SAVE_DIR = "/root/dgx_workspace/h2t/features/swav/tcga//breast/ffpe/"
    # args.JOB_ID = 0
    # args.START_IDX = 0
    # args.END_IDX = 100
    # rm_n_mkdir(args.SAVE_DIR)
    # *

    # *
    NETWORK_DATA = False
    ARCH = args.ARCH
    PRETRAINED = args.PRETRAINED
    WSI_DIR = args.WSI_DIR
    MSK_DIR = args.MSK_DIR
    CACHE_DIR = f"/root/dgx_workspace/h2t/cache/{args.JOB_ID}/"
    SAVE_DIR = args.SAVE_DIR
    # *

    # *
    # NETWORK_DATA = False
    # ARCH = 'resnet50'
    # PRETRAINED = '/mnt/storage_0/workspace/h2t/experiments/local/pretrained/resnet50-swav.tar'
    # WSI_DIR = '/mnt/storage_2/dataset/STAMPEDE/2022/Ki67/2022.07.01_ARM_A/'
    # MSK_DIR = "/mnt/storage_0/workspace/stampede/experiments/segment/new_set/Ki67/2022.07.01_ARM_A/tissue/processed/"
    # CACHE_DIR = "experiments/cache/dump/"
    # SAVE_DIR = "experiments/cache/save/"
    # *

    # *--------

    input_files = retrieve_input_files(
        WSI_DIR, MSK_DIR, SAVE_DIR, [args.START_IDX, args.END_IDX]
    )
    input_files = list(zip(*input_files))
    print(f"To be processed: {len(input_files)}")

    # *--------

    PRETRAINED = torch.load(PRETRAINED, map_location="cpu")
    PRETRAINED = convert_pytorch_checkpoint(PRETRAINED)
    model = get_model_class(ARCH)()
    missing_weights, unexpected_weights = model.backbone.load_state_dict(
        PRETRAINED, strict=False
    )
    log_info(f"Missing Keys: {missing_weights}")
    log_info(f"Unexpected Keys: {unexpected_weights}")

    segmentor = DeepFeatureExtractor(
        model=model, num_loader_workers=16, batch_size=256 * num_gpu
    )
    ioconfig = IOSegmentorConfig(
        input_resolutions=[
            {"units": "mpp", "resolution": 0.50},
        ],
        output_resolutions=[
            {"units": "mpp", "resolution": 0.50},
        ],
        patch_input_shape=[512, 512],
        patch_output_shape=[512, 512],
        stride_shape=[256, 256],
        # un-used config parameters, do not pay attention
        save_resolution={"units": "mpp", "resolution": 8.0},
    )

    # *--------

    # because the WSIs can be on network storage, to maximize
    # read speed, copying to local
    for wsi_path, msk_path in input_files:
        wsi_ext = wsi_path.split(".")[-1]
        wsi_name = pathlib.Path(wsi_path).stem

        cache_dir = f"{CACHE_DIR}/{wsi_name}/"
        cache_wsi_path = wsi_path
        mkdir(cache_dir)

        if NETWORK_DATA:
            stime = time.perf_counter()
            cache_wsi_path = f"{CACHE_DIR}/{wsi_name}.{wsi_ext}"
            shutil.copyfile(wsi_path, cache_wsi_path)
            etime = time.perf_counter()
            print(f"Copying to local storage: {etime - stime}")

        rmdir(f"{cache_dir}/")
        output_list = segmentor.predict(
            [cache_wsi_path],
            [msk_path],
            mode="wsi",
            on_gpu=True,
            ioconfig=ioconfig,
            crash_on_exception=False,
            save_dir=f"{cache_dir}/",
        )

        output_file = f"{cache_dir}/file_map.dat"
        if not os.path.exists(output_file):
            continue
        output_info = joblib.load(output_file)

        mkdir(f"{SAVE_DIR}/")
        for input_file, output_root in output_info:
            file_name = pathlib.Path(input_file).stem
            src_path = f"{output_root}.position.npy"
            dst_path = f"{SAVE_DIR}/{file_name}.position.npy"
            shutil.copyfile(src_path, dst_path)
            src_path = f"{output_root}.features.0.npy"
            dst_path = f"{SAVE_DIR}/{file_name}.features.npy"
            shutil.copyfile(src_path, dst_path)
        rmdir(f"{cache_dir}/")
