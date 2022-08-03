import argparse
import importlib
import logging
import os

import joblib
import numpy as np
import yaml

from loader import SequenceDataset
from misc.utils import (
    flatten_list,
    load_yaml,
    mkdir,
    rm_n_mkdir,
    rmdir,
    setup_logger,
    update_nested_dict,
)
from recipes.opt import ABCConfig


class DatasetConstructor:
    def __init__(self, root_dir, split_info):
        """
        Attributes:
            root_dir (str): Path to root directory that contains the
                sample to be read.
            split_info (dict): A dictionary of the form

                ```
                sample: [sample_identifer: List[str], label: str]
                dataset_sample_info: {dataset_identifier: str, List[sample]}
                ```

                Here, `sample_identifier` contains the slide name and associated
                directory structures. Combining `sample_identifier` and `root_dir`
                provides a valid path to sample to be read.

        """

        self.root_dir = root_dir
        self.split_info = split_info

    def __call__(self, run_mode=None, subset_name=None, setup_augmentor=None):
        selection_dir = None
        ds = SequenceDataset(
            root_dir=self.root_dir,
            sample_info_list=self.split_info[subset_name],
            run_mode=run_mode,
            selection_dir=selection_dir,
        )
        return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--ARCH_CODE", type=str, default=None)
    parser.add_argument("--SPLIT_IDX", type=int, default=0)
    parser.add_argument("--CLUSTER_CODE", type=str, default="[method=None]")
    parser.add_argument("--ROOT_DATA", type=str, default="[rootData=tcga]")
    parser.add_argument("--TASK_CODE", type=str, default="idc-lob")
    parser.add_argument("--SOURCE_TISSUE", type=str, default="[sourceTissue=None]")
    parser.add_argument(
        "--FEATURE_CODE", type=str, default="[SWAV]-[mpp=0.50]-[512-256]"
    )
    parser.add_argument("--WSI_FEATURE_CODE", type=str, default="dH-w")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(args)

    # * ------
    seed = 5
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    PWD = os.environ["PROJECT_WORKSPACE"]
    FEATURE_CODE = args.FEATURE_CODE
    CLUSTER_CODE = args.CLUSTER_CODE
    ROOT_DATA = args.ROOT_DATA
    SOURCE_TISSUE = args.SOURCE_TISSUE
    TASK_CODE = args.TASK_CODE
    ARCH_CODE = args.ARCH_CODE

    # * debug
    # TRAINING_CONFIG_PATH = f"{PWD}/downstream/params/clam.yml"
    # FEATURE_ROOT_DIR = "/mnt/storage_0/workspace/h2t/experiments/local/features/"
    # SAVE_PATH = "experiments/debug/"
    # SPLIT_INFO_PATH = "/mnt/storage_0/workspace/h2t/[normal-luad-lusc]_train=tcga_test=cptac.dat"

    # * LSF
    TRAINING_CONFIG_PATH = f"{PWD}/downstream/params/clam.yml"
    FEATURE_ROOT_DIR = "/root/dgx_workspace/h2t/features/"
    SAVE_PATH = (
        # f"{PWD}/experiments/downstream/"
        "/root/lsf_workspace/projects/atlas/media-v1/downstream/"
        f"{SOURCE_TISSUE}/{FEATURE_CODE}/{ROOT_DATA}/"
        f"{CLUSTER_CODE}/{TASK_CODE}/{ARCH_CODE}/"
    )
    TRAINING_CONFIG_PATH = f"{PWD}/downstream/params/{ARCH_CODE}.yml"
    if TASK_CODE == "idc-lob":
        SPLIT_INFO_PATH = f"{PWD}/data/splits/[idc-lob]_train=tcga.dat"
    elif TASK_CODE == "idc-lob-ffpe":
        SPLIT_INFO_PATH = f"{PWD}/data/splits/[idc-lob]_train=tcga_ffpe.dat"
    elif TASK_CODE == "ccrcc-prcc-chrcc":
        SPLIT_INFO_PATH = f"{PWD}/data/splits/[ccrcc-prcc-chrcc]_train=tcga.dat"

    # *

    # rmdir(save_path)

    # * ------

    paramset = load_yaml(TRAINING_CONFIG_PATH)

    def run_one_split_with_param_set(
        save_path: str, data_split_info: dict, param_kwargs: dict
    ):
        run_paramset = paramset.copy()
        update_nested_dict(run_paramset, param_kwargs)
        update_nested_dict(run_paramset, {"seed": seed})

        rm_n_mkdir(save_path)
        setup_logger(f"{save_path}/debug.log")

        with open(f"{save_path}/settings.yml", "w") as fptr:
            yaml.dump(run_paramset, fptr, default_flow_style=False)

        all_labels = flatten_list(list(data_split_info.values()))
        all_labels = np.unique([v[1] for v in all_labels])
        model_config = ABCConfig.config(
            run_paramset, list(data_split_info.keys()), num_types=len(all_labels)
        )

        run_kwargs = {
            "seed": seed,
            "debug": False,
            "logging": True,
            "log_dir": f"{save_path}/model/",
            "create_dataset": DatasetConstructor(
                f"{FEATURE_ROOT_DIR}/{FEATURE_CODE}/", data_split_info
            ),
            "model_config": model_config,
        }

        from engine.manager import RunManager

        trainer = RunManager(**run_kwargs)
        trainer.run()

    data_splits = joblib.load(SPLIT_INFO_PATH)
    if args.SPLIT_IDX is not None:
        save_path_ = f"{SAVE_PATH}/{args.SPLIT_IDX:02d}/"
        rm_n_mkdir(save_path_)
        run_one_split_with_param_set(save_path_, data_splits[args.SPLIT_IDX], {})
    else:
        for split_idx, split_info in enumerate(data_splits):
            save_path_ = f"{SAVE_PATH}/{split_idx:02d}/"
            run_one_split_with_param_set(save_path_, split_info, {})
