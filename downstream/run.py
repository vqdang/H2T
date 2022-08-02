import argparse
import importlib
import logging
import os

import joblib
import yaml

from loader import SequenceDataset
from misc.utils import mkdir, rm_n_mkdir, rmdir, load_yaml, update_nested_dict

from recipes.opt import ABCConfig


class DatasetConstructor():
    def __init__(self, split_info):
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

        self.root_dir = f"{FEATURE_ROOT_DIR}/{FEATURE_CODE}/"
        self.split_info = split_info

    def __call__(self,
        run_mode=None,
        subset_name=None,
        setup_augmentor=None
    ):
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
    parser.add_argument("--SPLIT_IDX", type=int, default=0)
    parser.add_argument("--run_mode", type=str, default="train")
    parser.add_argument("--TASK_CODE", type=str, default="LUAD-LUSC")
    parser.add_argument("--CLUSTER_CODE", type=str, default="[method=None]")
    parser.add_argument("--ATLAS_ROOT_DATA", type=str, default="[rootData=tcga]")
    parser.add_argument(
        "--ATLAS_SOURCE_TISSUE", type=str, default="[sourceTissue=None]"
    )
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

    PWD = os.environ["PWD"]
    TASK_CODE = args.TASK_CODE
    FEATURE_CODE = args.FEATURE_CODE
    CLUSTER_CODE = args.CLUSTER_CODE
    ATLAS_ROOT_DATA = args.ATLAS_ROOT_DATA
    ATLAS_SOURCE_TISSUE = args.ATLAS_SOURCE_TISSUE
    TRAINING_CONFIG_PATH = f"{PWD}/downstream/params/clam.yml"

    MPP = 0.50
    THRESHOLD = 0.50
    # SAVE_PATH = (
    #     f"{WORKSPACE_DIR}/downstream_filtered-{THRESHOLD:0.2f}/"
    #     f"{ATLAS_SOURCE_TISSUE}/{FEATURE_CODE}/{ATLAS_ROOT_DATA}/"
    #     f"{CLUSTER_CODE}/{TASK_CODE}/{MODEL_CODE}/"
    # )
    FEATURE_ROOT_DIR = "/mnt/storage_0/workspace/h2t/experiments/local/features/"
    SAVE_PATH = "experiments/debug/"
    # SPLIT_INFO_PATH = f"{PWD}/data/splits/[idc-lob]_train=tcga.dat"
    SPLIT_INFO_PATH = "/mnt/storage_0/workspace/h2t/[normal-luad-lusc]_train=tcga_test=cptac.dat"
    # rmdir(save_path)

    # * ------

    # -------------------------------------------------------------

    # ! transformer
    # paramset = load_yaml(f'param/hopfield.yml')
    # ! clam
    paramset = load_yaml(TRAINING_CONFIG_PATH)

    def setup_logger(path: str):
        """Will reset logger handler every single call."""
        logging.basicConfig(level=logging.INFO,)
        log_formatter = logging.Formatter(
            "|%(asctime)s.%(msecs)03d| [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d|%H:%M:%S",
        )
        log = logging.getLogger()  # root logger
        for hdlr in log.handlers[:]:  # remove all old handlers
            log.removeHandler(hdlr)
        new_hdlr_list = [
            logging.FileHandler(path),
            logging.StreamHandler(),
        ]
        for hdlr in new_hdlr_list:
            hdlr.setFormatter(log_formatter)
            log.addHandler(hdlr)


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

        model_config = ABCConfig.config(run_paramset, list(data_split_info.keys()))

        run_kwargs = {
            "seed": seed,
            "debug": False,
            "logging": True,
            "log_dir": f"{save_path}/model/",
            "create_dataset": DatasetConstructor(data_split_info),
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
        for split_idx, split_info in enumerate(data_split):
            save_path_ = f"{SAVE_PATH}/{split_idx:02d}/"
            run_one_split_with_param_set(save_path_, split_info, {})
    print("here")
