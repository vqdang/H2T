import argparse
import importlib
import os

import joblib
import numpy as np
import yaml

from h2t.downstream.loader import SequenceDataset, FeatureDataset
from h2t.misc.utils import (
    flatten_list,
    load_yaml,
    mkdir,
    rm_n_mkdir,
    rmdir,
    setup_logger,
    update_nested_dict,
)
from h2t.downstream.recipes.opt import ABCConfig


class DatasetConstructor:
    def __init__(
        self,
        split_info=None,
        generate_feature_path=None,
        generate_selection_path=None,
        dataset="sequence",
        **kwargs,
    ):
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
        assert dataset in ["sequence", "single"]

        self.kwargs = kwargs
        self.dataset = dataset
        self.generate_feature_path = generate_feature_path
        self.generate_selection_path = generate_selection_path
        self.split_info = split_info

    def __call__(self, run_mode=None, subset_name=None, setup_augmentor=None):
        selection_dir = None
        DatasetClass = {"sequence": SequenceDataset, "single": FeatureDataset}[
            self.dataset
        ]
        ds = DatasetClass(
            generate_feature_path=self.generate_feature_path,
            generate_selection_path=self.generate_selection_path,
            sample_info_list=self.split_info[subset_name],
            run_mode=run_mode,
            **self.kwargs,
        )
        return ds


class CMDArgumentParser():
    def __init__(self, **kwargs):
        self.generate_selection_path = None
        for key in kwargs:
            setattr(self, key, kwargs[key])

class MILArgumentParser(CMDArgumentParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # pointing to deep features
        self.DATASET_KWARGS = {}
        self.DATASET_MODE = "sequence"

    def generate_feature_path(self, sample_info):
        subset_code, wsi_code = sample_info
        path = (
            f"/mnt/storage_0/workspace/h2t/h2t/experiments/local/features/"
            f"{self.FEATURE_CODE}/{subset_code}/{wsi_code}"
        )
        return path

    def retrieve_paramset(self):
        return load_yaml(self.TRAINING_CONFIG_PATH)

class H2TArgumentParser(CMDArgumentParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.DATASET_KWARGS = {
            "feature_codes": self.WSI_FEATURE_CODE.split("#"),
            "target_shape": [512, 512],
        }
        self.DATASET_MODE = "single"
        self.CLUSTER_DIR = (
            # f"/mnt/storage_0/workspace/h2t/h2t/experiments/remote/media-v1/clustering/"
            "/root/lsf_workspace/projects/atlas/media-v1/clustering/"
            f"{self.CLUSTER_CODE}/{self.SOURCE_DATASET}/{self.FEATURE_CODE}/"
        )

    def generate_feature_path(self, sample_info, projection_code):
        subset_code, wsi_code = sample_info
        path = (
            # f"/mnt/storage_0/workspace/h2t/h2t/experiments/remote/media-v1/clustering/"
            "/root/lsf_workspace/projects/atlas/media-v1/clustering/"
            f"{self.CLUSTER_CODE}/{self.SOURCE_DATASET}/{self.FEATURE_CODE}/features/"
            f"{projection_code}/{subset_code}/{wsi_code}"
        )
        return path

    def retrieve_paramset(self):
        paramset = load_yaml(self.TRAINING_CONFIG_PATH)
        metadata = paramset["metadata"]
        wsi_projection_codes = self.WSI_FEATURE_CODE.split("#")

        cluster_config = load_yaml(f"{self.CLUSTER_DIR}/config.yaml")
        num_patterns = cluster_config["num_patterns"]

        model_kwargs = paramset["model_kwargs"]

        if any("H" == v for v in wsi_projection_codes):
            model_kwargs["num_input_channels"] = num_patterns
        elif any("dH" in v for v in wsi_projection_codes):
            model_kwargs["num_input_channels"] = metadata["num_features"] * num_patterns
        else:
            model_kwargs["num_input_channels"] = 0

        if any("C" == v for v in wsi_projection_codes):
            model_kwargs["num_input_channels"] += (num_patterns * num_patterns)
        if any("dC-raw" == v for v in wsi_projection_codes):
            colocal_kwargs = {
                "encode": None,
                "encode_kwargs": {"max_value": num_patterns + 1},
            }
            metadata["architecture_name"] = "cnn-probe"
        elif any("dC-onehot" == v for v in wsi_projection_codes):
            colocal_kwargs = {
                "encode": "onehot",
                "encode_kwargs": {"max_value": num_patterns + 1},
            }
            metadata["architecture_name"] = "cnn-probe"
        else:
            colocal_kwargs = None
            metadata["architecture_name"] = "linear-probe"

        model_kwargs["colocal"] = colocal_kwargs
        metadata["projection_codes"] = wsi_projection_codes
        metadata["cluster_method"] = cluster_config
        return paramset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--DATA_SPLIT_CODE", type=str)
    parser.add_argument("--TRAINING_CONFIG_CODE", type=str)
    parser.add_argument("--SPLIT_IDX", type=int, default=None)
    parser.add_argument("--ARCH_CODE", type=str, default=None)
    parser.add_argument("--FEATURE_CODE", type=str)
    parser.add_argument("--CLUSTER_CODE", type=str, default="")
    parser.add_argument("--SOURCE_DATASET", type=str, default="")
    parser.add_argument("--WSI_FEATURE_CODE", type=str, default="")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(args)

    # * ------
    seed = 5
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    PWD = os.environ["PROJECT_WORKSPACE"]
    SPLIT_IDX = args.SPLIT_IDX
    ARCH_CODE = args.ARCH_CODE
    FEATURE_CODE = args.FEATURE_CODE
    CLUSTER_CODE = args.CLUSTER_CODE
    SOURCE_DATASET = args.SOURCE_DATASET
    WSI_FEATURE_CODE = args.WSI_FEATURE_CODE

    DATA_SPLIT_CODE = args.DATA_SPLIT_CODE
    TRAINING_CONFIG_CODE = args.TRAINING_CONFIG_CODE

    # * debug MIL
    # PWD = "/mnt/storage_0/workspace/h2t/h2t/"
    # SAVE_PATH = f"{PWD}/experiments/debug/downstream/"
    # TRAINING_CONFIG_PATH = f"{PWD}/downstream/params/clam.yml"
    # SPLIT_INFO_PATH = f"{PWD}/[normal-luad-lusc]_train=tcga_test=cptac.dat"
    # FEATURE_CODE = "[SWAV]-[mpp=0.50]-[512-256]"

    # * debug H2T
    # PWD = "/mnt/storage_0/workspace/h2t/h2t/"
    # SAVE_PATH = f"{PWD}/experiments/debug/downstream/"
    # TRAINING_CONFIG_PATH = f"{PWD}/downstream/params/probe-template.yml"
    # DATA_SPLIT_CODE = "[idc-lob]_train=tcga"
    # CLUSTER_CODE = "spherical-kmean-16"
    # SOURCE_DATASET = f"tcga-breast-idc-lob"
    # WSI_FEATURE_CODE = "dH-n-w#dC-onehot"
    # # WSI_FEATURE_CODE = "C"
    # WSI_FEATURE_CODE = "dH-n-w"
    # FEATURE_CODE = "[SWAV]-[mpp=0.50]-[512-256]"
    # SPLIT_INFO_PATH = f"{PWD}/data/splits/{DATA_SPLIT_CODE}.dat"

    # * LSF
    PWD = os.environ["PROJECT_WORKSPACE"]
    TRAINING_CONFIG_PATH = f"{PWD}/downstream/params/{TRAINING_CONFIG_CODE}.yml"
    SPLIT_INFO_PATH = f"{PWD}/data/splits/{DATA_SPLIT_CODE}.dat"
    SAVE_PATH = (
        # f"{PWD}/experiments/downstream/"
        "/root/lsf_workspace/projects/atlas/media-v1/downstream-x/"
        f"{DATA_SPLIT_CODE}/{FEATURE_CODE}/"
        f"{SOURCE_DATASET}/{CLUSTER_CODE}/{ARCH_CODE}/{WSI_FEATURE_CODE}/"
    )

    # *

    if WSI_FEATURE_CODE is not None:
        arg_parser = H2TArgumentParser(
            PWD=PWD,
            FEATURE_CODE=FEATURE_CODE,
            CLUSTER_CODE=CLUSTER_CODE,
            SOURCE_DATASET=SOURCE_DATASET,
            WSI_FEATURE_CODE=WSI_FEATURE_CODE,
            TRAINING_CONFIG_PATH=TRAINING_CONFIG_PATH
        )
    else:
        arg_parser = MILArgumentParser(
            PWD=PWD,
            FEATURE_CODE=FEATURE_CODE,
            TRAINING_CONFIG_PATH=TRAINING_CONFIG_PATH,
        )

    # * ------

    def run_one_split_with_param_set(
        save_path: str, data_split_info: dict, param_kwargs: dict
    ):
        run_paramset = arg_parser.retrieve_paramset()
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
                split_info=data_split_info,
                generate_feature_path=arg_parser.generate_feature_path,
                generate_selection_path=arg_parser.generate_selection_path,
                dataset=arg_parser.DATASET_MODE,
                **arg_parser.DATASET_KWARGS,
            ),
            "model_config": model_config,
        }

        from h2t.engine.manager import RunManager

        trainer = RunManager(**run_kwargs)
        trainer.run()

    data_splits = joblib.load(SPLIT_INFO_PATH)
    if SPLIT_IDX is not None:
        save_path_ = f"{SAVE_PATH}/{SPLIT_IDX:02d}/"
        rm_n_mkdir(save_path_)
        run_one_split_with_param_set(save_path_, data_splits[SPLIT_IDX], {})
    else:
        for split_idx, split_info in enumerate(data_splits):
            save_path_ = f"{SAVE_PATH}/{split_idx:02d}/"
            run_one_split_with_param_set(save_path_, split_info, {})
