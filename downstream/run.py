import argparse
import importlib
import logging
import os

import joblib
import yaml

from loader import SequenceDataset
from misc.utils import mkdir, rm_n_mkdir, rmdir, load_yaml, update_nested_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--gpu", type=str, default="0,1")
    parser.add_argument("--SPLIT_IDX", type=int, default=None)
    parser.add_argument("--run_mode", type=str, default="train")
    # parser.add_argument('--TASK_CODE', type=str, default='NORMAL-LUAD-LUSC')
    parser.add_argument("--TASK_CODE", type=str, default="LUAD-LUSC")
    parser.add_argument("--CLUSTER_CODE", type=str, default="[method=None]")
    parser.add_argument("--ATLAS_ROOT_DATA", type=str, default="[rootData=tcga]")
    parser.add_argument(
        "--ATLAS_SOURCE_TISSUE", type=str, default="[sourceTissue=None]"
    )
    # parser.add_argument('--FEATURE_CODE', type=str, default='[SWAV]-[mpp=0.50]-[512-256]')
    parser.add_argument(
        "--FEATURE_CODE", type=str, default="[SUPERVISE]-[mpp=0.50]-[512-256]"
    )
    parser.add_argument("--WSI_FEATURE_CODE", type=str, default="dH-w")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    print(args)

    seed = 5
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    WORKSPACE_DIR = "exp_output/storage/a100/"

    MODEL_CODE = "clam"
    TASK_CODE = args.TASK_CODE
    FEATURE_CODE = args.FEATURE_CODE
    CLUSTER_CODE = args.CLUSTER_CODE
    ATLAS_ROOT_DATA = args.ATLAS_ROOT_DATA
    ATLAS_SOURCE_TISSUE = args.ATLAS_SOURCE_TISSUE

    MPP = 0.50
    THRESHOLD = 0.50
    save_path = (
        f"{WORKSPACE_DIR}/downstream_filtered-{THRESHOLD:0.2f}/"
        f"{ATLAS_SOURCE_TISSUE}/{FEATURE_CODE}/{ATLAS_ROOT_DATA}/"
        f"{CLUSTER_CODE}/{TASK_CODE}/{MODEL_CODE}/"
    )
    # rmdir(save_path)

    num_classes_in_task = {
        "LUAD-LUSC": 2,
        "NORMAL-TUMOR": 2,
        "NORMAL-LUAD-LUSC": 3,
    }

    ####
    if ATLAS_ROOT_DATA == "[rootData=tcga]":
        data_split = joblib.load(
            f"exp_output/split/[{TASK_CODE.lower()}]_train=tcga_test=cptac.dat"
        )
        new_data_split = []
        for split_idx, split_info in data_split:
            _split = {
                "train": split_info["tcga-train"],
                "valid": split_info["tcga-valid"],
                "test": split_info["cptac"],
            }
            if TASK_CODE == "NORMAL-TUMOR":
                _split = {k: [[vx[0][0], vx[1]] for vx in v] for k, v in _split.items()}
            new_data_split.append(_split)
        data_split = new_data_split
        del new_data_split
    else:
        data_split = joblib.load(
            f"exp_output/split/[{TASK_CODE.lower()}]_train=cptac_test=tcga.dat"
        )
        new_data_split = []
        for split_idx, split_info in data_split:
            new_data_split.append(
                {
                    "train": split_info["cptac-train"],
                    "valid": split_info["cptac-valid"],
                    "test": split_info["tcga-d"] + split_info["tcga-f"],
                }
            )
        data_split = new_data_split
        del new_data_split

    # -------------------------------------------------------------

    # ! transformer
    # paramset = load_yaml(f'param/hopfield.yml')
    # ! clam
    paramset = load_yaml(f"param/clam.yml")

    logging.basicConfig(
        level=logging.INFO,
    )

    def run_one_split_with_param_set(save_path, split_info, param_kwargs):
        run_paramset = paramset.copy()
        update_nested_dict(run_paramset, param_kwargs)
        update_nested_dict(run_paramset, {"seed": seed})

        # repopulate loader arg according to available subset info
        loader_kwargs = {
            k: paramset["loader_kwargs"]["train"]
            if "train" in k
            else paramset["loader_kwargs"]["infer"]
            for k in data_split[0].keys()
        }
        run_paramset["loader_kwargs"] = loader_kwargs
        run_paramset["model_kwargs"]["num_types"] = num_classes_in_task[TASK_CODE]

        rm_n_mkdir(save_path)
        # * reset logger handler
        log_formatter = logging.Formatter(
            "|%(asctime)s.%(msecs)03d| [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d|%H:%M:%S",
        )
        log = logging.getLogger()  # root logger
        for hdlr in log.handlers[:]:  # remove all old handlers
            log.removeHandler(hdlr)
        new_hdlr_list = [
            logging.FileHandler("%s/debug.log" % save_path),
            logging.StreamHandler(),
        ]
        for hdlr in new_hdlr_list:
            hdlr.setFormatter(log_formatter)
            log.addHandler(hdlr)
        #

        with open(f"{save_path}/settings.yml", "w") as fptr:
            yaml.dump(run_paramset, fptr, default_flow_style=False)

        train_loader_list = [v for v in split_info.keys() if "train" in v]
        infer_loader_list = [v for v in split_info.keys() if not ("train" in v)]

        cfg_module = importlib.import_module("model.amil.opt")
        cfg_getter = getattr(cfg_module, "get_config")
        model_config = cfg_getter(train_loader_list, infer_loader_list, **run_paramset)

        def create_dataset(run_mode=None, subset_name=None, setup_augmentor=None):
            root_dir = f"{WORKSPACE_DIR}/features/{FEATURE_CODE}/"
            selection_dir = (
                f"{WORKSPACE_DIR}/features/"
                f"mpp={MPP:0.2f}/selections-{THRESHOLD:0.2f}"
            )
            ds = SequenceDataset(
                root_dir=root_dir,
                task_code=TASK_CODE,
                sample_info_list=split_info[subset_name],
                run_mode=run_mode,
                selection_dir=selection_dir,
            )
            return ds

        run_kwargs = {
            "seed": seed,
            "debug": False,
            "logging": True,
            "log_dir": save_path + "/model/",
            "create_dataset": create_dataset,
            "model_config": model_config,
        }

        from engine.manager import RunManager

        trainer = RunManager(**run_kwargs)
        trainer.run()

    # if args.SPLIT_IDX is not None:
    #     save_path_ = f'{save_path}/{args.SPLIT_IDX:02d}/'
    #     rm_n_mkdir(save_path_)
    #     run_one_split_with_param_set(
    #         save_path_, data_split[args.SPLIT_IDX], {})
    # else:
    #     assert False
    #     for split_idx, split_info in enumerate(data_split):
    #         save_path_ = f'{save_path}/{split_idx:02d}/'
    #         # if os.path.exists(save_path_): continue
    #         run_one_split_with_param_set(save_path_, split_info, {})
