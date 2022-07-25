import argparse
import collections
import csv
import glob
import importlib
import json
import logging
import os
import pathlib
import re
from collections import OrderedDict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch.utils.data
import yaml

import dataset
from misc.utils import mkdir, recur_find_ext, rm_n_mkdir


####
def load_yaml(path):
    with open(path) as fptr:
        info = yaml.full_load(fptr)
    return info

####
def update_nested_dict(orig_dict, new_dict):
    for key, val in new_dict.items():
        if isinstance(val, collections.Mapping):
            tmp = update_nested_dict(orig_dict.get(key, { }), val)
            orig_dict[key] = tmp
        # elif isinstance(val, list):
        #     orig_dict[key] = (orig_dict.get(key, []) + val)
        else:
            orig_dict[key] = new_dict[key]
    return orig_dict

####
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--gpu', type=str, default='0,1,2,3')
    parser.add_argument('--run_mode', type=str, default='train')
    args = parser.parse_args()
    print(args)
    # exit()

    seed = 5
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    root_outdir = 'exp_output/UCL/tumor/'
    model_code = 'FCN-SA-v0.0=[DanBerney]'

    ####
    split_info = {}

    # ! k66 k68
    data_root_dir = 'exp_output/storage/dgx/UCL/cache/'
    split_name_dict = {
        'train': [
            'UCL-Ki67/k00005/',
            'UCL-Ki67/k00025/',
            'UCL-Ki67/k00026/',
            'UCL-Ki67/k00029/',
            'UCL-Ki67/k00032/',
            'UCL-Ki67/k00066/',
            'UCL-Ki67/k00068/',
        ],
        'infer-he-rgb': [
            'UCL-Ki67/k00011/',
            'UCL-Ki67/k00019/',
        ],
        'infer-he-gry': [
            'UCL-Ki67/k00011/',
            'UCL-Ki67/k00019/',
        ],
    }
    for split_name, split_dir_list in split_name_dict.items():
        path_list = [
            path for sub_dir in split_dir_list
            for path in recur_find_ext(f'{data_root_dir}/{sub_dir}', ['.png'])
        ]
        img_path_list = [v for v in path_list if '/imgs/' in v]
        msk_path_list = [v.replace('/imgs/', '/msks/') for v in img_path_list]
        info_list = list(zip(img_path_list, msk_path_list))
        split_info[split_name] = info_list

    # ---

    template_paramset = load_yaml('param/paramset_template.yml')
    # paramdist_dict  = load_yaml('param/paramdist.yml')

    # repopulate loader arg according to available subset info
    loader_kwargs = {
        k: template_paramset['loader_kwargs']['train'] if 'train' in k else
           template_paramset['loader_kwargs']['infer'] for k in split_info.keys()}
    template_paramset['loader_kwargs'] = loader_kwargs
    #

    def run_one_split_with_param_set(split, param_kwargs):
        run_paramset = template_paramset.copy()
        update_nested_dict(run_paramset, param_kwargs)
        update_nested_dict(run_paramset, {'seed': seed})

        save_formatted_code = ''
        save_path = '%s/%s/%s' % (root_outdir, model_code, save_formatted_code)
        mkdir(save_path)

        logging.basicConfig(
            level=logging.INFO,
            format='|%(asctime)s.%(msecs)03d| [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d|%H:%M:%S',
            handlers=[
                logging.FileHandler("%s/debug.log" % save_path),
                logging.StreamHandler()
            ]
        )

        with open("%s/settings.yml" % save_path, "w") as fptr:
            yaml.dump(run_paramset, fptr, default_flow_style=False)

        train_loader_list = [v for v in split_info.keys() if 'train' in v]
        infer_loader_list = [v for v in split_info.keys() if not ('train' in v)]

        cfg_module = importlib.import_module('model.opt')
        cfg_getter = getattr(cfg_module, 'get_config')
        model_config = cfg_getter(
                            train_loader_list,
                            infer_loader_list,
                            **run_paramset)

        from loader import PatchDataset
        def create_dataset(run_mode=None, subset_name=None, setup_augmentor=None):
            return PatchDataset(split_info,
                        run_mode=run_mode,
                        subset_name=subset_name)

        run_kwargs = {
            'seed': seed,
            'debug': False,
            'logging': args.run_mode=='train',
            'log_dir': save_path + '/model/',
            'create_dataset': create_dataset,
            'model_config': model_config,
        }

        from run_train import RunManager
        trainer = RunManager(**run_kwargs)
        trainer.run()
        return

    run_one_split_with_param_set(split_info, {})
