
import glob
import json
import os
import pathlib
import re
from collections import OrderedDict
import argparse
import copy

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn import metrics
from misc.utils import recur_find_ext, multiproc_dispatcher


def get_best_stat(stat_path, stat_name, start_epoch=0, stop_epoch=0):
    ext = pathlib.Path(stat_path).suffix
    if ext == '.dat':
        info = joblib.load(stat_path)
    elif ext == '.json':
        with open(stat_path) as stat_file:
            info = json.load(stat_file)

    epoch_val_list = []
    for idx, (tracker_idx, v) in enumerate(info.items()):
        tracker_idx = int(tracker_idx)
        if (tracker_idx < start_epoch
                or tracker_idx > stop_epoch):
            continue

        stat_dict = OrderedDict()
        for ds_code in ['valid', 'test']:
            epoch_info = info[str(tracker_idx)]
            true = np.array(epoch_info[f'infer-{ds_code}-true'])
            prob = np.array(epoch_info[f'infer-{ds_code}-pred'])

            if NUM_CLASSES == 2:
                auroc = metrics.roc_auc_score(true, prob[:, 1])
                stat_dict[f'{ds_code}-auroc'] = auroc

            mean_ap = np.mean([
                metrics.average_precision_score(
                    (true == i).astype(np.int32), prob[:, i])
                for i in range(NUM_CLASSES)
            ])
            stat_dict[f'{ds_code}-map'] = mean_ap
        epoch_val_list.append([tracker_idx, stat_dict])

    epoch_val_list = sorted(epoch_val_list, key=lambda x: x[1][stat_name])
    stat_dict = dict(epoch_val_list[-1][1])
    stat_dict['epoch'] = str(epoch_val_list[-1][0])
    # print(str(epoch_val_list[-1][0]))
    return stat_dict
####

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--TASK_CODE', type=str, default='LUAD-LUSC')
    parser.add_argument('--ATLAS_SOURCE_TISSUE', type=str, default='[sourceTissue=None]')
    parser.add_argument('--ATLAS_ROOT_DATA', type=str, default='[rootData=tcga]')
    args = parser.parse_args()

    NUM_CLASSES_MAP = {
        'NORMAL-TUMOR': 2,
        'LUAD-LUSC': 2,
        'NORMAL-LUAD-LUSC': 3
    }
    TASK_CODE = args.TASK_CODE
    NUM_CLASSES = NUM_CLASSES_MAP[TASK_CODE]

    if NUM_CLASSES == 2:
        METRICS = ['auroc', 'AP']
    else:
        METRICS = ['AP']

    stat_name = 'test-map'

    root_dir = (
        f'exp_output/storage/a100/downstream_filtered-0.50/'
        f'{args.ATLAS_SOURCE_TISSUE}/'
    )

    stat_path_list = recur_find_ext(root_dir, ['.dat', '.json'])
    stat_path_list = [
        v for v in stat_path_list
        if all(i in v for i in 
            ['stats', args.ATLAS_ROOT_DATA, f'/{TASK_CODE}/'])
    ]
    stat_path_list = [
        v for v in stat_path_list
        if any(i in v for i in
            ['clam'])
    ]
    assert len(stat_path_list) > 0

    run_info = [
        [get_best_stat, stat_path, stat_name, 2, 50]
        for stat_path in stat_path_list
    ]
    results = multiproc_dispatcher(
        run_info, num_workers=4,
        crash_on_exception=False
    )

    val_list = []
    metric_list = None
    score_dict = OrderedDict()
    for run_idx, stat_path in enumerate(stat_path_list):

        best_stat = results[run_idx]
        if best_stat is None:
            continue

        version_code = stat_path.replace(root_dir, '')
        version_code = version_code.split('/')[:-3]
        version_code = '/'.join(version_code)

        version_code = version_code.replace('-[512-256]', '')

        sub_dict = copy.deepcopy(best_stat[0])
        if sub_dict is None:
            continue
        sub_dict.pop('epoch', None)
        if version_code not in score_dict:
            score_dict[version_code] = []
        score_dict[version_code].append(np.array(list(sub_dict.values())))

    #
    feature_codes = [
        # '[SUPERVISE]-[mpp=0.25]',
        '[SUPERVISE]-[mpp=0.50]',
        # '[SWAV]-[mpp=0.25]',
        '[SWAV]-[mpp=0.50]',
    ]

    #
    wsi_desc_codes = [
        # 'transformer-1',
        # 'transformer-2'
        'clam'
    ]
    exp_codes = []
    for feature_code in feature_codes:
        for wsi_desc_code in wsi_desc_codes:
            exp_codes.append(
                f'{feature_code}/'
                f'{args.ATLAS_ROOT_DATA}/[method=None]/'
                f'{args.TASK_CODE}/{wsi_desc_code}'
            )

    for k in exp_codes:
        if k not in score_dict:
            print('-')
            continue
        scores = score_dict[k]
        if len(scores) != 5:
            print('-')
            continue
        mu = np.mean(scores, axis=0)
        va = np.std(scores, axis=0)
        num_metrics = mu.shape[-1]
        for metric_idx in range(num_metrics):
            print(
                f'{mu[metric_idx]:0.4f}±{va[metric_idx]:0.4f}', end='')
            if metric_idx == num_metrics-1:
                print('')
            else:
                print('\t', end='')

    print(*exp_codes, sep='\n')