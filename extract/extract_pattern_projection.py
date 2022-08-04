import argparse
import copy
import glob
import itertools
import json
import logging
import math
import os
import re
import pathlib
import tarfile
from collections import OrderedDict
from io import BytesIO

import joblib
import numpy as np
from sklearn.neighbors import KDTree

from features.graph import DelaunayFeatures, KNNFeatures, VoronoiFeature
from misc.utils import (log_info, mkdir, multiproc_dispatcher, recur_find_ext,
                        rm_n_mkdir, rmdir)


def load_npy_tar(tar, path_in_tar):
    bytesBuffer = BytesIO()
    bytesBuffer.write(tar.extractfile(path_in_tar).read())
    bytesBuffer.seek(0)
    return np.load(bytesBuffer, allow_pickle=False)


def load_subject(run_idx, wsi_code, load_features=False):

    # * tar container
    # tar_path = f'{pathlib.Path(position_path).parent}.tar'
    # tar = tarfile.open(tar_path)
    # path_in_tar = f'./{pathlib.Path(position_path).name}'
    # bounds = load_npy_tar(tar, path_in_tar).tolist()
    # features = None
    # if features_path is not None:
    #     path_in_tar = f'./{pathlib.Path(features_path).name}'
    #     features = np.squeeze(load_npy_tar(tar, path_in_tar))
    # tar.close()
    # * normal path
    root_path = f'{FEATURE_ROOT_DIR}/{wsi_code}'
    features = (
        np.load(f'{root_path}.features.npy', mmap_mode='r')
        if load_features else None
    )
    path = f'{root_path}.position.npy'
    position = np.load(path)
    
    root_path = f'{TRANSFORMED_DIR}/{wsi_code}'
    labels = np.load(f'{root_path}.label.npy')
    distances = np.load(f'{root_path}.dist.npy')

    selection_path = f'{SELECTION_ROOT_DIR}/{wsi_code}.npy'
    selections = np.load(selection_path)
    position, features, labels, distances = [
        v[selections > 0] if v is not None else None
        for v in [position, features, labels, distances]
    ]

    return run_idx, position, features, labels, distances


def process_one_subject(
        run_id,
        wsi_code,
        out_path, 
        feat_mode,
        num_types=8,
        scaler=None):

    load_features = 'dH' in feat_mode
    _, bounds, features, labels, distances = (
        load_subject(0, wsi_code, load_features)
    )
    # if scaler is not None:
    #     features = np.squeeze(features)
    #     features = scaler.transform(features, copy=True)
    #     features = features[:, :, None, None]

    # ! this is hardcoded
    # normalize position and only use topleft
    positions = np.array(bounds)[:, :2] / 256
    assert num_types > np.max(labels), np.unique(labels)

    all_types = np.arange(0, num_types)
    src_list = np.unique(all_types)
    dst_list = np.unique(all_types)
    pair_list = list(itertools.product(src_list, dst_list))

    mode_codes = feat_mode.split('-')
    if mode_codes[0] == 'H':
        nn_type_list = labels
        (
            unique_type_list,
            nn_type_frequency,
        ) = np.unique(nn_type_list, return_counts=True)
        # repopulate for information wrt provided type because the
        # subject may not contain all possible types within the
        # image/subject/dataset
        unique_type_freqency = np.zeros(len(all_types))
        unique_type_freqency[unique_type_list] = nn_type_frequency
        xfeat = unique_type_freqency / np.sum(unique_type_freqency)
    elif mode_codes[0] == 'C':
        # immediate neighbor (3x3)
        kdtree = KDTree(positions)
        fxtor = KNNFeatures(
                    kdtree=kdtree,
                    pair_list=pair_list,
                    unique_type_list=all_types)
        stats = fxtor.transform(positions, labels, radius=2)
        xfeat = stats
    elif mode_codes[0] == 'dH':
        type_avg_feats = []
        for type_id in all_types:
            type_dists = distances[..., type_id]
            if 'fk' in feat_mode:
                # furthest k-patches
                topk = re.findall(r'.*fk([0-9]*).*', feat_mode)
                assert len(topk) == 1
                # may not be in sorted order, k-smallest value
                topk = min(int(topk[0]), type_dists.shape[0]-1)
                sel = np.argpartition(-type_dists, topk)[:topk]
            elif 'k' in feat_mode:
                # nearest k-patches
                topk = re.findall(r'.*k([0-9]*).*', feat_mode)
                assert len(topk) == 1
                # may not be in sorted order, k-smallest value
                topk = min(int(topk[0]), type_dists.shape[0]-1)
                sel = np.argpartition(type_dists, topk)[:topk]
            else:
                sel = labels == type_id

            feats = (
                features[sel] if np.sum(sel) > 0
                else np.zeros([1, *features[0].shape])
            )
            feats = np.array(feats)

            dists = (
                type_dists[sel] if np.sum(sel) > 0
                else np.zeros([1])
            )
            if 't' in feat_mode:
                pattern = r'[-+]?\d*\.\d+|\d+'  # pattern for float
                threshold = re.findall(pattern, feat_mode)
                assert len(threshold) == 1
                threshold = float(threshold[0])
                sel = dists > threshold
                feats = (
                    feats[sel] if np.sum(sel) > 0
                    else np.zeros([1, *feats[0].shape])
                )
                dists = (
                    dists[sel] if np.sum(sel) > 0
                    else np.zeros([1])
                )
                # print('here')

            if 'wn' in feat_mode:
                dists = dists[:, None, None, None]
                type_feats = np.mean(feats * dists, axis=0)
            elif 'w' in feat_mode:
                dists = 1.0 - dists[:, None, None, None]
                type_feats = np.mean(feats * dists, axis=0)
            else:
                type_feats = np.mean(feats, axis=0)
            type_avg_feats.append(type_feats)
        type_avg_feats = np.stack(type_avg_feats, axis=0)
        xfeat = type_avg_feats
    else:
        assert False, feat_mode

    xfeat = xfeat.astype(np.float32)
    np.save(out_path, xfeat)
    return run_id, None

####
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--SUBSET_CODE', type=str, default='CPTAC-LUAD')
    parser.add_argument('--CLUSTER_CODE', type=str, default='[method=1]')
    parser.add_argument('--ATLAS_ROOT_DATA', type=str, default='[rootData=tcga]')
    # parser.add_argument('--ATLAS_SOURCE_TISSUE', type=str, default='dump')
    parser.add_argument('--ATLAS_SOURCE_TISSUE', type=str, default='[sourceTissue=Normal+LUAD+LUSC]')
    parser.add_argument('--FEATURE_CODE', type=str, default='[SWAV]-[mpp=0.50]-[512-256]')
    parser.add_argument('--WSI_FEATURE_CODE', type=str, default='dH-w')
    parser.add_argument('--NUM_EPOCHS', type=int, default=10)
    args = parser.parse_args()

    num_worker = 0

    # WORKSPACE_DIR = 'exp_output/storage/nima/'
    WORKSPACE_DIR = 'exp_output/storage/a100/'

    WSI_FEATURE_CODE = args.WSI_FEATURE_CODE

    FEATURE_CODE = args.FEATURE_CODE
    SUBSET_CODE = args.SUBSET_CODE
    CLUSTER_CODE = args.CLUSTER_CODE
    ATLAS_ROOT_DATA = args.ATLAS_ROOT_DATA
    ATLAS_SOURCE_TISSUE = args.ATLAS_SOURCE_TISSUE

    THRESHOLD = 0.50
    SELECTION_ROOT_DIR = (
        f'{WORKSPACE_DIR}/features/mpp=0.25/'
        f'/selections-{THRESHOLD:0.2f}/{SUBSET_CODE}/'
    )

    FEATURE_ROOT_DIR = (
        f'{WORKSPACE_DIR}/features/{FEATURE_CODE}/{SUBSET_CODE}/'
    )

    CLUSTER_ROOT_DIR = (
        f'{WORKSPACE_DIR}/cluster_filtered-{THRESHOLD:0.2f}/'
        # f'/ablation/[epochs={args.NUM_EPOCHS}]/'
        f'{ATLAS_SOURCE_TISSUE}/{FEATURE_CODE}/{ATLAS_ROOT_DATA}/'
        f'{CLUSTER_CODE}/'
    )
    cluster_model = joblib.load(f'{CLUSTER_ROOT_DIR}/model.dat')
    NUM_TYPES = cluster_model.cluster_centers_.shape[0]

    TRANSFORMED_DIR = (
        f'{CLUSTER_ROOT_DIR}/transformed/{SUBSET_CODE}/'
    )
    OUT_DIR = (
        f'{WORKSPACE_DIR}/downstream_filtered-{THRESHOLD:0.2f}//'
        # f'/ablation/[epochs={args.NUM_EPOCHS}]/'
        f'{ATLAS_SOURCE_TISSUE}/{FEATURE_CODE}/{ATLAS_ROOT_DATA}/'
        f'{CLUSTER_CODE}/features/{WSI_FEATURE_CODE}/{SUBSET_CODE}/'
        # f'{WORKSPACE_DIR}/dump/projection/{SUBSET_CODE}/'
    )
    rm_n_mkdir(OUT_DIR)

    logging.basicConfig(
        level=logging.DEBUG,
        format='|%(asctime)s.%(msecs)03d| [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d|%H:%M:%S',
        handlers=[
            logging.FileHandler("%s/debug.log" % (OUT_DIR)),
            logging.StreamHandler()
        ]
    )
    logging.info(args)

    scaler = None
    # if 'normed' in CLUSTER_ROOT_DIR:
    #     scaler = joblib.load(f'{CLUSTER_ROOT_DIR}/scaler.dat')

    label_paths = recur_find_ext(TRANSFORMED_DIR, ['.label.npy'])
    wsi_codes = [pathlib.Path(v).name for v in label_paths]
    wsi_codes = [v.replace('.label.npy', '') for v in wsi_codes]

    out_paths = [f'{OUT_DIR}/{v}.npy' for v in wsi_codes]
    input_info_list = list(
        zip(wsi_codes, out_paths))
    run_list = [
        [process_one_subject,
        *(list(info) + [WSI_FEATURE_CODE, NUM_TYPES, scaler])]
        for idx, info in enumerate(input_info_list)]
    multiproc_dispatcher(run_list, nr_worker=num_worker, crash_on_exception=True)
    logging.info('Finish')