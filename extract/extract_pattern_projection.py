# %%
import argparse
import glob
import json
import logging
import os
import pathlib
import pickle
import shutil
import tarfile
from io import BytesIO

import joblib
import numpy as np
import torch
from scipy.spatial import distance

from misc.utils import (log_info, mkdir, multiproc_dispatcher, recur_find_ext,
                        rm_n_mkdir, rmdir)


def load_npy_tar(tar, path_in_tar):
    bytesBuffer = BytesIO()
    bytesBuffer.write(tar.extractfile(path_in_tar).read())
    bytesBuffer.seek(0)
    return np.load(bytesBuffer, allow_pickle=False)


def load_subject(run_idx, subject_info, add_slide_info=True):
    subset_code, slide_code = subject_info

    # * tar container
    # tar_path = f'{TAR_ROOT_DIR}/{subset_code}.tar'
    # ds_tar = tarfile.open(tar_path)
    # path_in_tar = f'./{slide_code}.features.npy'
    # features = load_npy_tar(ds_tar, path_in_tar)
    # path_in_tar = f'./{slide_code}.position.npy'
    # position = load_npy_tar(ds_tar, path_in_tar).tolist()
    # ds_tar.close()
    # * normal path
    path = f'{FEAT_ROOT_DIR}/{slide_code}.features.npy'
    features = np.load(path)
    path = f'{FEAT_ROOT_DIR}/{slide_code}.position.npy'
    position = np.load(path).tolist()

    if add_slide_info:
        position = [[subset_code, slide_code, v] for v in position]
    features = np.squeeze(features)
    features /= np.linalg.norm(features, axis=-1, keepdims=True)
    return run_idx, features, position


def transform_one_subject(
        run_idx,
        subject_info,
        centroid_feat_list,
        root_save_path=None,
        scaler=None):
    subset_code, slide_code = subject_info
    _, roi_feat_list, roi_pos_list = load_subject(0, subject_info, add_slide_info=False)

    if scaler is not None:
        roi_feat_list = scaler.transform(roi_feat_list)
    roi_feat_list /= np.linalg.norm(roi_feat_list, axis=-1, keepdims=True)

    roi_dist_list = distance.cdist(roi_feat_list, centroid_feat_list)
    del roi_feat_list

    save_path = f'{root_save_path}/{slide_code}.dist.npy'
    np.save(save_path, roi_dist_list)

    roi_label_list = np.argmin(roi_dist_list, axis=-1)
    save_path = f'{root_save_path}/{slide_code}.label.npy'
    np.save(save_path, roi_label_list)
    return run_idx, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--SUBSET_CODE', type=str, default='ACDC')
    parser.add_argument('--CLUSTER_CODE', type=str, default='[method=1]')
    parser.add_argument('--ATLAS_ROOT_DATA', type=str, default='[rootData=tcga]')
    parser.add_argument('--ATLAS_SOURCE_TISSUE', type=str, default='[sourceTissue=Normal+LUAD+LUSC]')
    parser.add_argument('--FEATURE_CODE', type=str, default='[SWAV]-[mpp=0.50]-[512-256]')
    parser.add_argument('--NUM_EPOCHS', type=int, default=10)
    args = parser.parse_args()
    print(args)

    THRESHOLD = 0.50
    num_workers = 0
    WORKSPACE_DIR = 'exp_output/storage/a100/'

    CLUSTER_CODE = args.CLUSTER_CODE
    SUBSET_CODE = args.SUBSET_CODE
    FEATURE_CODE = args.FEATURE_CODE
    ATLAS_ROOT_DATA = args.ATLAS_ROOT_DATA
    ATLAS_SOURCE_TISSUE = args.ATLAS_SOURCE_TISSUE

    FEAT_ROOT_DIR = (
        f'exp_output/storage/a100/features/'
        f'{FEATURE_CODE}/{SUBSET_CODE}/'
    )
    CLUSTER_DIR = (
        f'{WORKSPACE_DIR}/cluster_filtered-{THRESHOLD:0.2f}/'
        # f'/ablation/[epochs={args.NUM_EPOCHS}]/'
        f'{ATLAS_SOURCE_TISSUE}/{FEATURE_CODE}/{ATLAS_ROOT_DATA}/'
        f'{CLUSTER_CODE}/'
    )
    SAVE_DIR = f'{CLUSTER_DIR}/transformed/{SUBSET_CODE}/'
    rm_n_mkdir(SAVE_DIR)
    # rmdir(SAVE_DIR)
    # exit()

    model = joblib.load(f'{CLUSTER_DIR}/model.dat')
    centroid_feat_list = model.cluster_centers_

    scaler = None
    if os.path.exists(f'{CLUSTER_DIR}/scaler.dat'):
        scaler = joblib.load(f'{CLUSTER_DIR}/scaler.dat')
    assert scaler is None

    data_split = joblib.load('exp_output/split/subset_info.dat')
    subject_info_list = [v[0] for v in data_split[SUBSET_CODE]]

    run_list = [[
        transform_one_subject,
        subject_info, centroid_feat_list, SAVE_DIR, scaler]
        for subject_info in subject_info_list]
    multiproc_dispatcher(run_list, num_workers, crash_on_exception=True)

    # !
    # num_workers = 0
    # paths = recur_find_ext(FEAT_ROOT_DIR, '.npy')
    # subject_info_list = [
    #     pathlib.Path(v).stem.split('.')[0]
    #     for v in paths
    # ]
    # subject_info_list = [
    #     ['ACDC', v] for v in np.unique(subject_info_list)
    # ]

    # run_list = [[
    #     transform_one_subject,
    #     subject_info, centroid_feat_list, SAVE_DIR, scaler]
    #     for subject_info in subject_info_list]
    # multiproc_dispatcher(run_list, num_workers, crash_on_exception=True)
