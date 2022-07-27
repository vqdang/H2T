
import argparse
import pathlib
import sys

import cv2
import numpy as np

sys.path.append('/root/storage_1/workspace/tiatoolbox/')

from tiatoolbox.wsicore.wsireader import OpenSlideWSIReader, VirtualWSIReader

from utils import multiproc_dispatcher, recur_find_ext, rm_n_mkdir


def process_one(path):
    base_name = pathlib.Path(path).stem
    base_name = base_name.replace('.position', '')

    mask_path = f'{MASK_DIR}/masks/{base_name}.png'
    position_path = f'{POSITION_DIR}/{base_name}.position.npy'

    bounds = np.load(position_path) * fx
    reader = VirtualWSIReader(mask_path, mode='bool')

    selections = np.zeros(bounds.shape[0], dtype=np.int32)
    for idx, bound in enumerate(bounds):
        img = reader.read_bounds(
            bound,
            interpolation="optimise",
            coord_space='resolution',
            resolution=1.0,
            units='baseline',
        )
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = np.array(img > 0)
        selections[idx] = (np.sum(img) / np.size(img)) >= THRESHOLD
    np.save(f'{SAVE_DIR}/{base_name}.npy', selections)    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--TISSUE', type=str, default='TCGA-LUAD')
    parser.add_argument('--THRESHOLD', type=float, default=0.5)
    args = parser.parse_args()

    THRESHOLD = args.THRESHOLD
    TISSUE = args.TISSUE
    ROOT_DIR = (
        'exp_output/storage/a100/features/[SWAV]-[mpp=0.25]-[512-256]'
    )
    POSITION_RESOLUTION = {'resolution': 0.25, 'units': 'mpp'}
    POSITION_DIR = f'{ROOT_DIR}/{TISSUE}/'

    ROOT_DIR = (
        'exp_output/storage/a100/tissue_segment/FCN-Generic-v0.1/predictions/'
    )
    MASK_RESOLUTION = {'resolution': 8.0, 'units': 'mpp'}
    MASK_DIR = f'{ROOT_DIR}/{TISSUE}/'

    SAVE_DIR = (
        f'exp_output/storage/a100/features/'
        f'/mpp=0.25/selections-{THRESHOLD:0.2f}/{TISSUE}/'
    )
    rm_n_mkdir(SAVE_DIR)

    fx = POSITION_RESOLUTION['resolution'] / MASK_RESOLUTION['resolution']

    paths = recur_find_ext(POSITION_DIR, ['.npy'])
    paths = [v for v in paths if '.position.npy' in v]
    runs = [
        [process_one, path]
        for path in paths
    ]

    multiproc_dispatcher(runs, num_workers=32, crash_on_exception=True)
    # print(np.sum(selections), bounds.shape[0])
