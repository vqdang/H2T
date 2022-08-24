

import os
import re
import cv2
import numpy as np
import pathlib

from misc.utils import cropping_center, rm_n_mkdir, mkdir, color_mask
from misc.wsi_handler import get_file_handler


####
def recur_find_ext(root_dir, ext):
    """
    recursively find all files in directories end with the `ext`
    such as `ext='.png'`
    """
    file_path_list = []
    for cur_path, dir_list, file_list in os.walk(root_dir):
        for file_name in file_list:
            file_ext = pathlib.Path(file_name).suffix
            if file_ext in ext:
                full_path = os.path.join(cur_path, file_name)
                file_path_list.append(full_path)
    file_path_list.sort()
    return file_path_list


####
def get_tissue_mask(root_dir, wsi_code):
    msk_path_png = '%s/%s.png' % (root_dir, wsi_code)
    msk_path_jpg = '%s/%s.jpg' % (root_dir, wsi_code)
    if (os.path.exists(msk_path_jpg)
            or os.path.exists(msk_path_png)):

        msk_path = (
            msk_path_png
            if os.path.exists(msk_path_png)
            else msk_path_jpg)
        msk = imread(msk_path)
        new_msk = np.zeros(msk.shape[:2], dtype=np.int32)

        new_msk[color_mask(msk, (255,   0,   0))] = 255  # tumor area
        new_msk[color_mask(msk, (255, 255, 255))] = 175  # normal area
        return new_msk.astype(np.int32)
    return None


####
def get_patch_top_left(img_shape, input_size, output_size, stride_size):
    """
    return in yx
    """
    in_out_diff = input_size - output_size
    # generating subpatches index from orginal
    output_tl_y_list = np.arange(0, img_shape[0], stride_size[0], dtype=np.int32)
    output_tl_x_list = np.arange(0, img_shape[1], stride_size[1], dtype=np.int32)
    output_tl_y_list, output_tl_x_list = np.meshgrid(output_tl_y_list, output_tl_x_list)
    output_tl = np.stack([output_tl_y_list.flatten(), output_tl_x_list.flatten()], axis=-1)
    output_br = output_tl + output_size[None]
    input_tl = output_tl - in_out_diff // 2
    input_br = input_tl + input_size[None]
    sel = (
        np.any(input_br > img_shape, axis=-1)
        | np.any(input_tl < 0, axis=-1))

    return input_tl[~sel]


####
def select_in_mask(hires_tl_list, lores_mask, down_sample, patch_shape):
    """
    Select valid patches from the list of input patch information.
    """
    def check_valid(tl, wsi_mask):
        tl = np.rint(tl).astype(np.int64)
        output_roi = wsi_mask[
            tl[0]: tl[0]+patch_shape[0],
            tl[1]: tl[1]+patch_shape[1],
        ]
        return np.sum(output_roi) > area

    patch_shape = (patch_shape * down_sample).astype(np.int32)
    # area = np.prod(patch_shape) * 0.01
    area = 0
    valid_indices = [check_valid(tl * down_sample, lores_mask)
                     for tl in hires_tl_list]
    # somehow multiproc is slower than single thread
    valid_indices = np.array(valid_indices)
    sel = np.nonzero(valid_indices)[0]
    return sel


####
def read_array(arr, tl, br, cval=0):
    # ! TODO: extend to allow reading pass top and left
    new_br = br.copy()
    if br[0] > arr.shape[0]:
        new_br[0] = arr.shape[0]
    if br[1] > arr.shape[1]:
        new_br[1] = arr.shape[1]
    request_shape = br - tl

    region = arr[tl[0]:new_br[0], tl[1]:new_br[1]]
    canvas = np.full(request_shape, cval, dtype=np.uint8)
    canvas[:region.shape[0], :region.shape[1]] = region
    return canvas


####
to2d = lambda x: np.array([x, x]) # NOQA
imread = lambda path: cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) # NOQA
imwrite = lambda path, img: cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # NOQA

# *
STORAGE_DIR = 'exp_output/storage/dgx/'
OUT_DIR = f'{STORAGE_DIR}/UCL/cache/'

WSI_ROOT_DIR = '../../../dataset/'
dir_dict = {
    'UCL-Ki67': [
        f'{WSI_ROOT_DIR}/UCL_STAMPEDE/Ki67/',
        'exp_output/UCL/dataset/Ki67/ann/dan_berney/mask_merged/'],
}
subset_list = ['UCL-Ki67']

proc_mpp = 0.5
patch_input_shape = to2d(2048)
patch_output_shape = to2d(1024)
overlap_shape = to2d(512)

####
for subset_name in subset_list:
    wsi_root_dir = dir_dict[subset_name][0]
    msk_root_dir = dir_dict[subset_name][1]
    wsi_code_list = recur_find_ext(msk_root_dir, ['.png'])
    wsi_code_list = [pathlib.Path(v).stem for v in wsi_code_list]
    for wsi_code in wsi_code_list:

        wsi_path = None
        for ext in ['.ndpi', '.svs']:
            path = f'{wsi_root_dir}/{wsi_code}{ext}'
            if os.path.exists(path):
                wsi_path = path
                break
        assert wsi_path is not None, f'{wsi_root_dir}/{wsi_code}'

        wsi_tissue_msk = get_tissue_mask(msk_root_dir, wsi_code)
        assert wsi_tissue_msk is not None

        wsi_out_dir = f'{OUT_DIR}/{subset_name}/{wsi_code}/'
        rm_n_mkdir(f'{wsi_out_dir}/imgs/')
        rm_n_mkdir(f'{wsi_out_dir}/msks/')

        wsi_handler = get_file_handler(wsi_path, pathlib.Path(wsi_path).suffix)
        wsi_hw = wsi_handler.get_dimensions(read_mpp=proc_mpp)[::-1]
        wsi_handler.prepare_reading(read_mpp=proc_mpp)

        msk_scale = np.array(wsi_tissue_msk.shape[:2]) / np.array(wsi_hw)
        msk_scale = msk_scale[0]

        patch_top_left_list = get_patch_top_left(
            wsi_hw, patch_input_shape, patch_output_shape, overlap_shape)

        wsi_tissue_msk = wsi_tissue_msk.astype(np.uint8)
        wsi_tissue_msk = cv2.resize(
                            wsi_tissue_msk,
                            tuple(wsi_hw[::-1].tolist()),
                            interpolation=cv2.INTER_NEAREST)

        new_list = []
        for top_left in patch_top_left_list:
            bot_right = top_left + patch_input_shape
            msk = read_array(wsi_tissue_msk, top_left, bot_right, cval=0)

            sel = msk >= 175
            if sel.sum() / np.prod(sel.shape) > 0.5:
                new_list.append(top_left)

        print(
            f'{wsi_code} #patches:',
            f'{len(new_list)}/{len(patch_top_left_list)}')
        patch_top_left_list = new_list
        for top_left in patch_top_left_list:
            img = wsi_handler.read_region(top_left[::-1], patch_input_shape)
            bot_right = top_left + patch_input_shape
            msk = read_array(wsi_tissue_msk, top_left, bot_right, cval=0)

            assert msk.shape[0] == img.shape[0]
            assert msk.shape[1] == img.shape[1]

            patch_code = f'{top_left[0]:06d}_{top_left[1]:06d}'
            img_path = f'{wsi_out_dir}/imgs/{patch_code}.png'
            msk_path = f'{wsi_out_dir}/msks/{patch_code}.png'
            imwrite(img_path, img)
            imwrite(msk_path, msk)
        print(wsi_out_dir)
