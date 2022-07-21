from typing import Union
import copy
import inspect
import itertools
import json
import logging
import os
import inspect
import collections
import pathlib
import re
import shutil
import sys
from PIL import ImageColor
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, as_completed, wait

import cv2
import numpy as np
from termcolor import colored
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

PERCEPTIVE_COLORS = [
    # "#000000", # ! dont use black
    "#FFFF00",
    "#1CE6FF",
    "#FF34FF",
    "#FF4A46",
    "#008941",
    "#006FA6",
    "#A30059",
    "#FFDBE5",
    "#7A4900",
    "#0000A6",
    "#63FFAC",
    "#B79762",
    "#004D43",
    "#8FB0FF",
    "#997D87",
    "#5A0007",
    "#809693",
    "#FEFFE6",
    "#1B4400",
    "#4FC601",
    "#3B5DFF",
    "#4A3B53",
    "#FF2F80",
    "#61615A",
    "#BA0900",
    "#6B7900",
    "#00C2A0",
    "#FFAA92",
    "#FF90C9",
    "#B903AA",
    "#D16100",
    "#DDEFFF",
    "#000035",
    "#7B4F4B",
    "#A1C299",
    "#300018",
    "#0AA6D8",
    "#013349",
    "#00846F",
    "#372101",
    "#FFB500",
    "#C2FFED",
    "#A079BF",
    "#CC0744",
    "#C0B9B2",
    "#C2FF99",
    "#001E09",
    "#00489C",
    "#6F0062",
    "#0CBD66",
    "#EEC3FF",
    "#456D75",
    "#B77B68",
    "#7A87A1",
    "#788D66",
    "#885578",
    "#FAD09F",
    "#FF8A9A",
    "#D157A0",
    "#BEC459",
    "#456648",
    "#0086ED",
    "#886F4C",
    "#34362D",
    "#B4A8BD",
    "#00A6AA",
    "#452C2C",
    "#636375",
    "#A3C8C9",
    "#FF913F",
    "#938A81",
    "#575329",
    "#00FECF",
    "#B05B6F",
    "#8CD0FF",
    "#3B9700",
    "#04F757",
    "#C8A1A1",
    "#1E6E00",
    "#7900D7",
    "#A77500",
    "#6367A9",
    "#A05837",
    "#6B002C",
    "#772600",
    "#D790FF",
    "#9B9700",
    "#549E79",
    "#FFF69F",
    "#201625",
    "#72418F",
    "#BC23FF",
    "#99ADC0",
    "#3A2465",
    "#922329",
    "#5B4534",
    "#FDE8DC",
    "#404E55",
    "#0089A3",
    "#CB7E98",
    "#A4E804",
    "#324E72",
    "#6A3A4C",
    "#83AB58",
    "#001C1E",
    "#D1F7CE",
    "#004B28",
    "#C8D0F6",
    "#A3A489",
    "#806C66",
    "#222800",
    "#BF5650",
    "#E83000",
    "#66796D",
    "#DA007C",
    "#FF1A59",
    "#8ADBB4",
    "#1E0200",
    "#5B4E51",
    "#C895C5",
    "#320033",
    "#FF6832",
    "#66E1D3",
    "#CFCDAC",
    "#D0AC94",
    "#7ED379",
    "#012C58",
]
PERCEPTIVE_COLORS_RGB = [ImageColor.getcolor(v, "RGB") for v in PERCEPTIVE_COLORS]
PERCEPTIVE_COLORS = np.array(PERCEPTIVE_COLORS)[3:]
PERCEPTIVE_COLORS_RGB = np.array(PERCEPTIVE_COLORS_RGB)[3:]
PERCEPTIVE_COLORS_RGB = PERCEPTIVE_COLORS_RGB.astype(np.uint8)


def imread(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def imwrite(path, img):
    return cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def channel_masks(arr, channel_vals: Union[list, np.ndarray]):
    """Assume last channel."""
    sel = np.full(arr.shape[:-1], True, dtype=bool)
    for idx, val in enumerate(channel_vals):
        sel &= arr[..., idx] == val
    return sel


def patch2tile(patches, num_cols, cval=255, border=1, border_color=255):
    """Tiling a list of patches, each patch may be of differen shapes."""
    num_channels = patches[0].shape[-1]
    shapes = [v.shape[:2] for v in patches]
    max_hw = np.max(np.array(shapes))
    padded = [center_pad_to_shape(v, [max_hw, max_hw], cval=cval) for v in patches]
    num_rows = int(np.ceil(len(padded) / num_cols))
    num_elements = num_rows * num_cols - len(patches)
    empty_placements = [np.zeros([max_hw, max_hw, num_channels])] * num_elements

    max_hw = max_hw + border
    padded_patches = padded + empty_placements
    padded_patches = [
        center_pad_to_shape(v, [max_hw, max_hw], cval=border_color)
        for v in padded_patches
    ]
    padded_patches = np.array(padded_patches)
    padded_patches = np.reshape(
        padded_patches, [num_rows, num_cols, max_hw, max_hw, num_channels]
    )
    padded_patches = np.transpose(padded_patches, [0, 2, 1, 3, 4])
    padded_patches = np.reshape(
        padded_patches, [num_rows * max_hw, num_cols * max_hw, num_channels]
    )
    return padded_patches


def center_pad_to_shape(img, size, cval=255):
    """Pad input image."""
    # rounding down, add 1
    pad_h = size[0] - img.shape[0]
    pad_w = size[1] - img.shape[1]
    pad_h = (pad_h // 2, pad_h - pad_h // 2)
    pad_w = (pad_w // 2, pad_w - pad_w // 2)
    if len(img.shape) == 2:
        pad_shape = (pad_h, pad_w)
    else:
        pad_shape = (pad_h, pad_w, (0, 0))
    img = np.pad(img, pad_shape, "constant", constant_values=cval)
    return img


def flatten_list(a_list):
    """Flatten a nested list."""
    return list(itertools.chain(*a_list))


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_json(path):
    with open(path, "r") as fptr:
        return json.load(fptr)


def remap_label(pred, by_size=False):
    """Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3]
    not [0, 2, 4, 6]. The ordering of instances (which one comes first)
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID.

    Args:
        pred (ndarray): the 2d array contain instances where each instances is marked
            by non-zero integer.
        by_size (bool): renaming such that larger nuclei have a smaller id (on-top).

    Returns:
        new_pred (ndarray): Array with continguous ordering of instances.

    """
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred


def cropping_center(x: np.ndarray, crop_shape, batch=False):
    """Crop an array at the centre with specified dimensions.

    Args:
        batch (bool): If `True`, input array is assumed to be
            of shape `NxHxWxC` where `N` is the number of image
            within the array.

    """
    orig_shape = x.shape
    if not batch:
        h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
        x = x[h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    else:
        h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
        x = x[:, h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    return x


def rm_n_mkdir(dir_path):
    """Remove and make directory."""
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def rmdir(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    return


def mkdir(dir_path):
    """Make directory if it does not exist."""
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def recur_find_ext(root_dir, ext_list):
    """Recursively find all files in directories end with the `ext` such as `ext='.png'`.

    Args:
        root_dir (str): Root directory to grab filepaths from.
        ext_list (list): File extensions to consider.

    Returns:
        file_path_list (list): sorted list of filepaths.
    """
    # turn "." into a literal character in regex
    patterns = [v.replace(".", "\.") for v in ext_list]
    patterns = [f".*{v}$" for v in patterns]

    file_path_list = []
    for cur_path, dir_list, file_list in os.walk(root_dir):
        for file_name in file_list:
            has_ext_flags = [
                re.match(pattern, file_name) is not None for pattern in patterns
            ]
            if any(has_ext_flags):
                full_path = os.path.join(cur_path, file_name)
                file_path_list.append(full_path)
    file_path_list.sort()
    return file_path_list


def rm_n_mkdir(dir_path):
    """Remove and then make a new directory."""
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def get_bounding_box(img: np.ndarray) -> np.ndarray:
    """Get the bounding box coordinates of a binary input- assumes a single object.

    Args:
        img: input binary image.

    Returns:
        bounding box coordinates

    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return np.array([rmin, rmax, cmin, cmax])


def print_dir(root_path):
    """Print out the entire directory content."""
    for root, subdirs, files in os.walk(root_path):
        print(f"-{root}")
        for subdir in subdirs:
            print(f"--D-{subdir}")
        for filename in files:
            file_path = os.path.join(root, filename)
            print(f"--F-{file_path}")


def save_as_json(data, save_path):
    """Save data to a json file.

    The function will deepcopy the `data` and then jsonify the content
    in place. Support data types for jsonify consist of `str`, `int`, `float`,
    `bool` and their np.ndarray respectively.

    Args:
        data (dict or list): Input data to save.
        save_path (str): Output to save the json of `input`.

    """
    shadow_data = copy.deepcopy(data)

    # make a copy of source input
    def walk_list(lst):
        """Recursive walk and jsonify in place."""
        for i, v in enumerate(lst):
            if isinstance(v, dict):
                walk_dict(v)
            elif isinstance(v, list):
                walk_list(v)
            elif isinstance(v, np.ndarray):
                v = v.tolist()
                walk_list(v)
            elif isinstance(v, np.generic):
                v = v.item()
            elif v is not None and not isinstance(v, (int, float, str, bool)):
                raise ValueError(f"Value type `{type(v)}` `{v}` is not jsonified.")
            lst[i] = v

    def walk_dict(dct):
        """Recursive walk and jsonify in place."""
        for k, v in dct.items():
            if isinstance(v, dict):
                walk_dict(v)
            elif isinstance(v, list):
                walk_list(v)
            elif isinstance(v, np.ndarray):
                v = v.tolist()
                walk_list(v)
            elif isinstance(v, np.generic):
                v = v.item()
            elif v is not None and not isinstance(v, (int, float, str, bool)):
                raise ValueError(f"Value type `{type(v)}` `{v}` is not jsonified.")
            if not isinstance(k, (int, float, str, bool)):
                raise ValueError(f"Key type `{type(k)}` `{k}` is not jsonified.")
            dct[k] = v

    if isinstance(shadow_data, dict):
        walk_dict(shadow_data)
    elif isinstance(shadow_data, list):
        walk_list(shadow_data)
    else:
        raise ValueError(f"`data` type {type(data)} is not [dict, list].")
    with open(save_path, "w") as handle:
        json.dump(shadow_data, handle, indent=4, sort_keys=True)


def wrap_func(idx, func, *args):
    """A wrapper so that any functions can be run
    with `dispatch_processing`.
    """
    try:
        return idx, func(*args)
    except Exception as exception_obj:
        # cache the exception stack trace
        # so that we can print out later if need
        print(exception_obj)
        exception_info = sys.exc_info()
        return [exception_obj, exception_info], idx, None


def dispatch_processing(
    data_list, num_workers=0, show_progress=True, crash_on_exception=False
):
    """
    data_list is alist of [[func, arg1, arg2, etc.]]
    Resutls are alway sorted according to source position
    """

    def handle_wrapper_results(result):
        if len(result) == 3 and crash_on_exception:
            exception_obj, exception_info = result[0]
            logging.info(exception_obj)
            del exception_info
            raise exception_obj
        elif len(result) == 3:
            result = result[1:]
        return result

    executor = None if num_workers <= 1 else ProcessPoolExecutor(num_workers)

    result_list = []
    future_list = []

    progress_bar = tqdm(
        total=len(data_list), ascii=True, position=0, disable=not show_progress
    )
    with logging_redirect_tqdm([logging.getLogger()]):
        for run_idx, dat in enumerate(data_list):
            func = dat[0]
            args = dat[1:]
            if num_workers > 1:
                future = executor.submit(wrap_func, run_idx, func, *args)
                future_list.append(future)
            else:
                # ! assume 1st return is alwasy run_id
                result = wrap_func(run_idx, func, *args)
                result = handle_wrapper_results(result)
                result_list.append(result)
                progress_bar.update()

        if num_workers > 1:
            for future in as_completed(future_list):
                if future.exception() is not None:
                    if crash_on_exception:
                        raise future.exception()
                    logging.info(future.exception())
                    continue
                result = future.result()
                result = handle_wrapper_results(result)
                result_list.append(result)
                progress_bar.update()
            executor.shutdown()
        progress_bar.close()

    # shutdown the pool, cancels scheduled tasks, returns when running tasks complete
    # if executor:
    #     executor.shutdown(wait=True, cancel_futures=True)

    result_list = sorted(result_list, key=lambda k: k[0])
    result_list = [v[1] for v in result_list]
    return result_list


def convert_pytorch_checkpoint(net_state_dict: dict) -> dict:
    """Convert Pytorch checkpoint nested in nn.DataParallel to single GPU mode."""
    variable_name_list = list(net_state_dict.keys())
    is_in_parallel_mode = all(v.split(".")[0] == "module" for v in variable_name_list)
    if is_in_parallel_mode:
        colored_word = colored("WARNING", color="red", attrs=["bold"])
        print(
            (
                f"{colored_word}:"
                "Detect checkpoint saved in data-parallel mode."
                " Converting saved model to single GPU mode."
            ).rjust(80)
        )
        net_state_dict = {
            ".".join(k.split(".")[1:]): v for k, v in net_state_dict.items()
        }
    return net_state_dict


def log_debug(msg):
    (
        frame,
        filename,
        line_number,
        function_name,
        lines,
        index,
    ) = inspect.getouterframes(inspect.currentframe())[1]
    line = lines[0]
    indentation_level = line.find(line.lstrip())
    logging.debug("{i} {m}".format(i="." * indentation_level, m=msg))


def log_info(msg):
    (
        frame,
        filename,
        line_number,
        function_name,
        lines,
        index,
    ) = inspect.getouterframes(inspect.currentframe())[1]
    line = lines[0]
    indentation_level = line.find(line.lstrip())
    logging.info("{i} {m}".format(i="." * indentation_level, m=msg))


def set_logger(path):
    logging.basicConfig(level=logging.INFO)
    # * reset logger handler
    log_formatter = logging.Formatter(
        "|%(asctime)s.%(msecs)03d| [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d|%H:%M:%S",
    )
    log = logging.getLogger()  # root logger
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)
    new_hdlr_list = [logging.FileHandler(path), logging.StreamHandler()]
    for hdlr in new_hdlr_list:
        hdlr.setFormatter(log_formatter)
        log.addHandler(hdlr)


def max_resolution(a, b):
    assert a["units"] == b["units"]
    if a["units"] == "mpp":
        return a if a["resolution"] < b["resolution"] else b
    elif a["units"] == "power":
        return b if a["resolution"] < b["resolution"] else a
    else:
        assert False, f"Unknown resolution units: `{a['units']}`"


def convert_to_resolution(dat, resolution):
    el_resolution = dat["element-resolution"]
    el_dat = dat["elements"]
    if resolution["units"] == "mpp":
        fx = el_resolution["resolution"] / resolution["resolution"]
    elif resolution["units"] == "power":
        fx = resolution["resolution"] / el_resolution["resolution"]
        assert False, f"Unknown resolution units: `{resolution['units']}`"

    el_dat_ = {}
    for el_id, el in el_dat.items():
        el_ = {"type": el["type"]}
        for geo in ["box", "contour", "centroid"]:
            el_[geo] = (np.array(el[geo]) * fx).astype(np.int32)
        el_dat_[el_id] = el_

    return {
        "source-image-resolution": dat["source-image-resolution"],
        "element-resolution": resolution,
        "elements": el_dat_,
    }
