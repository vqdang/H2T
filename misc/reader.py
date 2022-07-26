import os
import sys
import numpy as np

sys.path.append(os.environ["TIATOOLBOX"])

from tiatoolbox.wsicore.wsireader import VirtualWSIReader, WSIReader, WSIMeta

from misc.utils import imread


def get_reader(img_path) -> WSIReader:
    img_path = str(img_path)
    if any(v in img_path for v in [".png", ".npy"]):
        if ".png" in img_path:
            image = imread(img_path)
        elif ".npy" in img_path:
            image = np.load(img_path, mmap_mode="r")

        metadata = WSIMeta(
            axes="YXS",
            slide_dimensions=image.shape[:2][::-1],
            level_count=1,
            level_dimensions=[image.shape[:2][::-1]],
            level_downsamples=[1.0],
            mpp=[4.0, 4.0],
        )        
        reader = VirtualWSIReader(
            image,
            info=metadata,
            mode='bool'
        )
        return reader
    # elif ".tif" in img_path:
    #     return TIFFWSIReader(img_path)
    elif ".tif" in img_path:
        return WSIReader.open(img_path)
    else:
        return WSIReader.open(img_path)

