import pathlib

import cv2
import imgaug as ia
import numpy as np
import torch.utils.data
from imgaug import augmenters as iaa
from torch.nn.utils.rnn import pad_sequence

from augs import (add_to_brightness, add_to_contrast, add_to_hue,
                  add_to_saturation, gaussian_blur, median_blur)

from misc.wsi_handler import get_file_handler


####
class PatchDataset(torch.utils.data.Dataset):
    def __init__(self,
        info_dict,
        run_mode='train',
        subset_name='train',  
        setup_augmentor=True):

        self.run_mode = run_mode
        self.subset_name = subset_name
        if subset_name in info_dict:
            self.info_list = info_dict[subset_name]
        else:
            assert False, 'Unknown subset `%s`' % subset_name

        self.id = 0
        if setup_augmentor:
            self.setup_augmentor(0, 0)

        self.loader_collate_fn = None

        return


    def setup_augmentor(self, worker_id, seed):
        self.augmentor = self.__get_augmentation(self.run_mode, seed)
        self.shape_augs = iaa.Sequential(self.augmentor[0])
        self.input_augs = iaa.Sequential(self.augmentor[1])
        self.id = self.id + worker_id
        return

    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, idx):

        img_path, msk_path = self.info_list[idx]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(msk_path)[..., 0]

        if self.shape_augs is not None:
            shape_augs = self.shape_augs.to_deterministic()
            img = shape_augs.augment_image(img)
            msk = shape_augs.augment_image(msk)

        if self.input_augs is not None:
            input_augs = self.input_augs.to_deterministic()
            img = input_augs.augment_image(img)

        msk = cv2.resize(
            msk, (0, 0), fx=0.5, fy=0.5,
            interpolation=cv2.INTER_NEAREST)

        msk = np.array(msk == 255).astype(np.uint8)
        return img, msk

    def __get_augmentation(self, mode, rng):
        if mode == "train":
            shape_augs = [
                iaa.Sometimes(0.5,
                    iaa.Affine(
                        # scale images to 80-120% of their size, individually per axis
                        # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        # translate by -A to +A percent (per axis)
                        # translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
                        # shear=(-5, 5),  # shear by -5 to +5 degrees
                        rotate=(-179, 179),  # rotate by -179 to +179 degrees
                        order=0,  # use nearest neighbour
                        backend="cv2",  # opencv for fast processing
                        # mode="reflect",  # padding type at border
                        cval=0, mode='constant', # zero pad it
                        seed=rng,
                    )
                ),
                # set position to 'center' for center crop
                # else 'uniform' for random crop
                iaa.Fliplr(0.5, seed=rng),
                iaa.Flipud(0.5, seed=rng),
                iaa.CropToFixedSize(1024, 1024, position='center'),
            ]

            input_augs = [
                iaa.OneOf([
                        iaa.Lambda(seed=rng, func_images=lambda *args: gaussian_blur(*args, max_ksize=3)),
                        iaa.Lambda(seed=rng, func_images=lambda *args: median_blur(*args, max_ksize=3)),
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                    ]
                ),
                iaa.OneOf([
                    iaa.Sometimes(0.50, iaa.Grayscale()),
                    # apply color augmentation 90% of time
                    iaa.Sometimes(0.90, 
                        iaa.Sequential([
                                iaa.Lambda(seed=rng,func_images=lambda *args: add_to_hue(*args, range=(-8, 8))),
                                iaa.Lambda(seed=rng,func_images=lambda *args: add_to_saturation(*args, range=(-0.2, 0.2))),
                                iaa.Lambda(seed=rng,func_images=lambda *args: add_to_brightness(*args, range=(-26, 26))),
                                iaa.Lambda(seed=rng,func_images=lambda *args: add_to_contrast(*args, range=(0.75, 1.25))),
                            ],
                            random_order=True,
                        ),
                    ),
                ]) 
            ]
        elif 'gry' in self.subset_name:
            shape_augs = [                
                iaa.CropToFixedSize(1024, 1024, position='center'), # random crop
            ]
            input_augs = [iaa.Grayscale()]
        elif 'rgb' in self.subset_name:
            shape_augs = [
                iaa.CropToFixedSize(1024, 1024, position='center'), # random crop
            ]
            input_augs = []
        return shape_augs, input_augs
####
