
import tarfile
from io import BytesIO
import numpy as np


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


def load_npy_tar(tar, path_in_tar):
    bytesBuffer = BytesIO()
    bytesBuffer.write(tar.extractfile(path_in_tar).read())
    bytesBuffer.seek(0)
    return np.load(bytesBuffer, allow_pickle=False)


def load_sample_with_info(
        root_dir,
        sample_info,
        load_positions=True,
        selection_dir=None
    ):
    ds_code, wsi_code = sample_info

    feature_path = f"{root_dir}/{ds_code}/{wsi_code}.features.npy"
    position_path = f"{root_dir}/{ds_code}/{wsi_code}.position.npy"
    patch_features = np.load(feature_path, mmap_mode="r")

    if load_positions:
        patch_positions = np.load(position_path, mmap_mode="r")

    if selection_dir is not None:
        selection_path = f"{selection_dir}/{ds_code}/{wsi_code}.npy"
        selections = np.load(selection_path) > 0
        patch_features = patch_features[selections]

        if load_positions:
            patch_positions = patch_positions[selections]

    patch_features = np.squeeze(patch_features)
    assert len(patch_features.shape) == 2, "Only 1 patch exists in WSI!"
    if load_positions:
        return patch_features, patch_positions
    return patch_features
