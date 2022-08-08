
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


def normalize_positions(positions, step_shape=None, dtype=np.int32):
    """Quantize/Normalize the positions with respect to the step size.

    Args:
        positions (np.ndarray): Array of shape `(num_patches, bounds)`
            where bounds contains `(top_left_x, top_left_y, bot_right_x, bot_right_y)`

    Example: Consider a list of positions `[[256, 256], [512, 512], [1024, 1024]]`
        in `(x, y)` coordinates, in term of step of shape `(H, W) = (256, 256)`,
        the above list is first converted to `[[1, 1], [2, 2], [4, 4]]`. Afterward,
        they are shifted to the top-left normalized corner and become
        `[[0, 0], [1, 1], [3, 3]]`.

    """
    assert len(positions.shape) == 2
    assert positions.shape[1] == 4

    top_left = np.min(positions[:, :2], axis=0, keepdims=True)
    positions_ = positions.copy()
    positions_[:, :2] = (positions_[:, :2] - top_left[None])
    positions_[:, 2:] = (positions_[:, 2:] - top_left[None])
    if step_shape is None:
        patch_sizes = positions_[:, 2:] - positions_[:, :2]
        patch_sizes = np.unique(patch_sizes, axis=0)
        assert len(patch_sizes) == 1
        assert patch_sizes[0][0] == patch_sizes[0][1]
        step_shape = patch_sizes[0][0]
    positions_ = (positions_ / step_shape).astype(dtype)
    return positions_
