import argparse
import itertools
import numpy as np
from sklearn.neighbors import KDTree

from h2t.extract.features.graph import KNNFeatures
from h2t.misc.utils import (
    load_yaml,
    log_info,
    mkdir,
    dispatch_processing,
    recur_find_ext,
    rm_n_mkdir,
    rmdir,
)
from h2t.extract.utils import load_sample_with_info, normalize_positions
from h2t.data.utils import retrieve_dataset_slide_info


class Selector:
    def furthest_to_patterns(self, distances, topk: int):
        """Return topk indices of items having furthest distance.

        Args:
            distances: An array of shape `(N, P)` where `N` is the
                the number of sample and `P` is the number of the
                patterns.

        Return:
            np.ndarray: An array of selected indices within `distances`

        """
        # may not be in sorted order, k-smallest value
        topk = min(topk, distances.shape[0] - 1)
        sel = np.argpartition(-distances, topk)[:topk]
        return sel

    def closest_to_patterns(self, distances, topk: int):
        """Return topk indices of items having smallest distance.

        Args:
            distances: An array of shape `(N, P)` where `N` is the
                the number of sample and `P` is the number of the
                patterns.

        Return:
            np.ndarray: An array of selected indices within `distances`

        """
        # may not be in sorted order, k-smallest value
        topk = min(topk, distances.shape[0] - 1)
        sel = np.argpartition(distances, topk)[:topk]
        return sel

    def outside_distance_to_pattern(self, distances, threshold: float):
        """Return indices of items having distance larger than threshold.

        Args:
            distances: An array of shape `(N, P)` where `N` is the
                the number of sample and `P` is the number of the
                patterns.

        Return:
            np.ndarray: An array of selected indices outside `distances`

        """
        sel = distances > threshold
        return np.nonzero(sel)[0]

    def within_distance_to_pattern(self, distances, threshold: float):
        """Return indices of items having distance smaller than threshold.

        Args:
            distances: An array of shape `(N, P)` where `N` is the
                the number of sample and `P` is the number of the
                patterns.

        Return:
            np.ndarray: An array of selected indices within `distances`

        """
        sel = distances > threshold
        return np.nonzero(sel)[0]

    def run(self, distances, mode):
        if "fk" in mode:
            opt = int(mode.replace("fk", ""))
            return self.furthest_to_patterns(distances, opt)
        elif "k" in mode:
            opt = int(mode.replace("k", ""))
            return self.closest_to_patterns(distances, opt)
        elif "it" in mode:
            opt = float(mode.replace("it", ""))
            return self.within_distance_to_pattern(distances, opt)
        elif "ot" in mode:
            opt = float(mode.replace("ot", ""))
            return self.outside_distance_to_pattern(distances, opt)
        elif "n" == mode:
            return np.arange(distances.shape[0])
        else:
            assert False


class Combinator:
    def run(self, features, distances, mode):
        distances = distances[:, None]
        if "wn" in mode:
            combined = np.mean(features * distances, axis=0)
        elif "w" in mode:
            combined = np.mean(features * (1.0 - distances), axis=0)
        elif "m" in mode:
            combined = np.mean(features, axis=0)
        else:
            assert False
        return combined


class WSIProjector(object):
    def __init__(
        self, feature_dir=None, cluster_dir=None, selection_dir=None, num_patterns=None
    ):
        self.feature_dir = feature_dir
        self.cluster_dir = cluster_dir
        self.selection_dir = selection_dir
        self.num_patterns = num_patterns

    def pattern_histogram(self, labels):
        """Return the counting of each unique types.

        Args:
            labels (np.ndarray): Array of shape `(num_patches,)`
                where each value is the assigned pattern id in integer
                to the patch at the same index.

        Returns:
            np.ndarray: Histogram of unique values in `labels`
                with respect to a set of unique value in range
                `[0, self.num_patterns]`.

        """
        nn_type_list = labels
        (unique_type_list, nn_type_frequency,) = np.unique(
            nn_type_list, return_counts=True
        )
        # repopulate for information w.r.t provided type because the
        # subject may not contain all possible types within the
        # image/subject/dataset
        unique_type_freqency = np.zeros(self.num_patterns)
        unique_type_freqency[unique_type_list] = nn_type_frequency
        xfeat = unique_type_freqency / np.sum(unique_type_freqency)
        return xfeat

    def pattern_colocalization(self, labels, bounds):
        """Return summarized statistics about the pairwise
        co-occurence of patterns.

        Args:
            bounds (np.ndarray): Array of shape `(num_patches, bounds)`
                where bounds are `(top_left_x, top_left_y, bot_right_x, bot_right_y)`.
            labels (np.ndarray): Array of shape `(num_patches,)`
                where each value is the assigned pattern id in integer
                to the patch at the same index.

        """
        positions = normalize_positions(bounds)[:, :2]

        pattern_uids = np.arange(0, self.num_patterns)
        pattern_pairs = list(itertools.product(pattern_uids, pattern_uids))

        # immediate neighbor (3x3)
        kdtree = KDTree(positions)
        fxtor = KNNFeatures(
            kdtree=kdtree, pair_list=pattern_pairs, unique_type_list=pattern_uids
        )
        xfeat = fxtor.transform(positions, labels, radius=2)
        return xfeat

    def montage_patterns(self, labels, bounds):
        """Assemble labels in 1D form back to the 2D form.
        
        Return:
            np.ndarray: 2D image of `labels`, value `0` denote
                locations that do not exist in `bounds`. Original
                values in `labels` are shifted by 1.

        """
        assert np.min(labels) >= 0

        positions = normalize_positions(bounds)[:, :2]
        w, h = np.max(positions, axis=0)
        canvas = np.full((h + 1, w + 1), -1, dtype=np.int32)
        # -1 to denote background and to prevent overwriting exisitng
        # location with label 0 in `labels`
        canvas[positions[:, 1], positions[:, 0]] = labels
        canvas += 1  # shift up so that background is zero
        return canvas

    def deep_feature_projection(self, labels, features, distances, options):
        """
        Args:
            distances (np.ndarray): Array of shape `(num_patches, num_patterns)`
                where each row is the distance of the sample at the same index
                to all patterns.
            features (np.ndarray): Array of shape `(num_patches, num_features)`
                where each row is the distance of the sample at the same index
                to all patterns.

        """
        selection_args, combination_args = options

        combinator = Combinator()
        selector = Selector()

        assert len(features.shape) == 2
        num_instances, num_features = features.shape

        type_combined_feats = []
        for pattern_uid in range(self.num_patterns):
            # select out patches assigned to pattern
            sel = labels == pattern_uid
            distance_p = distances[sel][..., pattern_uid]
            features_p = features[sel]

            sel = selector.run(distance_p, selection_args)

            features_ = np.zeros([1, num_features])
            distances_ = np.zeros([1])
            if len(sel) > 0:
                features_ = features_p[sel]
                distances_ = distance_p[sel]
            features_ = np.array(features_)
            distances_ = np.array(distances_)

            features_ = combinator.run(features_, distances_, combination_args)
            type_combined_feats.append(features_)
        type_combined_feats = np.stack(type_combined_feats, axis=0)
        return type_combined_feats

    def _load_sample(self, sample_info):
        ds_code, wsi_code = sample_info
        features, positions = load_sample_with_info(
            self.feature_dir, sample_info, load_positions=True
        )

        labels = np.load(f"{self.cluster_dir}/{ds_code}/{wsi_code}.label.npy")
        distances = np.load(f"{self.cluster_dir}/{ds_code}/{wsi_code}.dist.npy")

        statistics = [positions, features, labels, distances]
        if self.selection_dir:
            selections = np.load(f"{self.cluster_dir}/{ds_code}/{wsi_code}.npy")
            selections = selections > 0
            statistics = [v[selections] for v in statistics]

        return statistics

    def __call__(self, sample_info, projection_mode):
        """
        Args:
            projection_mode (str): A string to denote which projection approach
                to be used. The string is also coded as a way to provide
                argument to the projection method.

                - `H`: Histogram of pattern assigned to patches

                - `C`: Pairwise colocalization of pattern assigned to patches

                - `dH-{SELECTION}-{COMBINATION}`: Combination of features of patches
                    that are assigned to each pattern. `{SELECTION}` and `{COMBINATION}`
                    are string of the form `{METHOD}{ARGUMENT}`.

                    - For `{SELECTION}`, `{METHOD}` includes:
                            - "n": No selection, return all.
                            - "t": For each pattern, within their assigned patches, select
                            patches that have their distances larger than a threshold.
                            - "k": For each pattern, within their assigned patches, select
                            patches that are the closest to it.
                            - "fk": For each pattern, within their assigned patches, select
                            patches that are the furthest away from it.

                    - For `{COMBINATION}`, `{METHOD}` includes:
                        of the selection method. It is in the form "{METHOD}{ARGUMENT}".
                        Currently, `METHOD` includes:
                            - "m": Averaging features of patches assigned to and selected for
                            (by `{SELECTION}`) each pattern.
                            - "w": Weighted sum features of patches assigned to and selected for
                            (by `{SELECTION}`) each pattern. Weigths are the distances to the
                            corresponding pattern.

        """

        bounds, features, labels, distances = self._load_sample(sample_info)
        assert self.num_patterns > np.max(labels), np.unique(labels)

        processing_codes = projection_mode.split("-")
        feature_type = processing_codes[0]
        options = processing_codes[1:]

        if feature_type == "H":
            xfeat = self.pattern_histogram(labels)
        elif feature_type == "C":
            xfeat = self.pattern_colocalization(labels, bounds)
        elif feature_type == "dH":
            xfeat = self.deep_feature_projection(labels, features, distances, options)
        elif feature_type == "dC":
            xfeat = self.montage_patterns(labels, bounds)
        else:
            assert False

        xfeat = xfeat.astype(np.float32)
        return xfeat


def process_once(
    sample_info,
    feature_dir,
    cluster_dir,
    selection_dir,
    save_dir,
    projection_mode,
    num_patterns,
):
    ds_code, wsi_code = sample_info
    out_path = f"{save_dir}/{ds_code}/{wsi_code}.npy"
    projector = WSIProjector(feature_dir, cluster_dir, selection_dir, num_patterns)
    features = projector(sample_info, projection_mode)
    np.save(out_path, features)


def functional_test():
    temp_dir = "/mnt/storage_0/workspace/h2t/h2t/experiments/debug/"
    sample_info = [".", "sample"]

    num_patterns = 8
    x_num_patches = 8
    y_num_patches = 16
    num_patches = x_num_patches * y_num_patches

    np.random.seed(5)
    features = np.random.rand(num_patches, 16)

    positions = np.nonzero(np.ones([y_num_patches, x_num_patches]))
    positions = np.transpose(np.array(positions), [1, 0])
    positions = np.concatenate([positions, positions + 1], axis=-1) * 256

    distances = np.random.rand(num_patches, num_patterns)
    labes = np.random.randint(0, num_patterns, num_patches)

    np.save(f"{temp_dir}/sample.features.npy", features)
    np.save(f"{temp_dir}/sample.position.npy", positions)
    np.save(f"{temp_dir}/sample.label.npy", labes)
    np.save(f"{temp_dir}/sample.dist.npy", distances)

    projection_modes = [
        "H",
        "C",
        "dC",
        "dH-n-m",
        "dH-n-w",
        "dH-k8-m",
        "dH-fk8-m",
        "dH-t0.2-m",
    ]

    selection_dir = None
    feature_dir = cluster_dir = save_dir = f"{temp_dir}/"

    for projection_mode in projection_modes:
        print(projection_mode)
        process_once(
            sample_info,
            feature_dir,
            cluster_dir,
            selection_dir,
            save_dir,
            projection_mode,
            num_patterns,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--FEATURE_CODE", type=str)
    parser.add_argument("--METHOD_CODE", type=str)
    parser.add_argument("--SOURCE_DATASET", type=str)
    parser.add_argument("--TARGET_DATASET", type=str)
    parser.add_argument("--WSI_PROJECTION_CODE", type=str, default="dH-n-w")
    args = parser.parse_args()

    NUM_WORKERS = 0
    SOURCE_DATASET = args.SOURCE_DATASET
    TARGET_DATASET = args.TARGET_DATASET
    FEATURE_CODE = args.FEATURE_CODE
    METHOD_CODE = args.METHOD_CODE
    WSI_PROJECTION_CODE = args.WSI_PROJECTION_CODE

    # * ---
    # PWD = "/mnt/storage_0/workspace/h2t/h2t/"
    # TARGET_DATASET = "tcga/lung/ffpe/lscc"
    # FEATURE_ROOT_DIR = f"{PWD}/experiments/local/features/[SWAV]-[mpp=0.50]-[512-256]/"
    # CLUSTER_ROOT_DIR = (
    #     "/mnt/storage_0/workspace/h2t/h2t/experiments/"
    #     "debug/cluster/sample/tcga-lung-luad-lusc/[SWAV]-[mpp=0.50]-[512-256]/"
    # )
    # WSI_PROJECTION_CODE = "C"
    # SELECTION_DIR = None

    # # * ---

    # SELECTION_DIR = None
    # PWD = "/root/local_storage/storage_0/workspace/h2t/h2t/"
    # FEATURE_ROOT_DIR = f"/root/dgx_workspace/h2t/features/{FEATURE_CODE}/"
    # CLUSTER_ROOT_DIR = (
    #     # f"{PWD}/experiments/debug/cluster/"
    #     f"/root/lsf_workspace/projects/atlas/media-v1/clustering/"
    #     f"{METHOD_CODE}/{SOURCE_DATASET}/{FEATURE_CODE}/"
    # )
    # SAVE_DIR = f"{CLUSTER_ROOT_DIR}/features/{WSI_PROJECTION_CODE}/"

    # * --- DEBUG LSF

    FEATURE_CODE = "[SWAV]-[mpp=0.50]-[512-256]"
    METHOD_CODE = "spherical-kmean-32"
    SOURCE_DATASET = "tcga-kidney-ccrcc-prcc-chrcc"
    TARGET_DATASET = "tcga/kidney/ffpe"
    # WSI_PROJECTION_CODE = "dH-it0.2-w"
    WSI_PROJECTION_CODE = "dH-n-w"

    # * ---

    SELECTION_DIR = None
    PWD = "/root/local_storage/storage_0/workspace/h2t/h2t/"
    FEATURE_ROOT_DIR = f"/root/dgx_workspace/h2t/features/{FEATURE_CODE}/"
    CLUSTER_ROOT_DIR = (
        # f"{PWD}/experiments/debug/cluster/"
        f"/root/lsf_workspace/projects/atlas/media-v1/clustering/"
        f"{METHOD_CODE}/{SOURCE_DATASET}/{FEATURE_CODE}/"
    )
    SAVE_DIR = f"{CLUSTER_ROOT_DIR}/features/{WSI_PROJECTION_CODE}/"

    # * ---
    dataset_identifiers = [
        # "tcga/lung/ffpe/lscc",
        # "tcga/lung/frozen/lscc",
        # "tcga/lung/ffpe/luad",
        # "tcga/lung/frozen/luad",
        # "cptac/lung/luad",
        # "cptac/lung/lscc",
        "tcga/breast/ffpe",
        "tcga/breast/frozen",
        "tcga/kidney/ffpe",
        "tcga/kidney/frozen",
    ]
    CLINICAL_ROOT_DIR = f"{PWD}/data/clinical/"
    dataset_sample_info = retrieve_dataset_slide_info(
        CLINICAL_ROOT_DIR, FEATURE_ROOT_DIR, dataset_identifiers
    )
    sample_info_list = dataset_sample_info[TARGET_DATASET]
    sample_info_list = [v[0] for v in sample_info_list]

    # premade all directories to prevent possible collisions
    ds_codes, _ = list(zip(*sample_info_list))
    ds_codes = np.unique(ds_codes)

    for ds_code in ds_codes:
        rm_n_mkdir(f"{SAVE_DIR}/{ds_code}/")

    # * ---

    cluster_config = load_yaml(f"{CLUSTER_ROOT_DIR}/config.yaml")
    num_patterns = cluster_config["num_patterns"]

    run_list = [
        [
            process_once,
            sample_info,
            FEATURE_ROOT_DIR,
            f"{CLUSTER_ROOT_DIR}/transformed/",
            SELECTION_DIR,
            SAVE_DIR,
            WSI_PROJECTION_CODE,
            num_patterns,
        ]
        for sample_info in sample_info_list
    ]
    dispatch_processing(run_list, num_workers=NUM_WORKERS, crash_on_exception=True)
