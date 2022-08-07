# %%

import pathlib

import numpy as np
from sklearn.model_selection import StratifiedKFold

import h2t.data.dataset as ds
from h2t.misc.utils import flatten_list, recur_find_ext


def get_cv_split(Y, num_splits=5, random_state=None, shuffle=True):
    Y = np.array(Y)
    X = np.arange(0, Y.shape[0])  # dummy

    skf = StratifiedKFold(
        n_splits=num_splits, random_state=random_state, shuffle=shuffle
    )
    full_split = []
    for ext_idx, (train_idx, valid_idx) in enumerate(skf.split(X, Y)):
        full_split.append([(ext_idx), [train_idx, valid_idx]])
    return full_split


def retrieve_dataset_slide_info(
        clinical_root_dir,
        feature_root_dir,
        dataset_identifiers
    ):

    sample_info_per_dataset = {}
    for identifier in dataset_identifiers:
        slide_names = recur_find_ext(
            f"{feature_root_dir}/{identifier}/", [".features.npy"]
        )
        slide_names = [pathlib.Path(v).stem for v in slide_names]
        slide_names = [str(v).replace(".features", "") for v in slide_names]
        assert len(slide_names) > 0, f"{feature_root_dir}/{identifier}/"
        print(f"#Feature Files in {identifier}: {len(slide_names)}")

        if "tcga" in identifier:
            organ_identifier = identifier.split("/")[1]
            slide_info_list = ds.TCGA.retrieve_labels(
                slide_names,
                f"{clinical_root_dir}/tcga-{organ_identifier}/clinical.tsv",
                f"{clinical_root_dir}/tcga-{organ_identifier}/slides.json",
            )
            slide_info_list = [[[identifier, v[0]], v[1]] for v in slide_info_list]
        elif "cptac" in identifier:
            organ_identifier = [v for v in identifier.split("/") if len(v) > 0]
            organ_identifier = "-".join(organ_identifier)
            slide_info_list = ds.CPTAC.retrieve_labels(
                slide_names,
                f"{clinical_root_dir}/{organ_identifier}.json",
            )
        else:
            assert False
        sample_info_per_dataset[identifier] = slide_info_list
    return sample_info_per_dataset


def retrieve_subset(subset_info, dataset_sample_info):
    label_mapping = subset_info["labels"]
    label_mapping = {k.lower(): int(v) for k, v in label_mapping.items()}

    samples = []
    for identifier in subset_info["identifiers"]:
        identifier = [v for v in identifier.split("/") if len(v) > 0]
        identifier = "/".join(identifier)
        identifer_samples = dataset_sample_info[identifier]
        samples.extend(identifer_samples)
    
    #
    compositions = np.unique([v[1] for v in samples], return_counts=True)
    print(identifier)
    print(*list(zip(*compositions)), sep="\n")
    print("----")
    #

    # filter samples with labels that are not within selected
    # set out (contained as keys within `label_mapping`)
    samples = [v for v in samples if v[1] in label_mapping]

    sample_codes, sample_labels = list(zip(*samples))
    sample_labels = [label_mapping[v] for v in sample_labels]
    return [sample_codes, sample_labels]


def per_dataset_stratified_split(
    config: dict, dataset_sample_info: dict, num_splits: int = 5
):
    """Groupping and splitting dataset stratifically at label and dataset levels.

    Args:
        config (dict): A dictionary contains information how each dataset within
            each split is generated (called `split dataset`). A `split dataset`
            can be a composition of multiple dataset defined within `dataset_sample_info`,
            For more details, check the `config.yaml`.

        dataset_sample_info (dict): A dictionary contains the information about
            the samples within each dataset which has the format

            ```
            sample: [sample_identifer: List[str], biological_label: str]
            dataset_sample_info: {dataset_identifier: str, List[sample]}
            ```

            Here, `sample_identifier` contains the slide name and associated
            directory structures. `dataset_identifier` can be used to
            define composite `split dataset` in `config`.

    """

    # stratified split for each sub dataset first
    # then merge the split per each dataset to create the cv splits

    subset_info = config["train-valid"]
    splits = [{"train": [], "valid": []} for _ in range(num_splits)]

    subset_info = [retrieve_subset(v, dataset_sample_info) for v in subset_info]
    for sample_codes, sample_labels in subset_info:
        subset_splits = get_cv_split(
            sample_labels, num_splits=num_splits, random_state=5, shuffle=True
        )

        samples = np.array(list(zip(sample_codes, sample_labels)))
        for split_idx, (train_idx, valid_idx) in subset_splits:
            splits[split_idx]["train"].extend(samples[train_idx])
            splits[split_idx]["valid"].extend(samples[valid_idx])

    # stratified split for each sub dataset first
    # then merge the split per each dataset to create the cv splits
    if "test" in config:
        subset_info = config["test"]
        subset_info = retrieve_subset(subset_info, dataset_sample_info)
        subset_info = [
            list(zip(sample_codes, sample_labels))
            for sample_codes, sample_labels in subset_info
        ]
        subset_info = flatten_list(subset_info)
        for subsplit in splits:
            subsplit["tests"] = subset_info
    return splits
