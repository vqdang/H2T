# Dataset Definition & Generation

This folder contain the clinical information of all dataset utilized in the paper. It also contains the code used for constructing all sub datasets (such as data splitting) within the paper.

For convenience, all data splits utilized in the [paper](https://arxiv.org/abs/2202.07001) are provided in `data/splits`.

- `config.yaml` contains the definition of each splitting used for downstream analysis. Each keyword within the file corrspond to `<DATA-SPLIT-CODE>` which is described in [main readme](../../README.md#experimental-api).
    - Each identifier within `<DATA-SPLIT-CODE>.identifiers` in the file corresponds to the `<DATASET-CODE>` defined in [main readme](../../README.md#experimental-api).
    - `<DATA-SPLIT-CODE>.labels`: select WSIs that are of any of these tissue types from all above identifiers for constructing this `SOURCE_DATASET`.

- `generate_split.py`: entry file to generate data split defined within `config.yaml` based on WSIs that have  their features extracted and associated clinical information, which is provided within `data/clinical`.
