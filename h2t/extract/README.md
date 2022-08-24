
# Feature Extractions


This folder contains the code for extracting deep features as well as
the H2T prototypical patterns and the WSI projections
in the [paper](https://arxiv.org/abs/2202.07001).

- `config.yaml` contains the definition of the `SOURCE_DATASET` to be used for subsequent H2T opereations. Name of each `SOURCE_DATASET` is the keyword within the file.
    - Each identifier within `SOURCE_DATASET.identifiers` in the file corresponds to the `<DATASET-CODE>` defined in [main readme](../README.md#experimental-api).
    - `SOURCE_DATASET.labels`: select WSIs that are of any of these tissue types from all above identifiers for constructing this `SOURCE_DATASET`.

- `extract_patterns.py`:
    - `METHOD_CODE`: Same as `<CLUSTERING-METHOD-CODE>` described in [main readme](../README.md#experimental-api).
    - `SOURCE_DATASET`: Same as `<SOURCE_DATASET>` described in [main readme](../README.md#experimental-api).
    - `FEATURE_CODE`: Same as `<FEATURE-CODE>` described in [main readme](../README.md#experimental-api).

    - `NUM_EPOCHS`: Number of epochs for mining the patterns.
    - `SCALER`: Default to `False`. Standardizing 
    each patch feature using the mean and standard **across the entire** `SOURCE_DATASET` or not.
    - `NUM_CLUSTERS`: Number of protypical patterns to mine for.

- `extract_pattern_projection.py`:
    - `METHOD_CODE`: Same as `<CLUSTERING-METHOD-CODE>` described in [main readme](../README.md#experimental-api).
    - `SOURCE_DATASET`: Same as `<SOURCE_DATASET>` described in [main readme](../README.md#experimental-api).
    - `TARGET_DATASET`: Same as `<TARGET_DATASET>` described in [main readme](../README.md#experimental-api).
    - `FEATURE_CODE`: Same as `<FEATURE_CODE>` described in [main readme](../README.md#experimental-api).

- `extract_pattern_projection.py`:
    - `METHOD_CODE`: Same as `<CLUSTERING-METHOD-CODE>` described in [main readme](../README.md#experimental-api).
    - `SOURCE_DATASET`: Same as `<SOURCE_DATASET>` described in [main readme](../README.md#experimental-api).
    - `TARGET_DATASET`: Same as `<TARGET_DATASET>` described in [main readme](../README.md#experimental-api).
    - `FEATURE_CODE`: Same as `<FEATURE_CODE>` described in [main readme](../README.md#experimental-api).

    - `WSI_PROJECTION_CODE`: Name of the projection, refer to the document of
    `projection_mode` [here](./extract_wsi_projection.py#L257) for details.
    The codes in the program are mapped to those in the paper as follows
        ```python
        code2paper = {
         `dH-n-w`  : `dH-w`,
         `dH-k*-m` : `dH-fk*`,
         `dH-fk*-m`: `dH-fk*`,
         `dH-ot*-m`: `dH-t*`,
        }
        ```

