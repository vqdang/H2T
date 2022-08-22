# Pipeline for Downstream Analysis

This folder contains the code for running the downstream analysis utilized
in the [paper](https://arxiv.org/abs/2202.07001).

Followings are the main components:

- `params`: folder contains the configurations to be used for training. Followings are the most important parameters
    - `metadata.architecture_name`: name of the architecture to be used,
        The name must be pre-defined in function `get_architecture` in `downstreams/recipes/opt.py`.

    - `metadata.option_name`: name of the training recipe pre-defined in
        `ABCConfig.config` in `downstreams/recipes/opt.py`

    - `model_kwargs`: arguments to instantiate the class of the model
    retrieved by `get_architecture` based on `metadata.architecture_name`.

- `recipes`: folder contains the actual code describing the training configurations, this includes
    - Training/Validation/Inference steps associated with a training recipe.
    - Metric calculations associated with a training recipe.
    - Loss functions associated with a training recipe.
    - Number of epochs.
    - How the often the validation is conducted.
    - How the model is saved.
    - etc.

- `run.py`: the entry file to actually start the running process.
    - `DATA_SPLIT_CODE`: Same as `<DATA-SPLIT-CODE>` described in [main readme](../README.md#experimental-api)
    - `FEATURE_CODE`: Same as `<FEATURE-CODE>` in [main readme](../README.md#experimental-api)
    - `CLUSTER_CODE`: Same as `<CLUSTERING-METHOD-CODE>` in [main readme](../README.md#experimental-api)
    - `SOURCE_DATASET`: Same as `<SOURCE_DATASET>` in [main readme](../README.md#experimental-api). If `ARCH_CODE` is not used for H2T projection (i.e. `probe`), then please set it to `None`.
    - `WSI_FEATURE_CODE`: A code name for a set of `<WSI-PROJECTION-CODE>` which is defined in [main readme](../README.md#experimental-api) to be used as H2T representation. `#` is a delimiter for each `<WSI-PROJECTION-CODE>`. For example, `dC-onehot#dH-n-w` denotes a H2T representation obtained by `dC-onehot` and `dH-n-w`.

    - `TRAINING_CONFIG_CODE`: Name of the training config file. The file must be defined within `downstream/params`.
    - `SPLIT_IDX`: The index of the data split within `DATA_SPLIT_CODE` to train on.
    - `ARCH_CODE`: The name of the model architecture defined within `get_architecture.py` within `downstreams/recipes/opt.py`. Following architectures are available
        - `clam`: Baseline CLAM model, check [clam.yml](./params/transformer-1.yml) and [CLAM definition](../models/mil/clam.py)for details.
        - `transformer-1`: Baseline Transformer model with no full self-attention, check [transformer-1.yml](./params/transformer-1.yml) and [Transformer definition](../models/mil/transformer.py)for details.
        - `transformer-2`: Baseline Transformer model with an additional full self-attention, check [transformer-2.yml](./params/transformer-1.yml) and [Transformer definition](../models/mil/transformer.py)for details.
        - `probe`: Linear probing for H2T representation. If `WSI_FEATURE_CODE` contains colocalization projection via pattern assignment maps (PAMs),  denoted by `dC`, `probe` will be expanded to `cnn-probe`. Here, a CNN is attached on ontop the PAMs before being merged with other H2T projections for linear probing. Otherwise, `probe` is `linear-probe`.

