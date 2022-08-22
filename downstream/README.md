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
    - `DATA_SPLIT_CODE`: Same as `<DATA-SPLIT-CODE>` in [main readme](../README.md#experimental-api)
    - `TRAINING_CONFIG_CODE`: 
    - `SPLIT_IDX`: 
    - `ARCH_CODE`: 
    - `FEATURE_CODE`: 
    - `CLUSTER_CODE`: 
    - `SOURCE_DATASET`: 
    - `WSI_FEATURE_CODE`: 

