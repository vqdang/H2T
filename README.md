# Handcrafted Histological Transformer (H2T): Unsupervised Representation of Whole Slide Images

This repository contains the implementation of H2T: a framework to derive/extract Whole Slide
Image (WSI) representation in an unsupervised manner and neural network free based on
Transformer mechanisms. The framework builds the WSI representation from a combination of
a set of input patches and a set of prototypical patterns. These patterns can be easily obtained
either manually or automatedly via clustering. 

> **Note**: The paper is published at Medical Image Analysis
> and can be accessed at: [[arxiv]](https://arxiv.org/abs/2202.07001) [[MedIA]](https://www.sciencedirect.com/science/article/pii/S136184152300004X)]

## Setup Environment

Setup the python environment by doing the following
```
conda create -n h2t python=3.9 
pip install -r requirements.txt
```

> **Note**: You may need to install the ```pytorch``` separately to
make it in line with your local cuda version (or vice versa).

> **Note**: You may need to install `tiatoolbox` seperately because it 
> will over-write other packages. In this repository, we clone the 
> [tiatoolbox](https://github.com/TissueImageAnalytics/tiatoolbox)
> and expose it as environment variable.
>
> ```
> export TIATOOLBOX="path/to/tiatoolbox"
> ```
>

This framwork involves multiple pipelines where each pipeline usually can be a repository in and of itself (such as tissue segmentation for image patches, handcrafted/deep feature extraction for WSI, etc.). To make the code more organized, each of these pipeline has been structured in a self-contained directory where shared functions are refactored out. The entire repository is structured in a monolithic manner. To work with many inter-dependency import, we
turn the project into an **editable package** by the following command

```
pip install -e .
```

## Data Sharing

You can download intermediate results and some pretrained models utilized in the paper by following the instructions [here](https://warwick.ac.uk/fac/cross_fac/tia/data/h2t/).

We share the followings data:
- Deep features for TCGA-Lung, TCGA-Breast, TCGA-Kidney, CPTAC-Lung.
- Tissue masks for TCGA-Lung, TCGA-Breast, TCGA-Kidney, CPTAC-Lung.
- Pretrained models for feature extraction Supervised-ResNet50, SWAV-ResNet50.
- Prototype patterns of tumorous or normal tissue that is from either breast, lung or kidney WSIs within TCGA or CPTAC dataset.

> **Note**: All data shared by us is licensed under [![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa].

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

## Instructions
1. Extract tissue area by navigating to `segment/inference`
    - Run the `infer.py` for generating prediction
    - Run the `postproc.py` to turn the prediction into the tissue masks
2. Extracting deep features by navigating to `extract`
    - Run the `extract_deep_features.py`
    - In case you want to use different tissue masks but do not want to perform
    the extraction again, run `generate_selection.py` to generate mask arrays where each indicates which patches (within those already extracted) to be used.
3. Extracting the prototypical patterns by navigating to `extract`
    - Run the `extract_patterns.py` to obtain the prototypical patterns from a `<SOURCE-DATASET>`.
    - Run the `extract_pattern_projection.py` to project all patches within a `<TARGET-DATASET>` againts patterns from a `<SOURCE-DATASET>`.
    - Run the `extract_wsi_projection.py` to turn the above projection into H2T representation.
3. Generate the data split for downstream analysis
by navigating to `data`
    - Run the `generate_split.py` to generate training/validation/testing splits (stratified across labelled and dataset) based on a combination of datasets.
    > **Note**: We provide the training/validation/testing splits utilized for the paper within `data/splits`.
3. Perform downstream analyses such as linear probing by navigating to `downstream`
    - Run the `run.py`.

> **Note**: In case you only want to obtain the results of CLAM and Transformer baselines, you can skip `step 3`.

> **Note**: For each step, please read the instruction within each associated folder.

## Experimental API

Here, we describe the how the experiment output is structured.
```
PWD
|--requirements.txt
|
|  # please either create it or symlink it
|--experiments 
   |
   |--features
   |  |--<FEATURE-CODE>
   |      |--<DATASET-CODE>
   |
   |--segment
   |  |--<SEGMENTATION-METHOD-CODE>
   |      |--<DATASET-CODE>
   |
   |--clustering
   |  |--<CLUSTERING-METHOD-CODE>
   |      |--<SOURCE-DATASET>
   |          |--<FEATURE-CODE>
   |              |--features
   |              |  |--<WSI-PROJECTION-CODE>
   |              |      |--<TARGET-DATASET>
   |              |--transformed
   |                 |--<TARGET-DATASET>
   |
   |--downstream
   |  |--<DATA-SPLIT-CODE>
   |      |--<FEATURE-CODE>
   |          |--<SOURCE-DATASET>
   |              |--<DOWNSTREAM-METHOD-CODE>
```
All the codes can contain `/`, we describe them here:
- `<FEATURE-CODE>`: Name of the set of extracted feature (e.g. "[SWAV]-[mpp=0.50]-[512-256]").
- `<DATASET-CODE>`: Name of the dataset (e.g. 'tcga/breast/frozen', 'tcga/breast/ffpe').
- `<SOURCE-DATASET>`: Name of the source dataset
used when we mine for the prototypical patterns.
The names as well as the tissue types of and within these source datasets are defined in 
`extract/config.yaml`.
- `<DATA-SPLIT-CODE>`: Name for a list of data splits where each subset is generated from a combination of `<DATASET-CODE>` (such as cancerous slides in TCGA-LUAD and TCGA-LUSC to make TCGA-lung-tumor). Actual combination utilized in this study are described in `data/config.yaml`. Pre-generated data splits and their associated names (the file names) are provided in `data/splits`.
- `<TARGET-DATASET>`: Name of the dataset that are projected using the patterns obtained from `<SOURCE-DATASET>`. For this work, they are the same dataset used for `<DATASET-CODE>`.
- `<CLUSTERING-METHOD-CODE>:` Name of the clustering method. For examples, 'spherical-kmean-8'
- `<WSI-PROJECTION-CODE>`: Name of the H2T projection method.
- `<DOWNSTREAM-METHOD-CODE>`: Name of the method utilized for WSI classifcations. In case of H2T linear probing, it
is `<CLUSTERING-METHOD-CODE>/<WSI-PROJECTION-CODE>/<DOWNSTREAM-METHOD-CODE>`.

## Citation

If any part of this code is used, please give appropriate citation to our paper.

BibTex entry: 
```
@article{vu2022h2t,
  title={Handcrafted Histological Transformer (H2T): Unsupervised Representation of Whole Slide Images},
  author={Vu, Quoc Dang and Rajpoot, Kashif and Raza, Shan E Ahmed and Rajpoot, Nasir},
  journal={arXiv preprint arXiv:2202.07001},
  year={2022}
}
@article{vu2023h2t,
  title={Handcrafted Histological Transformer (H2T): Unsupervised representation of whole slide images},
  author={Vu, Quoc Dang and Rajpoot, Kashif and Raza, Shan E Ahmed and Rajpoot, Nasir},
  journal={Medical Image Analysis},
  pages={102743},
  year={2023},
  publisher={Elsevier}
}
```