# H2T

# Setup Working Directory
As this project involves multiple pipelines where each pipeline can be
an entire new repository or algorithm (tissue segmentation for image patches,
handcrafted/deep feature extraction for WSI, etc.). A good way to structure
their dependency (miscellaneous reading etc.) is a big monolithic folder
with inter-dependency import. For that to work, apply the following
to make the repository/project folder into an editable package

```
pip install -e .
```

# Instructions
1. Extract tissue area by navigating to `segment/inference`
    - Run the `infer.py` for generating prediction
    - Run the `postproc.py` to turn the prediction into the tissue masks
2. Extracting deep features by navigating to `extract`
    - Run the `extract_deep_features.py`
    - In case you want to use different tissue masks but do not want to perform
    the extraction again, run `generate_selection.py` to generate mask arrays
    which patches within the already extracted patches to be used.
3. Extracting the prototypical patterns by navigating to `extract`
