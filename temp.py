
# %%
import joblib

# %%

# a = joblib.load("/mnt/storage_0/workspace/h2t/h2t/data/splits/[ccrcc-prcc-chrcc]_train=tcga.dat")
# b = a[0]["train"]
# print(len(a[0]["train"]))
# print(len(a[0]["valid"]))
# print(len([v for v in b if "ffpe" in v[0][0]]))
# %%

# import numpy as np
# a = np.load("/mnt/storage_0/workspace/h2t/h2t/experiments/remote/media-v1/clustering/spherical-kmean-8/tcga-breast-idc-lob/[SWAV]-[mpp=0.50]-[512-256]/features/C/tcga/breast/ffpe/TCGA-3C-AALI-01Z-00-DX1.F6E9A5DF-D8FB-45CF-B4BD-C6B76294C291.npy")
# print(a.shape)
# a = np.load("/mnt/storage_0/workspace/h2t/h2t/experiments/remote/media-v1/clustering/spherical-kmean-16/tcga-breast-idc-lob/[SWAV]-[mpp=0.50]-[512-256]/features/C/tcga/breast/ffpe/TCGA-3C-AALI-01Z-00-DX1.F6E9A5DF-D8FB-45CF-B4BD-C6B76294C291.npy")
# print(a.shape)
# a = np.load("/mnt/storage_0/workspace/h2t/h2t/experiments/remote/media-v1/clustering/spherical-kmean-32/tcga-breast-idc-lob/[SWAV]-[mpp=0.50]-[512-256]/features/C/tcga/breast/ffpe/TCGA-3C-AALI-01Z-00-DX1.F6E9A5DF-D8FB-45CF-B4BD-C6B76294C291.npy")
# print(a.shape)

# PWD = "/mnt/storage_0/workspace/h2t/h2t/data/splits/"
# split_code = "[luad-lusc]_train=cptac_test=tcga"
# old_splits = joblib.load(f"{PWD}/{split_code}.dat")
# old_splits = [v[1] for v in old_splits]

# new_splits = []
# for split in old_splits:
#     split_ = {
#         "train": split["cptac-train"],
#         "valid": split["cptac-valid"],
#         "test": split["tcga-d"] + split["tcga-f"]
#     }
#     new_splits.append(split_)
# joblib.dump(new_splits, f"{split_code}.dat")

# PWD = "/mnt/storage_0/workspace/h2t/h2t/data/splits/"
# split_code = "[normal-tumor]_train=tcga_test=cptac"
# old_splits = joblib.load(f"{PWD}/{split_code}.dat")
# old_splits = [v[1] for v in old_splits]

# new_splits = []
# for split in old_splits:
#     split_ = {
#         "train": split["tcga-train"],
#         "valid": split["tcga-valid"],
#         "test": split["cptac"]
#     }
#     new_splits.append(split_)
# joblib.dump(new_splits, f"{split_code}.dat")
# print("here")

# import numpy as np
# PWD = '/mnt/storage_0/workspace/h2t/h2t/data/splits/'
# SPLIT = f"{PWD}/[ccrcc-prcc-chrcc]_train=tcga.dat"
# split = joblib.load(SPLIT)[0]
# wsi_codes = [[v[0][1], v[1]] for v in split['train'] + split['valid']]
# wsi_codes = [['-'.join(v.split('-')[:3]), u] for v, u in wsi_codes]
# wsi_codes, labels = list(zip(*wsi_codes))
# wsi_codes = np.array(wsi_codes)
# labels = np.array(labels)
# for label_id in np.unique(labels):
#     print(len(np.unique(wsi_codes[labels==label_id])))

import pkg_resources
installed_packages = pkg_resources.working_set
installed_packages_list = sorted(["%s==%s" % (i.key, i.version)
     for i in installed_packages])
print(*installed_packages_list, sep="\n")