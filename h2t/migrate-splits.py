
import os
import joblib

SRC_SPLIT_ROOT = "/mnt/storage_0/workspace/h2t/h2t/data/splits/backup/"
DST_SPLIT_ROOT = "/mnt/storage_0/workspace/h2t/h2t/data/splits/"

SPLIT_SETS = [
    # '[luad-lusc]_train=cptac_test=tcga.dat',
    # '[luad-lusc]_train=tcga_test=cptac.dat',
    # '[normal-luad-lusc]_train=cptac_test=tcga.dat',
    # '[normal-luad-lusc]_train=tcga_test=cptac.dat',
    '[normal-tumor]_train=cptac_test=tcga.dat',
    '[normal-tumor]_train=tcga_test=cptac.dat',
]

root_maps = {
    "CPTAC-LSCC": "cptac/lung/",
    "CPTAC-LUAD": "cptac/lung/",
    "TCGA-LSCC": "tcga/lung/ffpe/",
    "TCGA-LSCC-Frozen": "tcga/lung/frozen/",
    "TCGA-LUAD": "tcga/lung/ffpe/",
    "TCGA-LUAD-Frozen": "tcga/lung/frozen/",
}

FEATURE_ROOT = (
    "/run/user/1000/gvfs/sftp:host=tialab-romesco.dcs.warwick.ac.uk,user=tialab-dang/"
    "/home/tialab-dang/lsf_workspace/projects/atlas/"
    "/final/features/[SWAV]-[mpp=0.50]-[512-256]/"
)
for split_code in SPLIT_SETS:
    split_path = f"{SRC_SPLIT_ROOT}/{split_code}"
    splits = [v[1] for v in joblib.load(split_path)]

    new_splits = []
    for split in splits:
        new_split = {}
        for sub_split, data in split.items():
            new_data = []
            for idx, item in enumerate(data):
                label = item[-1]
                if len(item[0]) == 1:
                    root_header, wsi_code = item[0][0]
                else:
                    root_header, wsi_code = item[0]
                root_header = root_maps[root_header]
                file_path = f"{FEATURE_ROOT}/{root_header}/{wsi_code}.features.npy"
                if not os.path.exists(file_path):
                    assert False
                new_data.append([[root_header, wsi_code], label])
            new_split[sub_split] = new_data
        new_split_ = {}
        if "cptac-train" in new_split:
            new_split_["train"] = new_split["cptac-train"]
            new_split_["valid"] = new_split["cptac-valid"]
            new_split_["test"] = new_split["tcga-d"] + new_split["tcga-f"]
        else:
            new_split_["train"] = new_split["tcga-train"]
            new_split_["valid"] = new_split["tcga-valid"]
            new_split_["test"] = new_split["cptac"]
        new_splits.append(new_split_)
    joblib.dump(new_splits, f"{DST_SPLIT_ROOT}/{split_code}")
