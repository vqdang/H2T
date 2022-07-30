import numpy as np
import json
import pathlib
import re
import os
from collections import OrderedDict
import pandas as pd

from misc.utils import load_json, recur_find_ext


class TCGA(object):

    @staticmethod
    def retrieve_labels(slide_names, clinical_file, slide_metadata_file):
        slide_case_info_ = load_json(slide_metadata_file)
        slide_case_info = {}
        for slide_info in slide_case_info_:
            entity = slide_info["associated_entities"]
            assert len(entity) == 1
            entity = entity[0]
            slide_name = slide_info["file_name"]
            slide_name = slide_name.replace(".svs", "")
            case_submitter_id = entity["entity_submitter_id"]
            case_submitter_id = "-".join(case_submitter_id.split("-")[:3])
            slide_case_info[slide_name] = {
                "case_id": entity["case_id"],
                "case_submitter_id": case_submitter_id
            }

        clinical_info = pd.read_csv(clinical_file, sep="\t")
        clinical_info = clinical_info[["case_id", "case_submitter_id", "primary_diagnosis"]]
        # print(np.unique(clinical_info["primary_diagnosis"]))

        # a patient can be diagnosed with disease X (recorded within
        # `clinical_info`). However, a tissue slide can be either normal / disease,
        # (cancer adjacent slide). As such, we only assign the diagnostic labels
        # if the slide has cancerous code. Non cancerous/normal codes are excluded

        slide_labels = []
        for slide_name in slide_names:
            tissue_type_code = slide_name.split('.')[0].split('-')[3]
            tissue_type_code = int(tissue_type_code[:2])

            is_normal = (tissue_type_code >= 10) & (tissue_type_code <= 19)
            is_cancer = (tissue_type_code >=  1) & (tissue_type_code <=  9)
            assert is_normal or is_cancer

            case_info = slide_case_info[slide_name]
            slide_label = "normal"
            if is_cancer:
                slide_label = clinical_info.query(
                    f"case_id == '{case_info['case_id']}' & "
                    f"case_submitter_id == '{case_info['case_submitter_id']}'"
                )["primary_diagnosis"]
                slide_label = np.unique(slide_label)
                assert len(slide_label) == 1
                slide_label = slide_label[0]
            slide_labels.append(slide_label.lower())

        assert len(slide_names) == len(slide_labels)
        return list(zip(slide_names, slide_labels))


class CPTAC(object):

    @staticmethod
    def retrieve_labels(slide_names, clinical_file):
        with open(clinical_file, "r") as handle:
            info = json.load(handle)

        slide_info_list = []
        for subject in info:
            for slide_info in subject["specimens"]:
                if "slide_id" not in slide_info:
                    continue
                slide_info_list.append(
                    (slide_info["slide_id"], slide_info["tissue_type"])
                )

        slide_info_list = sorted(slide_info_list, key=lambda x: x[0])
        slide_info_dict = OrderedDict(slide_info_list)

        labels = [
            "normal",
            "blood",
        ]
        if "lscc" in clinical_file:
            labels.append("Squamous cell carcinoma")
        elif "luad" in clinical_file:
            labels.append("adenocarcinoma")

        output = []
        for v in slide_names:
            code = pathlib.Path(v).stem
            if code in slide_info_dict:
                vx = slide_info_dict[code]
                if vx not in labels:
                    continue
                output.append((v, vx))
        return output
