import numpy as np
import json
import pathlib
import glob
import re
from collections import OrderedDict

class TCGA(object):
    def __init__(self):
        return

    def load(self, slide_dir):

        patterning = lambda x : re.sub('([\[\]])', '[\\1]', x)
        slide_path_list = glob.glob(patterning(slide_dir + '/*.npy'))
        slide_path_list.sort()
        assert len(slide_path_list) != 0, '`%s` is empty' % slide_dir

        code_list = [pathlib.Path(v).name for v in slide_path_list]
        code_list = [v.split('.')[0].split('-')[3] for v in code_list]
        tissue_type_code = np.array([int(v[:2]) for v in code_list])
        # print(np.unique(tissue_type_code, return_counts=True))

        # cancer is 01-09
        # normal is 10-19
        cancer_sel = (tissue_type_code >=  1) & (tissue_type_code <=  9)
        normal_sel = (tissue_type_code >= 10) & (tissue_type_code <= 19)
        sel = cancer_sel | normal_sel
        # sel = cancer_sel

        label_list = (cancer_sel[sel]).astype(np.int32)
        path_list  = (np.array(slide_path_list)[sel]).tolist()
        return path_list, label_list

class CPTAC(object):
    def __init__(self):
        self.set_info = {
            'CPTAC-LSCC' : 'dataset/CPTAC-LSCC.json',
            'CPTAC-LUAD' : 'dataset/CPTAC-LUAD.json',
        }

        return
    
    def load(self, slide_dir, data_code):
        patterning = lambda x : re.sub('([\[\]])', '[\\1]', x)
        slide_path_list = glob.glob(patterning(slide_dir + '/*.npy'))
        slide_path_list.sort()
        assert len(slide_path_list) != 0, '`%s` is empty' % slide_dir

        with open(self.set_info[data_code], 'r') as handle:
            info = json.load(handle)

        slide_info_list = []
        for subject in info:
            for slide_info in subject['specimens']:
                if 'slide_id' not in slide_info: 
                    continue
                slide_info_list.append(
                    (slide_info['slide_id'], 
                    slide_info['tissue_type'])
                )

        slide_info_list = sorted(slide_info_list, key=lambda x: x[0])
        slide_info_dict= OrderedDict(slide_info_list)

        label_map = {
            'normal' : 0,
            'blood'  : 0,
            'tumor'  : 1,
        }
        output = []
        for v in slide_path_list:
            code = pathlib.Path(v).stem
            if code in slide_info_dict:
                vx = slide_info_dict[code]
                if vx not in label_map: continue
                output.append((v, label_map[vx]))
        # print('%d/%d' % (len(output), len(slide_path_list)))
        return zip(*output)

class CRCX(object):
    def load(self, slide_dir):
        patterning = lambda x : re.sub('([\[\]])', '[\\1]', x)
        slide_path_list = glob.glob(patterning(slide_dir + '/*.npy'))
        slide_path_list.sort()
        assert len(slide_path_list) != 0, '`%s` is empty' % slide_dir
        code_path_dict = {pathlib.Path(v).stem.split('.')[0] : v \
                          for v in slide_path_list}
        assert len(code_path_dict) == len(slide_path_list)

        # ! Treat 1 case as label, is this right for MSI ?
        def adapt_info(save):
            slide_path_list = save['paths']
            label_path_list = save['targets']
            slide_path_list = [pathlib.Path(v).stem for v in slide_path_list]
            slide_path_list = [code_path_dict[v] for v in slide_path_list]
            return list(zip(slide_path_list, label_path_list))

        import torch
        split_info = []
        for fold_idx in range(1, 5):
            path = 'dataset/TCGA-CRX/MSI-ALL-FOLDS/'
            train = torch.load('%s/fold%d/%s' % (path, fold_idx, 'lib_train'))
            valid = torch.load('%s/fold%d/%s' % (path, fold_idx, 'lib_valid'))
            test  = torch.load('%s/fold%d/%s' % (path, fold_idx, 'lib_test'))
            split_info.append((
                (fold_idx, fold_idx),
                {
                    'train' : adapt_info(train), 
                    'valid' : adapt_info(valid), 
                    'test'  : adapt_info(test)
                }
            ))
        return split_info