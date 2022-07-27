import numpy as np
import json
import pathlib
import glob
import re
import os
from collections import OrderedDict
import pandas as pd

def find_path_with_ext_in_dir(root_dir, ext):
    path_list = []
    for full_path, subdirs, files in os.walk(root_dir):
        for file_path in files:
            file_ext = pathlib.Path(file_path).suffix
            if file_ext == ext:
                path_list.append(full_path + '/' + file_path)
    path_list.sort()
    return path_list

class TCGA(object):
    @staticmethod
    def load(slide_dir):

        patterning = lambda x : re.sub('([\[\]])', '[\\1]', x)
        slide_path_list = []
        for full_path, subdirs, files in os.walk(slide_dir):
            for file_path in files:
                file_ext = pathlib.Path(file_path).suffix
                if file_ext == '.npy':
                    slide_path_list.append(full_path + '/' + file_path)
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

class TCGA_MAF(object):
    @staticmethod
    def load(slide_dir_list, cancer_code):
        clinical_root_dir = 'dataset/'

        cancer_code = cancer_code.lower()
        if cancer_code == 'luad':
            clinical_path = '%s/tcga_luad.maf' % (clinical_root_dir)
        elif cancer_code == 'lusc':
            clinical_path = '%s/tcga_%s+crc.tsv' % (clinical_root_dir)
        # ! Hugo_Symbol is the mutation code
        ma_df = pd.read_csv(clinical_path, sep='\t')
        ma_df = ma_df[['Tumor_Sample_Barcode', 'Hugo_Symbol']]

        slide_path_list = []
        for slide_dir in slide_dir_list:            
            path_list = find_path_with_ext_in_dir(slide_dir, '.npy')
            slide_path_list.extend(path_list)
        assert len(slide_path_list) != 0, '`%s` is empty' % slide_dir

        ft_allcode_list = [pathlib.Path(v).stem for v in slide_path_list]
        ft_barcode_list = [v.split('.')[0] for v in ft_allcode_list]
        ft_barcode_list = ['-'.join(v.split('-')[:4])[:-1] for v in ft_barcode_list]
        ft_data = np.array([[ft_barcode_list[idx], ft_allcode_list[idx], 0] 
                            for idx in range(len(ft_barcode_list))])
        ft_df = pd.DataFrame(data=ft_data, columns=['Barcode', 'Full_Code', 'Label'])
        ft_df = ft_df.set_index('Barcode')

        mut_code_list = ['KRAS']
        mut_df = ma_df.loc[ma_df['Hugo_Symbol'].isin(mut_code_list)]
        mut_barcode_list = mut_df['Tumor_Sample_Barcode'].to_numpy().tolist()
        mut_barcode_list = ['-'.join(v.split('-')[:4])[:-1] for v in mut_barcode_list]

        shared_barcode = set.intersection(set(ft_barcode_list), set(mut_barcode_list))
        ft_df.loc[shared_barcode, 'Label'] = '1' 
        sample_path_list = ft_df['Full_Code'].to_numpy().tolist()
        sample_label_list = ft_df['Label'].to_numpy().tolist()
        sample_label_list = [int(v) for v in sample_label_list]
        print(np.unique(sample_label_list, return_counts=True))
        return sample_path_list, sample_label_list

# root_dir_list = [
#     'exp_output/feat/swav_resnet50/[TCGA-LUAD]-[512_256]-[mpp=0.50]',
#     # 'exp_output/feat/swav_resnet50/[TCGA-LUAD-Frozen]-[512_256]-[mpp=0.50]',
# ]
# _, label = TCGA_MAF.load(root_dir_list, 'luad')
# print(np.unique(label, return_counts=True))

class CPTAC(object):
    @staticmethod    
    def load(slide_dir, data_code):
        set_info = {
            'CPTAC-LSCC' : 'dataset/CPTAC-LSCC.json',
            'CPTAC-LUAD' : 'dataset/CPTAC-LUAD.json',
        }

        patterning = lambda x : re.sub('([\[\]])', '[\\1]', x)
        slide_path_list = []
        for full_path, subdirs, files in os.walk(slide_dir):
            for file_path in files:
                file_ext = pathlib.Path(file_path).suffix
                if file_ext == '.npy':
                    slide_path_list.append(full_path + '/' + file_path)
        slide_path_list.sort()
        assert len(slide_path_list) != 0, '`%s` is empty' % slide_dir

        with open(set_info[data_code], 'r') as handle:
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

# root_dir = '/root/storage_1/dataset/CPTAC-LSCC/'
# _, label = CPTAC.load(root_dir, 'CPTAC-LSCC')
# print(np.unique(label, return_counts=True))

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