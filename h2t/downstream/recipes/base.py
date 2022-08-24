from abc import ABC

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics


class ABCRecipe(ABC):

    @classmethod
    def train_step(cls, batch_data, run_info):
        raise NotImplementedError

    @classmethod
    def valid_step(cls, batch_data, run_info):
        raise NotImplementedError

    @classmethod
    def process_accumulated_step_output(cls, runner_name, raw_data, save_path=None):
        # TODO: add auto populate from main state track list
        track_dict = {'scalar': {}, 'conf_mat': {},  "array": {}}

        def track_value(name, value, vtype):
            return track_dict[vtype].update({name: value})

        logit = np.concatenate([v["prob"] for v in raw_data])
        true = np.concatenate([v["label"] for v in raw_data])

        logit = np.squeeze(logit)
        true = np.squeeze(true)
        num_classes = len(np.unique(true))

        loss = F.cross_entropy(
            torch.from_numpy(logit),
            torch.from_numpy(true), reduction='none')
        prob = F.softmax(torch.from_numpy(logit), -1).cpu().numpy()
        loss = loss.numpy()
        loss = loss.mean()

        if num_classes == 2:
            auroc = metrics.roc_auc_score(true, prob[:, -1])
            track_value('auroc', auroc, 'scalar')

        ap_list = []
        class_uids = np.unique(true)
        # ! assume full list and values are contiguous
        for idx, class_uid in enumerate(class_uids):
            ap = metrics.average_precision_score(
                    (true == class_uid).astype(np.int32), prob[:, idx])
            ap_list.append(ap)
            track_value(f'AP-{class_uid}', ap, 'scalar')
        track_value('mAP', np.mean(ap_list), 'scalar')

        track_value('loss', loss.item(), 'scalar')
        track_dict['array']['true'] = true
        track_dict['array']['pred'] = prob
        return track_dict

    @staticmethod
    def recipe(model_code):
        from h2t.downstream.recipes.mil import CLAMRecipe, TransformerRecipe
        from h2t.downstream.recipes.probe import ProbeRecipe
        if "clam" in model_code:
            return CLAMRecipe
        elif "transformer" in model_code:
            return TransformerRecipe
        elif "probe" in model_code:
            return ProbeRecipe
        else:
            assert False    
