from collections import OrderedDict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from misc.utils import center_pad_to_shape, cropping_center
from sklearn import metrics

from model.utils import find_optimal_roc_cutoff
from model.viz_utils import plot_roc_curve

####
def train_step(batch_data, run_info):
    # TODO: synchronize the attach protocol
    run_info, state_info = run_info

    # use 'ema' to add for EMA calculation, must be scalar!
    result_dict = {'EMA' : {}} 
    track_value = lambda name, value: result_dict['EMA'].update({name: value})

    ####
    model     = run_info['net']['desc']
    optimizer = run_info['net']['optimizer']

    ####
    seq_feat_list, seq_pos_list, seq_len_list, seq_msk_list, label_list = batch_data
    
    seq_feat_list = seq_feat_list.to('cuda') # Batch x Time step x Feat
    seq_pos_list = seq_pos_list.to('cuda').type(torch.float32)
    seq_len_list = seq_len_list.to('cuda')
    seq_msk_list = seq_msk_list.to('cuda')
    label_list = label_list.to('cuda').type(torch.int64)
    # label_list = label_list.to('cuda').type(torch.float32)

    ####
    model.train() 
    model.zero_grad() # not rnn so not accumulate

    # ! clam
    logits, attentions, features, _ = model(
        seq_feat_list, seq_len_list, seq_msk_list
    )
    instance_loss = model.module.instance_loss(
        attentions, features, label_list, seq_msk_list
    )
    loss = F.cross_entropy(logits, label_list)
    loss += instance_loss

    # ! tranformer
    # logits = model(seq_feat_list, seq_pos_list, seq_msk_list)
    # loss = F.cross_entropy(logits, label_list)

    track_value('overall_loss', loss.cpu().item())
    # * gradient update

    # torch.set_printoptions(precision=10)
    loss.backward()
    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # TODO: expose this out
    optimizer.step()
    ####

    return result_dict

####
def valid_step(batch_data, run_info):
    run_info, state_info = run_info
    ####
    model = run_info['net']['desc']
    model.eval() # infer mode

    ####
    seq_feat_list, seq_pos_list, seq_len_list, seq_msk_list, label_list = batch_data
    
    seq_feat_list = seq_feat_list.to('cuda') # Batch x Time step x Feat
    seq_pos_list = seq_pos_list.to('cuda').type(torch.float32)
    seq_len_list = seq_len_list.to('cuda')
    seq_msk_list = seq_msk_list.to('cuda')

    # label_list = label_list.to('cuda').type(torch.float32)
    label_list = label_list.to('cuda').type(torch.int64)

    # --------------------------------------------------------------
    with torch.no_grad(): # dont compute gradient
        # ! transformer
        # logits = model(seq_feat_list, seq_pos_list, seq_msk_list)

        # ! clam
        logits, attentions, features, _ = model(
            seq_feat_list, seq_len_list, seq_msk_list
        )

    # * Its up to user to define the protocol to process the raw output per step!
    result_dict = { # protocol for contents exchange within `raw`
        'raw': {
            'label' : label_list.cpu().numpy(),
            'prob'  : logits.cpu().numpy(),
        }
    }
    return result_dict
  
#
def proc_cum_step_output(runner_name, raw_data, save_path=None):

    # TODO: add auto populate from main state track list
    track_dict = {'scalar': {}, 'conf_mat': {},  "array": {}}

    def track_value(name, value, vtype):
        return track_dict[vtype].update({name: value})

    logit = np.squeeze(np.array(raw_data['prob']))
    true = np.squeeze(np.array(raw_data['label']))
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
