import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from misc.utils import center_pad_to_shape, cropping_center
from collections import OrderedDict


def upsample2x(feat):
    return F.interpolate(
        feat, scale_factor=2, mode="bilinear", align_corners=False
    )

def crop_op(x, cropping, data_format="NCHW"):
    """Center crop image
    Args:
        x: input image
        cropping: the substracted amount
        data_format: choose either `NCHW` or `NHWC`
    """
    crop_t = cropping[0] // 2
    crop_b = cropping[0] - crop_t
    crop_l = cropping[1] // 2
    crop_r = cropping[1] - crop_l
    if data_format == "NCHW":
        x = x[:, :, crop_t:-crop_b, crop_l:-crop_r]
    else:
        x = x[:, crop_t:-crop_b, crop_l:-crop_r, :]
    return x

def train_step(batch_data, run_info):
    # TODO: synchronize the attach protocol
    run_info, state_info = run_info

    # use 'ema' to add for EMA calculation, must be scalar!
    result_dict = {'EMA': {}} 
    track_value = lambda name, value: result_dict['EMA'].update({name: value})  # noqa

    ####
    model     = run_info['net']['desc']
    optimizer = run_info['net']['optimizer']

    ####
    img_list, msk_list = batch_data

    img_list = img_list.to('cuda').type(torch.float32)
    msk_list = msk_list.to('cuda').type(torch.int64)
    img_list = img_list.permute(0, 3, 1, 2)  # to NCHW

    ####
    model.train()
    model.zero_grad()  # not rnn so not accumulate

    logit_list = model(img_list)

    loss = F.cross_entropy(logit_list, msk_list)

    # ignore_mask = (msk_list != 2).type(torch.float32)
    # msk_list = (msk_list == 1).type(torch.int64)
    # loss = F.cross_entropy(logit_list, msk_list, reduction='none')
    # loss = torch.mean(loss * ignore_mask)

    # loss = F.cross_entropy(logit_list, msk_list, 
    #             weight=torch.FloatTensor([1.0, 1.0, 5.0]).to('cuda'))

    track_value('overall_loss', loss.cpu().item())
    # * gradient update

    loss.backward()
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
    img_list, msk_list = batch_data
    
    img_list = img_list.to('cuda').type(torch.float32) 
    msk_list = msk_list.to('cuda').type(torch.int64)
    img_list = img_list.permute(0, 3, 1, 2) # to NCHW

    # --------------------------------------------------------------
    with torch.no_grad(): # dont compute gradient
        logit_list = model(img_list)
        logit_list = logit_list.permute(0, 2, 3, 1) # to NHWC
        prob_list = F.softmax(logit_list, -1)

    # * Its up to user to define the protocol to process the raw output per step!

    # * Its up to user to define the protocol to process the raw output per step!
    result_dict = { 'raw':# protocol for contents exchange within `raw`
        {
            'img': img_list,
            'true' : {'op' : msk_list.cpu().numpy() },
            'pred' : {'op' : prob_list.cpu().numpy()} ,
        }
    }
    return result_dict

####
def infer_step(batch_data, model):

    ####
    model.eval() # infer mode

    ####
    img_list = batch_data
    
    img_list = img_list.to('cuda').type(torch.float32) 
    img_list = img_list.permute(0, 3, 1, 2) # to NCHW

    # --------------------------------------------------------------
    with torch.no_grad(): # dont compute gradient
        logit_list = model(img_list)
        logit_list = logit_list.permute(0, 2, 3, 1) # to NHWC
        prob_list = F.softmax(logit_list, -1)

        prob_list = prob_list.permute(0, 3, 1, 2) # to NCHW
        prob_list = upsample2x(prob_list)
        prob_list = crop_op(prob_list, [512, 512])
        prob_list = prob_list.permute(0, 2, 3, 1) # to NHWC
        # pred_list = torch.argmax(prob_list, dim=-1)

    # * Its up to user to define the protocol to process the raw output per step!
    prob_list = prob_list.cpu().numpy()
    return prob_list
    
####
def proc_cum_epoch_output(runner_name, epoch_data):

    # TODO: add auto populate from main state track list
    track_dict = {'scalar': {},  "image": {}}
    def track_value(name, value, vtype): return track_dict[vtype].update({name: value})


    from functools import reduce
    import operator
    def get_from_nested_dict(nested_dict, nested_key_list):
        return reduce(operator.getitem, nested_key_list, nested_dict)
    def flatten_dict_hierarchy(nested_key_list, raw_data):
        output_list = []
        for step_output in raw_data:
            step_output = get_from_nested_dict(step_output, nested_key_list)
            step_output = np.split(step_output, step_output.shape[0], axis=0)
            output_list.extend(step_output)
        output_list = [np.squeeze(v) for v in output_list]
        return output_list

    ####
    cum_stat_dict = epoch_data[1]

    for target_name, cum_stat in cum_stat_dict.items():
        for type_id in [1,2]:
            nr_pixels    = cum_stat['nr_pixels-%d' % type_id]
            over_inter   = cum_stat['over_inter-%d' % type_id]
            over_total   = cum_stat['over_total-%d' % type_id]
            over_correct = cum_stat['over_correct-%d' % type_id]
            acc  = over_correct/ nr_pixels
            dice = 2 * over_inter / (over_total + 1.0e-8)
            track_value('%s-accu-%d' % (target_name, type_id),  acc, "scalar")
            track_value('%s-dice-%d' % (target_name, type_id), dice, "scalar")
    # calculate average dice score for foreground prediction

    return track_dict

####
from run_utils.callbacks import BaseCallbacks

class ProcStepRawOutput(BaseCallbacks):
    def run(self, state, event):   
        ####
        def _dice_info(true, pred, label):
            true = np.array(true == label, np.int32)
            pred = np.array(pred == label, np.int32)
            inter = np.sum(pred * true, axis=(1, 2)) # collapse HW 
            total = np.sum(pred + true, axis=(1, 2)) # collapse HW 
            return inter, total

        # tissue_type
        def get_batch_stat(patch_prob, patch_true, cum_dict, target_name):
            patch_true = np.squeeze(patch_true) # ! may be wrong for n=1
            patch_prob = np.squeeze(patch_prob)
            patch_pred = np.argmax(patch_prob, axis=-1)

            n, h, w = patch_prob.shape[:3]
            for type_id in [1, 2]:
                patch_size = np.array([h * w for i in range(n)])                
                inter, total = _dice_info(patch_true, patch_pred, type_id)
                correct = np.sum(patch_true == patch_pred, axis=(1, 2))
                cum_dict['over_inter-%d' % type_id]   += np.sum(inter)
                cum_dict['over_total-%d' % type_id]   += np.sum(total)
                cum_dict['over_correct-%d' % type_id] += np.sum(correct)
                cum_dict['nr_pixels-%d' % type_id]    += np.sum(patch_size)
            return cum_dict

        step_output = state.step_output['raw']
        step_pred_output = step_output['pred']
        step_true_output = step_output['true']
        target_name_list = list(step_pred_output.keys())
        
        state_cum_output = state.epoch_accumulated_output
        # custom init and protocol
        if state.curr_epoch_step == 0:
            stat_list = []
            sub_stat_list = ['over_inter', 'over_total', 'over_correct', 'nr_pixels']
            for type_id in [1, 2]:
                for stat_name in sub_stat_list:
                    stat_list.append('%s-%d' % (stat_name, type_id))

            step_cum_stat_dict = {k: {s : 0 for s in stat_list} 
                            for k in target_name_list}
            state_cum_output = [[], step_cum_stat_dict]
            state.epoch_accumulated_output = state_cum_output
        state_cum_output = state.epoch_accumulated_output
        #

        # edit by reference also, variable is a reference, not a deep copy
        step_cum_stat_dict = state_cum_output[1]
        for target_name in target_name_list:
            new_cum_dict = get_batch_stat(
                                step_pred_output[target_name], 
                                step_true_output[target_name], 
                                step_cum_stat_dict[target_name], 
                                target_name)
            step_cum_stat_dict[target_name] = new_cum_dict

        # accumulate the batch basing on % such that not all dataset is stored
        # this will be useful for viz later
        # if np.random.uniform(0.1, 1.0) <= 0.01:    
        #     state_cum_output[0].append(step_output)
        state_cum_output[1] = step_cum_stat_dict

        return