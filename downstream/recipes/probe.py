import torch
import torch.nn.functional as F

from .base import ABCRecipe


class ProbeRecipe(ABCRecipe):

    @classmethod
    def train_step(cls, batch_data, run_info):
        # TODO: synchronize the attach protocol
        run_info, state_info = run_info

        # use 'ema' to add for EMA calculation, must be scalar!
        result_dict = {'EMA' : {}} 
        track_value = lambda name, value: result_dict['EMA'].update({name: value})

        ####
        model     = run_info['net']['desc']
        optimizer = run_info['net']['optimizer']

        ####
        feat_list = batch_data['features']
        label_list = batch_data['label']

        feat_list = (
            feat_list.to('cuda').type(torch.float32)
            if len(feat_list) > 0 else None
        )
        label_list = label_list.to('cuda').type(torch.int64)

        img_list = None
        if 'img' in batch_data:
            img_list = batch_data['img']
            img_list = img_list.to('cuda').type(torch.float32)

        ####
        model.train()
        model.zero_grad()  # not rnn so not accumulate

        logits = model(feat_list, img_list)
        loss = F.cross_entropy(logits, label_list)

        track_value('overall_loss', loss.cpu().item())
        # * gradient update

        # torch.set_printoptions(precision=10)
        loss.backward()
        optimizer.step()
        ####

        return result_dict

    @classmethod
    def valid_step(batch_data, run_info):
        run_info, state_info = run_info
        ####
        model = run_info['net']['desc']
        model.eval()  # infer mode

        ####
        feat_list = batch_data['features']
        label_list = batch_data['label']

        feat_list = (
            feat_list.to('cuda').type(torch.float32)
            if len(feat_list) > 0 else None
        )
        label_list = label_list.to('cuda').type(torch.int64)

        img_list = None
        if 'img' in batch_data:
            img_list = batch_data['img']
            img_list = img_list.to('cuda').type(torch.float32)

        # --------------------------------------------------------------
        with torch.no_grad():  # dont compute gradient
            logits = model(feat_list, img_list)

        result_dict = {
            'raw': {
                'label': label_list.cpu().numpy(),
                'prob': logits.cpu().numpy(),
            }
        }
        return result_dict

