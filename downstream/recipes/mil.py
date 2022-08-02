import torch
import torch.nn.functional as F

from .base import ABCRecipe


class MILRecipe(ABCRecipe):
    @classmethod
    def inference(cls):
        raise NotImplementedError

    @classmethod
    def train_step(cls, batch_data, run_info):
        # TODO: synchronize the attach protocol
        run_info, state_info = run_info

        # use 'ema' to add for EMA calculation, must be scalar!
        result_dict = {"EMA": {}}
        track_value = lambda name, value: result_dict["EMA"].update({name: value})

        ####
        model = run_info["net"]["desc"]
        optimizer = run_info["net"]["optimizer"]

        ####
        seq_feat_list, seq_pos_list, seq_len_list, seq_msk_list, label_list = batch_data

        seq_feat_list = seq_feat_list.to("cuda")  # Batch x Time step x Feat
        seq_pos_list = seq_pos_list.to("cuda").type(torch.float32)
        seq_len_list = seq_len_list.to("cuda")
        seq_msk_list = seq_msk_list.to("cuda")
        label_list = label_list.to("cuda").type(torch.int64)
        # label_list = label_list.to('cuda').type(torch.float32)

        ####
        model.train()
        model.zero_grad()  # not rnn so not accumulate

        loss = cls.training(
            model, [seq_feat_list, seq_pos_list, seq_len_list, seq_msk_list], label_list
        )

        track_value("overall_loss", loss.cpu().item())
        # * gradient update

        # torch.set_printoptions(precision=10)
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # TODO: expose this out
        optimizer.step()
        ####

        return result_dict

    @classmethod
    def valid_step(cls, batch_data, run_info):
        run_info, state_info = run_info
        ####
        model = run_info["net"]["desc"]
        model.eval()  # infer mode

        ####
        seq_feat_list, seq_pos_list, seq_len_list, seq_msk_list, label_list = batch_data

        seq_feat_list = seq_feat_list.to("cuda")  # Batch x Time step x Feat
        seq_pos_list = seq_pos_list.to("cuda").type(torch.float32)
        seq_len_list = seq_len_list.to("cuda")
        seq_msk_list = seq_msk_list.to("cuda")

        # label_list = label_list.to('cuda').type(torch.float32)
        label_list = label_list.to("cuda").type(torch.int64)

        # --------------------------------------------------------------
        logits = cls.inference(
            model, [seq_feat_list, seq_pos_list, seq_len_list, seq_msk_list]
        )

        # * Its up to user to define the protocol to process the raw output per step!
        result_dict = {  # protocol for contents exchange within `raw`
            "raw": {"label": label_list.cpu().numpy(), "prob": logits.cpu().numpy(),}
        }
        return result_dict


class TransformerRecipe(MILRecipe):
    @classmethod
    def training(cls, model, inputs, labels):
        model = model.train()
        seq_feat_list, seq_pos_list, seq_len_list, seq_msk_list = inputs
        logits = model(seq_feat_list, seq_pos_list, seq_msk_list)
        loss = F.cross_entropy(logits, labels)
        return loss

    @classmethod
    def inference(cls, model, inputs):
        model = model.eval()
        seq_feat_list, seq_pos_list, seq_len_list, seq_msk_list = inputs
        with torch.inference_mode():  # dont compute gradient
            logits = model(seq_feat_list, seq_pos_list, seq_msk_list)
        return logits


class CLAMRecipe(MILRecipe):
    @classmethod
    def training(cls, model, inputs, labels):
        model = model.train()
        seq_feat_list, seq_pos_list, seq_len_list, seq_msk_list = inputs
        logits, attentions, features, _ = model(
            seq_feat_list, seq_len_list, seq_msk_list
        )
        instance_loss = model.module.instance_loss(
            attentions, features, labels, seq_msk_list
        )
        loss = F.cross_entropy(logits, labels)
        loss += instance_loss
        return loss

    @classmethod
    def inference(cls, model, inputs):
        model = model.eval()
        seq_feat_list, seq_pos_list, seq_len_list, seq_msk_list = inputs
        with torch.inference_mode():  # dont compute gradient
            logits, attentions, features, _ = model(
                seq_feat_list, seq_len_list, seq_msk_list
            )
        return logits
