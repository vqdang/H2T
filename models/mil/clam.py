import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class GatedAttention(nn.Module):
    def __init__(self, L=1024, D=256, dropout=0.25, num_classes=1):
        super(GatedAttention, self).__init__()

        dropout = dropout if dropout is not None else 0.0

        self.attention_a = nn.Sequential(
            nn.Linear(L, D), nn.Tanh(), nn.Dropout(dropout)
        )
        self.attention_b = nn.Sequential(
            nn.Linear(L, D), nn.Sigmoid(), nn.Dropout(dropout)
        )

        self.attention_c = nn.Linear(D, num_classes)

    def forward(self, seq_feat_list, seq_len_list, seq_mask_list=None):
        """
        seq_feat_list : (N,L,E) Batch Size x Sequence Length x Embedding Size
        seq_mask_list : (N,L) Mask to exclude position i-th in sequence
                        out from calculation (1 or True to flag it),
                        usually used to treat padded elements.
        """
        input = seq_feat_list.permute(1, 0, 2)  # NxLxE => LxNxE

        a = self.attention_a(input)  # LxNxD
        b = self.attention_b(input)  # LxNxD

        a = a.permute(1, 0, 2)  # NxLxD
        b = b.permute(1, 0, 2)  # NxLxD

        # ! dot product not matmul in CLAM repos
        # https://github.com/mahmoodlab/CLAM/blob/5efe3ea57c4ac39e3ec8971fde4e508dfeea005e/models/model_clam.py#L62
        A = a * b
        A = self.attention_c(A)  # N x L x n_classes

        if seq_mask_list is not None:
            A.masked_fill_(seq_mask_list[..., None], 0.0)

        return A


class CLAM_SB(nn.Module):
    def __init__(
        self,
        mode="small",
        dims=[1024, 512, 256],
        dropout=0.25,
        top_k_samples=8,
        num_types=2,
        instance_loss_fn=nn.CrossEntropyLoss(),
        subtyping=False,
    ):
        super(CLAM_SB, self).__init__()
        # self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}

        num_classes = num_types
        self.compress = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.attention = GatedAttention(
            L=dims[1], D=dims[2], dropout=dropout, num_classes=1
        )
        self.classifiers = nn.Linear(dims[1], num_classes)

        instance_classifiers = [nn.Linear(dims[1], 2) for _ in range(num_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)

        self.top_k_samples = top_k_samples
        self.instance_loss_fn = instance_loss_fn
        self.num_classes = num_classes
        self.subtyping = subtyping
        initialize_weights(self)

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    # instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    # instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, seq_feat_list, seq_len_list, seq_mask_list=None):
        """
        seq_feat_list : (N,L,E) Batch Size x Sequence Length x Embedding Size
        seq_mask_list : (N,L) Mask to exclude position i-th in sequence
                        out from calculation (1 or True to flag it),
                        usually used to treat padded elements.
        """
        # N x L x E
        h = self.compress(seq_feat_list)
        # N x L x #classes
        A = self.attention(h, seq_len_list, seq_mask_list)

        # flush filled padded instances
        # so that they will have zero contributions
        if seq_mask_list is not None:
            A.masked_fill_(seq_mask_list[..., None], float("-inf"))

        # L x #classes x N
        A = A.permute(0, 2, 1)
        A = F.softmax(A, dim=-1)  # softmax over N

        # batch matmul (matmul over the dim 0th)
        # #classes x L matmul L x E => #classes x E
        M = torch.bmm(A, h)

        logits = self.classifiers(M)[:, 0]
        return logits, A, h, M

    def instance_loss(self, seq_attns, seq_feats, seq_labels, seq_masks=None):
        def pseudo_instance_labels(
            seq_attns, seq_feats, seq_masks, only_negative=False
        ):
            batch_size = seq_feats.shape[0]

            seq_attns_ = seq_attns
            # select top-k positive and negative
            top_k_pos = torch.topk(seq_attns_, self.top_k_samples, dim=-1, largest=True)
            pos_labels = torch.full((self.top_k_samples,), 1)

            # negative will collide with padded values (0.0)
            # so we will duplicate A and flip these values to inf first
            if seq_masks is not None:
                # N x L to N x 1 x L
                seq_masks = seq_masks[..., None].permute(0, 2, 1)
                seq_attns_ = seq_attns.detach().clone()
                seq_attns_.masked_fill_(seq_masks, float("inf"))
            top_k_neg = torch.topk(
                seq_attns_, self.top_k_samples, dim=-1, largest=False
            )
            neg_labels = torch.full((self.top_k_samples,), 0)

            # print(only_negative)
            # print(seq_attns)

            if only_negative:
                indices = top_k_neg[-1]
                inst_labels = neg_labels
            else:
                indices = torch.cat([top_k_pos[-1], top_k_neg[-1]], dim=-1)
                inst_labels = torch.cat([pos_labels, neg_labels])
            # [:, 0] to remove singleton from attention
            # (N x #classes x L but #classes is always 1)
            indices = indices[:, 0]

            # very tricky to use torch.gather when dims are not the same
            # (due to the existence of the feature dimension), use loop
            # for now
            inst_features = torch.cat([seq_feats[i, v] for i, v in enumerate(indices)])
            inst_labels = torch.cat([inst_labels] * batch_size)

            # print(indices)
            # print(inst_labels)
            # print('\n')

            return inst_features, inst_labels

        inst_loss = 0.0
        for seq_label in range(self.num_classes):
            sel = seq_label == seq_labels
            seq_attns_ = seq_attns[sel]
            seq_feats_ = seq_feats[sel]
            seq_masks_ = seq_masks[sel]

            for idx, clf_ in enumerate(self.instance_classifiers):
                only_negative = idx != seq_label

                # entire batch is of different class
                if only_negative or seq_feats_.shape[0] == 0:
                    seq_attns_ = seq_attns
                    seq_feats_ = seq_feats
                    seq_masks_ = seq_masks

                (inst_features, inst_labels) = pseudo_instance_labels(
                    seq_attns_,
                    seq_feats_,
                    seq_masks=seq_masks_,
                    only_negative=only_negative,
                )
                inst_logits = clf_(inst_features)
                inst_labels = inst_labels.to("cuda").type(torch.int64)
                label_loss = F.cross_entropy(inst_logits, inst_labels)
                inst_loss += label_loss

        if self.subtyping:
            return inst_loss / 2.0
        return inst_loss


if __name__ == "__main__":
    import numpy as np
    from torch.nn.utils.rnn import (
        pack_padded_sequence,
        pad_packed_sequence,
        pad_sequence,
    )

    torch.manual_seed(5)
    batch = 4
    embed_dim = 5
    seq_len = torch.randint(4, 6, (batch,))
    labels = torch.randint(0, 2, (batch,))
    # for manually testing batch version
    model = CLAM_SB(dims=[embed_dim, 512, 256], top_k_samples=2, num_types=3)
    model = model.to("cuda")

    #
    # NxLxE
    sample = [torch.rand([v, embed_dim]) for v in seq_len]
    mask = [torch.zeros([v]) for v in seq_len]
    sample = pad_sequence(sample, batch_first=True, padding_value=0.0)
    mask = pad_sequence(mask, batch_first=True, padding_value=1.0)
    mask = mask == 1.0
    mask[0][0] = True  # enfore instance 0th of item subject 0th to be removed
    # print(mask)
    # print(sample)

    mask = mask.to("cuda")
    sample = sample.to("cuda")
    seq_len = seq_len.to("cuda")
    logits, attentions, features, _ = model(sample, seq_len, mask)
    loss = model.instance_loss(attentions, features, labels, mask)

    sample_labels = [
        torch.from_numpy(np.array([0, 0, 0, 0])),
        torch.from_numpy(np.array([1, 1, 1, 1])),
        torch.from_numpy(np.array([2, 2, 2, 2])),
        torch.from_numpy(np.array([1, 1, 1, 2])),
        torch.from_numpy(np.array([1, 0, 1, 0])),
        torch.from_numpy(np.array([1, 0, 1, 2])),
    ]
    for labels in sample_labels:
        mask = mask.to("cuda")
        sample = sample.to("cuda")
        seq_len = seq_len.to("cuda")
        logits, attentions, features, _ = model(sample, seq_len, mask)
        loss = model.instance_loss(attentions, features, labels, mask)

    print("here")
