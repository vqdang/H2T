

class VanillaPool(nn.Module):
    def __init__(self, 
            embed_dim=None,
            aggregation_mode='avg',
            nr_types=2):
        """
        Batch version
        Attention MIL, Ilse & Max Welling
        """
        super(VanillaPool, self).__init__()
        assert isinstance(aggregation_mode, list)

        self.norm = nn.LayerNorm(embed_dim)
        self.aggregation_mode = aggregation_mode
        self.clf = nn.Linear(embed_dim * len(aggregation_mode), nr_types)
        return

    def forward(self, input, mask=None):
        """
        input : (N,L,E) Batch Size x Sequence Length x Embedding Size
        mask  : (N,L)   Mask to exclude position i-th in sequence 
                        out from calculation (1 or True to flag it),
                        usually used to treat padded elements
        """

        input = self.norm(input)

        if mask is not None:
            agg_list = []
            for mode in self.aggregation_mode:
                if mode == 'average':
                    dat = (input * mask[...,None]).sum(dim=1)
                    dat = dat / mask.sum(dim=1, keepdim=True)
                elif mode == 'max':
                    dat, _ = (input * mask[...,None]).max(dim=1)
                elif mode == 'median':
                    dat = []
                    for batch_idx in range(input.shape[0]):
                        # ! what will happen if batch size 1 ?
                        sub_dat = input[batch_idx][mask[batch_idx]]
                        sub_dat, _ = torch.median(sub_dat, dim=0)
                        dat.append(sub_dat)
                    dat = torch.stack(dat, dim=0)
                agg_list.append(dat)
        agg_list = torch.cat(agg_list, dim=-1)
        out = self.clf(agg_list)
        return out
