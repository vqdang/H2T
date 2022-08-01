import numpy as np
import torch
import torch.nn as nn

from downstream.models.encoding import PositionalEncoding2DList

from ..backbone import hopfield as hf


class Transformer(nn.Module):
    def __init__(self,
            layers=[dict(
                state_dim=None,
                num_states=None,
                num_heads=1,
                drop_out=0.5
            )],
            proj_dim=None,
            embed_dim=None,
            num_types=None,):
        super(Transformer, self).__init__()
        # Normalize over last dimension 
        self.norm = nn.LayerNorm(embed_dim)

        self.proj = None
        if proj_dim is not None:
            self.proj = nn.Sequential(
                nn.Linear(embed_dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.ReLU()
            )
            embed_dim = proj_dim

        pool_layer_opt = layers[-1]
        self_attn_layers = layers[:-1]
        self.self_attn_layers = nn.ModuleList()
        for layer_opts in self_attn_layers:
            attn_encoder = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=layer_opts['num_heads'],
                dim_feedforward=layer_opts['state_dim'],
                dropout=layer_opts['drop_out'],
                batch_first=True
            )
            self.self_attn_layers.append(attn_encoder)
        
        # beta=None, scaling = 1.0 / sqrt(state_dim) (as in transformer)
        # output will be N x Q x Y, will project Q, K, V down to state_dim
        # (such as Nx2048 => Nxstate_dim) before starting matmul association
        # Q is set of weights of size [quantity x embedded dim]
        self.attn_pooling = hf.HopfieldPooling(
            input_size=embed_dim,  # Y
            # emdbedding size for each Q
            hidden_size=pool_layer_opt['state_dim'],
            # number of state pattern Q
            quantity=pool_layer_opt['num_states'],
            # number of attention heads
            num_heads=pool_layer_opt['num_heads'],
            dropout=pool_layer_opt['drop_out'],
            batch_first=True,               
        )
        self.clf = nn.Linear(
            embed_dim * pool_layer_opt['num_states'], num_types)
        self.position_encoding = PositionalEncoding2DList(embed_dim)
        return

    def forward(self, seq_feat_list, seq_pos_list, seq_mask_list=None):
        x = self.norm(seq_feat_list)

        encoded_positions = self.position_encoding(seq_pos_list)
        if self.proj is None:
            x = x + encoded_positions
        else:
            x = self.proj(x) + encoded_positions

        for layer in self.self_attn_layers:
            x = layer(x, src_key_padding_mask=seq_mask_list)

        # stored_pattern_padding_mask internally treated as key_padding_mask
        x = self.attn_pooling(x, stored_pattern_padding_mask=seq_mask_list)
        x = self.clf(x.flatten(1)) # N x Q x Y => N x Q * Y
        return x


if __name__ == '__main__':
    # for manually testing batch version
    torch.manual_seed(5)
    # model = GatedAttentionX(16, 4, True)
    # model = HopfieldModel(5, 6, 7, 8, 2)
    # model = VanillaPool(5, ['average', 'max', 'median'])
    # batch = 4
    # embed_dim = 5
    # #
    # model = LSTM_Model(embed_dim, 2)
    # seq_len = torch.randint(3, 4, (batch,))
    # # NxLxE
    # sample = [torch.rand([v, embed_dim]) for v in seq_len]
    # mask   = [torch.zeros([v]) for v in seq_len]
    # sample = pad_sequence(sample, batch_first=True, padding_value=0.0)
    # mask   = pad_sequence(  mask, batch_first=True, padding_value=1.0)
    # mask = mask == 1.0
    # print(mask)
    # print(sample)
    # model(sample, seq_len, mask)
    # print('here')

    # pos_list = np.array([[1 ,2], [1, 1], [3, 4]])
    # pos_list = torch.from_numpy(pos_list)
    # pe = PositionEncodingSine2D()
    # print(pe(pos_list, 4))

    n_ch = 256
    pos_list = np.array([
        [[1, 1], [2, 2], [3, 3]],
        [[4, 4], [5, 5], [6, 6]],
        [[3, 4], [1, 8], [2, 7]]
    ])
    pos_list = torch.from_numpy(pos_list)
    pe_flat = PositionalEncoding2DList(n_ch)
    out_flat = pe_flat(pos_list.type(torch.float32))
    print(out_flat.shape)

    pe_full = PositionalEncoding2DImage(n_ch)
    sample = np.random.random([1, 9, 9, 3])
    out_full = pe_full(torch.from_numpy(sample))

    out_flat = out_flat.numpy()
    out_full = out_full.numpy()

    pos_list = np.reshape(pos_list, [-1, 2])
    out_flat = np.reshape(out_flat, [-1, n_ch])
    for i, (x, y) in enumerate(pos_list):
        assert np.sum(out_flat[i] - out_full[x, y]) == 0
    print('here')

