import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from lstm_utils import LSTMState, flatten_states

# ! currently script_lnlstm `bidirectional` broken because it could not do padded
# ! sequence correctly, unless batch size == 1, also too slow
class LSTM_Model(nn.Module):
    def __init__(
        self,
        embed_dim=None,
        nr_types=None,
        #
        dropout=0.0,
        nr_layers=1,
        hidden_dim=None,
        bidirectional=False,
    ):
        super(LSTM_Model, self).__init__()

        self.norm = nn.LayerNorm(embed_dim)

        self.lstm_kwargs = {
            "input_size": embed_dim,
            "hidden_size": hidden_dim,
            "num_layers": nr_layers,
            "dropout": dropout,
            "bidirectional": False,
        }

        # ! non native version too slow!
        # self.lstm = script_lnlstm(**self.lstm_kwargs)
        self.lstm = nn.LSTM(**self.lstm_kwargs)

        # The linear layer that maps from hidden state space to output
        nr_direction = 2 if self.lstm_kwargs["bidirectional"] else 1
        hidden_size = (
            self.lstm_kwargs["num_layers"]
            * nr_direction
            * self.lstm_kwargs["hidden_size"]
        )

        self.norm_clf = nn.LayerNorm(hidden_size)
        self.clf = nn.Linear(hidden_size, nr_types)

    def get_init_hidden(self, batch_size):
        # https://discuss.pytorch.org/t/initialization-of-first-hidden-state-in-lstm-and-truncated-bptt/58384
        # https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
        # num_layers * num_directions, batch, hidden_size
        if self.lstm_kwargs["bidirectional"]:
            states = [
                [
                    LSTMState(
                        torch.zeros(batch_size, self.lstm_kwargs["hidden_size"]).to(
                            "cuda"
                        ),
                        torch.zeros(batch_size, self.lstm_kwargs["hidden_size"]).to(
                            "cuda"
                        ),
                    )
                    for _ in range(2)
                ]
                for _ in range(self.lstm_kwargs["num_layers"])
            ]
        else:
            states = [
                LSTMState(
                    torch.zeros(batch_size, self.lstm_kwargs["hidden_size"]).to("cuda"),
                    torch.zeros(batch_size, self.lstm_kwargs["hidden_size"]).to("cuda"),
                )
                for _ in range(self.lstm_kwargs["num_layers"])
            ]
            states = flatten_states(states)  # make compatible with native pytorch LSTM
        return states

    def forward(self, seq_feat_list, seq_len_list, seq_mask_list):
        #### NxLxE

        batch_size, _, _ = seq_feat_list.shape

        seq_feat_list = self.norm(seq_feat_list)  # NxLxE
        x = seq_feat_list.permute(1, 0, 2)  # NxLxE => LxNxE

        # reset the LSTM/RNN hidden state. Must be done before running a new batch.
        # Otherwise the LSTM will treat a new batch as a continuation of a sequence
        # hidden state for t = seq_len and cell state for t = seq_len

        # make compatible with native pytorch LSTM
        x = pack_padded_sequence(
            x, seq_len_list, batch_first=False, enforce_sorted=False
        )
        self.lstm.flatten_parameters()

        states = self.get_init_hidden(batch_size)
        x, out_state = self.lstm(
            x, states
        )  # output state, hidden state at last time step

        # if self.lstm_kwargs['bidirectional']:
        #     hidden_state, cell_state = double_flatten_states(out_state)
        # else:
        #     hidden_state, cell_state = flatten_states(out_state)

        # cell state stored the last hidden state in the sequence (after padding)
        # retrieve hidden state at last idx (actual last time step of the
        # output sequence) before padding
        # if want to verify, must indexing x with correct last seq len due to the padding
        # e.g (hidden_state[0,1] - x[1][seq_len_list[1]-1]).sum() should be 0

        # x = x.permute(1, 0, 2) # LxNxE => NxLxE
        # x = x[torch.arange(batch_size), seq_len_list-1] # NxE

        # * feed the aggregation to the clf

        # undo the packing operation
        # x, _ = pad_packed_sequence(x, batch_first=False)
        (
            hidden_state,
            cell_state,
        ) = out_state  # shape: num_layers * num_directions, batch, hidden_size
        x = hidden_state.permute(1, 0, 2)
        x = x.contiguous()
        x = x.view(x.shape[0], -1)

        # run through actual linear layer
        x = self.clf(self.norm_clf(x))
        return x
