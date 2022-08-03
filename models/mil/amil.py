import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttention(nn.Module):
    def __init__(
        self,
        embed_dim=None,  # or L as in the paper
        hidden_dim=None,  # or D as in the paper
        projection_dim=None,
        activation=False,  # to add relu after attention aggregation (before clf)
        nr_types=2,
    ):
        """
        Batch version
        Attention MIL, Ilse & Max Welling
        """
        super(GatedAttention, self).__init__()

        # Normalize over last dimension
        if projection_dim is not None:
            self.linear = nn.Sequential(
                nn.LayerNorm(embed_dim), nn.Linear(embed_dim, hidden_dim)
            )
        else:
            projection_dim = embed_dim
            self.linear = nn.Sequential(
                nn.LayerNorm(embed_dim),
            )

        K_dim = 1
        self.linear_u = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, hidden_dim)
        )
        self.linear_v = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, hidden_dim)
        )
        self.linear_a = nn.Linear(hidden_dim, K_dim)

        self.activation = activation
        self.clf = nn.Linear(projection_dim, nr_types)
        return

    def forward(self, seq_feat_list, seq_len_list, seq_mask_list=None):
        """
        input : (N,L,E) Batch Size x Sequence Length x Embedding Size
        mask  : (N,L)   Mask to exclude position i-th in sequence
                        out from calculation (1 or True to flag it),
                        usually used to treat padded elements
        """

        input = seq_feat_list.permute(1, 0, 2)  # NxLxE => LxNxE

        A_V = self.linear_u(input)  # LxNxD
        A_U = self.linear_v(input)  # LxNxD
        A = self.linear_a(A_U * A_V)  # LxNx1
        A = A.transpose(1, 0)[..., 0]  # NxL
        if seq_mask_list is not None:
            if seq_mask_list.dtype == torch.bool:
                A.masked_fill_(seq_mask_list, float("-inf"))
        A = F.softmax(A, -1).unsqueeze(1)  # Nx1xL

        x = self.linear(input)
        x = x.permute(1, 0, 2)  # LxNxE => NxLxE
        M = torch.bmm(A, x)[:, 0]  # batch N matmul L * LxE  => NxE
        M = F.relu(M) if self.activation else M

        out = self.clf(M)
        return out
