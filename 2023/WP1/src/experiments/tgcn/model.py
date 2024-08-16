import torch
import torch.nn as nn

from tgcn import TGCN





class TGCN_regression(nn.Module):
    def __init__(self, adj, tgcn_hidden_dims=128, out_dim=3):
        super().__init__()

        self.tgcn = TGCN(adj, tgcn_hidden_dims)

        self.dense = nn.Linear(tgcn_hidden_dims, out_dim)

    def forward(self, x):
        x = self.tgcn(x)
        x.relu()
        x = self.dense(x)

        return x


if __name__ == '__main__':
    pass