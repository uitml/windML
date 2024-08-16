import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv
from model_utils import DirGNNConv


class Mini_gnn(nn.Module):
    def __init__(self, dataset):
        super().__init__()

        conv1 = GCNConv(dataset.num_features, 64)
        conv2 = GCNConv(64, 100)


        self.conv1 = DirGNNConv(conv1)
        self.conv2 = DirGNNConv(conv2)
        self.fc = nn.Linear(100, dataset.num_classes)

    def forward(self, x, edge_index, edge_weight=None):

        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = nn.functional.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index, edge_weight)
        x = x.relu()
        x = nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)

        return x
