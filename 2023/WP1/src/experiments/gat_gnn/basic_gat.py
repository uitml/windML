import torch
import torch.nn as nn

from torch_geometric.nn import GATConv
from model_utils import DirGNNConv

class Mini_gnn(nn.Module):
    def __init__(self, dataset):
        super().__init__()

        conv1 = GATConv(dataset.num_features, 12, 12)
        conv2 = GATConv(144, dataset.num_classes, 1)

        self.fc = nn.Linear(144, dataset.num_classes)

        self.conv1 = DirGNNConv(conv1)
        self.conv2 = DirGNNConv(conv2)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = nn.functional.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index, edge_attr)

        return x