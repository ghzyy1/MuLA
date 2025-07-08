import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, Linear, MixHopConv


class MixHop(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.conv1 = MixHopConv(input_dim, 60, powers=[0, 1, 2])
        self.norm1 = BatchNorm(3 * 60)

        self.conv2 = MixHopConv(3 * 60, 60, powers=[0, 1, 2])
        self.norm2 = BatchNorm(3 * 60)

        self.conv3 = MixHopConv(3 * 60, 60, powers=[0, 1, 2])
        self.norm3 = BatchNorm(3 * 60)

        self.lin = Linear(3 * 60, num_classes)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.7, training=self.training)

        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.dropout(x, p=0.9, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.dropout(x, p=0.9, training=self.training)

        x = self.conv3(x, edge_index)
        x = self.norm3(x)
        x = F.dropout(x, p=0.9, training=self.training)

        return self.lin(x)