import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import APPNP


class My_APPNP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, K, alpha, dropout):
        super().__init__()
        self.dropout = dropout
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, num_classes)
        self.prop1 = APPNP(K, alpha)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)