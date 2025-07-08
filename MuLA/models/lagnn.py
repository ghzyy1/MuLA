import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SGConv, GCN2Conv, ChebConv, APPNP

"""
    Note: 没有对网络层进行参数初始化
"""

class LAGCN(nn.Module):

    def __init__(self, num_concat, input_dim, hidden_dim, num_class, dropout):
        super(LAGCN, self).__init__()

        self.gcn_list = nn.ModuleList()
        for _ in range(num_concat):
            self.gcn_list.append(GCNConv(input_dim, hidden_dim, cached=True))
        self.gcn = GCNConv(num_concat*hidden_dim, num_class)
        self.dropout = dropout

    def forward(self, x_list, adj):
        hidden_list = []
        for k, con in enumerate(self.gcn_list):
            x = F.dropout(x_list[k], self.dropout, training=self.training)
            hidden_list.append(F.relu(con(x, adj)))
        x = torch.cat((hidden_list), dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn(x, adj)
        return x


class LAGAT(torch.nn.Module):

    def __init__(self, num_concat, input_dim, hidden_dim, num_head_1, num_head_2, num_class, dropout):
        super().__init__()
        self.dropout = dropout

        self.gat_list = nn.ModuleList()
        for _ in range(num_concat):
            self.gat_list.append(GATConv(input_dim, hidden_dim, heads=num_head_1, dropout=dropout))

        # On the Pubmed dataset, use heads=8 in conv2.
        self.gat = GATConv(hidden_dim*num_head_1*num_concat, num_class, heads=num_head_2, concat=False, dropout=dropout)

    def forward(self, x_list, edge_index):
        hidden_list = []
        for k, con in enumerate(self.gat_list):
            x = F.dropout(x_list[k], p=self.dropout, training=self.training)
            hidden_list.append(F.elu(con(x, edge_index)))

        x = torch.cat((hidden_list), dim=-1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat(x, edge_index)
        return x


class LASGC(torch.nn.Module):

    def __init__(self, num_concat, input_dim, num_class, K):
        super(LASGC, self).__init__()

        self.sgc_list = nn.ModuleList()
        for _ in range(num_concat):
            self.sgc_list.append(SGConv(input_dim, num_class, K=K))

        self.sgc = SGConv(num_class*num_concat, num_class, K=K)

    def forward(self, x_list, edge_index):
        hidden_list = []
        for k, con in enumerate(self.sgc_list):
            hidden_list.append(F.relu(con(x_list[k], edge_index)))

        x = torch.cat((hidden_list), dim=-1)
        x = self.sgc(x, edge_index)
        return x
    




class LAGCNII(torch.nn.Module):

    def __init__(self, num_concat, input_dim, hidden_dim, num_layer, num_class, alpha, theta, shared_weights=True, dropout=0.0):
        super(LAGCNII, self).__init__()
        self.dropout = dropout

        self.linear_list = torch.nn.ModuleList()
        for _ in range(num_concat):
            self.linear_list.append(Linear(input_dim, hidden_dim))

        self.gcnii_list = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.gcnii_list.append(GCN2Conv(hidden_dim*num_concat, alpha, theta, layer+1, shared_weights, normalize=False))

        self.linear = Linear(hidden_dim*num_concat, num_class)

    def forward(self, x_list, edge_index):
        hidden_list = []
        for k, con in enumerate(x_list):
            x = F.dropout(x_list[k], p=self.dropout, training=self.training)
            hidden_list.append(F.relu(self.linear_list[k](x)))

        h = h_0 = torch.cat((hidden_list), dim=-1)
        for gcnii in self.gcnii_list:
            h = F.dropout(h, self.dropout, training=self.training)
            h = gcnii(h, h_0, edge_index)
            h = h.relu()

        h = F.dropout(h, self.dropout, training=self.training)
        h = self.linear(h)

        return h
    
class LACheb(torch.nn.Module):
    def __init__(self, num_concat, input_dim, hidden_dim, num_class, num_hops, dropout):
        super(LACheb, self).__init__()
        self.cheb_list = nn.ModuleList()
        for _ in range(num_concat):
            self.cheb_list.append(ChebConv(input_dim, hidden_dim,  num_hops))
        self.cheb = GCNConv(num_concat*hidden_dim, num_class, num_hops)
        self.dropout = dropout

    def forward(self, x_list, edge_index):
        hidden_list = []
        for k, con in enumerate(self.cheb_list):
            x = F.dropout(x_list[k], self.dropout, training=self.training)
            hidden_list.append(F.relu(con(x, edge_index)))
        x = torch.cat((hidden_list), dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.cheb(x, edge_index)
        return x

class LAAPPNP(torch.nn.Module):
     def __init__(self, num_concat, input_dim, hidden_dim, num_class, K, alpha, dropout):
         super(LAAPPNP, self).__init__()
         self.dropout = dropout
         
         self.appnp_list = torch.nn.ModuleList()
         for _ in range(num_concat):
               self.appnp_list.append(APPNP(input_dim, hidden_dim, K, alpha))
         self.appnp = APPNP(num_concat*hidden_dim, num_class, K, alpha)
     
     def forward(self, x_list, edge_index):
        hidden_list = []
        for k, con in enumerate(self.appnp_list):
            x = F.dropout(x_list[k], self.dropout, training=self.training)
            hidden_list.append(F.relu(con(x, edge_index)))
        x = torch.cat((hidden_list), dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.appnp(x, edge_index)
        return x
