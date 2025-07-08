from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import SGConv


class SGC(nn.Module):

    def __init__(self, input_dim, num_classes, K):
        super(SGC, self).__init__()
        self.conv1 = SGConv(input_dim, num_classes, K=K, cached=True)

    def reset_parameters(self):
        self.conv1.reset_parameters()

    def forward(self, x, edge_index):
        print(f"Input x shape: {x.shape}")
        # print(f"Edge index shape: {edge_index.shape}")
        x = self.conv1(x, edge_index)
        print(f"Output after SGConv shape: {x.shape}")
        return F.log_softmax(x, dim=1)