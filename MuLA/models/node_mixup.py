import torch.nn as nn

from util.ops_al import get_basic_model


class NodeMixup(nn.Module):

    def __init__(self, args):
        super(NodeMixup, self).__init__()
        self.model = get_basic_model(args)

    def forward(self, X, A, mixup_dict=None):
        logits = self.model(X, A)
        eq_mixup_logits = None
        neq_mixup_logits = None
        if mixup_dict is not None:
            eq_mixup_x, neq_mixup_x, mixup_adj = mixup_dict['eq_mixup_x'], mixup_dict['neq_mixup_x'], mixup_dict['mixup_adj']
            if eq_mixup_x.shape[0] > 0:
                # eq_mixup_logits = self.model(eq_mixup_x, adj=mixup_adj[0], edge_weight=mixup_adj[1])
                eq_mixup_logits = self.model(eq_mixup_x, mixup_adj)
            if neq_mixup_x.shape[0] > 0:
                # print(f"neq_mixup_x shape: {neq_mixup_x.shape}")
                neq_mixup_logits = self.model(neq_mixup_x, mixup_dict['E'])

        return logits, eq_mixup_logits, neq_mixup_logits

    def get_embeddings(self):
        return self.model.hid_list


