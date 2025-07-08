import torch
import torch.nn.functional as F


def feature_tensor_normalize(feature):
    rowsum = torch.div(1.0, torch.sum(feature, dim=1))
    rowsum[torch.isinf(rowsum)] = 0.
    feature = torch.mm(torch.diag(rowsum), feature)
    return feature


def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return (BCE + KLD) / x.size(0)
