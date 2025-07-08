import torch
from util import ops_cvae_pt


# Adapted from GRAND: https://github.com/THUDM/GRAND
def consis_loss(logps, temp, lam):
    ps = [torch.exp(p) for p in logps]
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p
    avg_p = sum_p /len(ps)
    # p2 = torch.exp(logp2)

    sharp_p = (torch.pow(avg_p, 1./temp) / torch.sum(torch.pow(avg_p, 1./temp), dim=1, keepdim=True)).detach()
    loss = 0.
    for p in ps:
        loss += torch.mean(( p -sharp_p).pow(2).sum(1))
    loss = loss /len(ps)
    return lam * loss

