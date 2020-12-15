import os, sys
import numpy as np

import torch as tc
import torch.nn as nn

class Dummy(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, x, y=None):
        if y is not None:
            x['logph'] = x['logph_y'] ##TODO: better way?
        return x


def dist_mah(xs, cs, Ms, sqrt=True):
    diag = True if len(Ms.size()) == 2 else False
    assert(diag)
    assert(xs.size() == cs.size())
    assert(xs.size() == Ms.size())

    diff = xs - cs
    dist = diff.mul(Ms).mul(diff).sum(1)
    if sqrt:
        dist = dist.sqrt()
    return dist


def neg_log_prob(yhs, yhs_logvar, ys, var_min=1e-16):

    d = ys.size(1)
    yhs_var = tc.max(yhs_logvar.exp(), tc.tensor(var_min, device=yhs_logvar.device))
    loss_mah = 0.5 * dist_mah(ys, yhs, 1/yhs_var, sqrt=False)
    # if not all(loss_mah >= 0):
    #     print('loss_mah', loss_mah)
    #     print('ys', ys)
    #     print('yhs', yhs)
    #     print('yhs_var', yhs_var)
    #     print('yhs_logvar', yhs_logvar)
    assert(all(loss_mah >= 0))
    loss_const = 0.5 * np.log(2.0 * np.pi) * d
    loss_logdet = 0.5 * yhs_logvar.sum(1)
    loss = loss_mah + loss_logdet + loss_const

    return loss

